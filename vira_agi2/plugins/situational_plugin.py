import asyncio
import logging
import requests
from datetime import datetime, timedelta
import vira4_6t
from typing import Dict, Any
import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)

try:
    from cuml.cluster import DBSCAN as CumlDBSCAN
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    logger.warning("cuml not available; falling back to basic situational processing.")

class Plugin:
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", None)
        self.api_keys = config.get("external_api_settings", {}).get("api_keys", {})
        self.enabled_sources = config.get("situational_settings", {}).get("enabled_sources", [])
        self.context = {}
        self.user_state = {
            "location": config.get("situational_settings", {}).get("default_location", "Spokane, Washington"),
            "goals": [],
            "preferences": {},
            "habits": {}
        }
        self.memory_bank = vira4_6t.memory_bank
        self.memory_async_lock = vira4_6t.memory_async_lock
        self.db_pool = vira4_6t.db_pool
        self.llama_model = vira4_6t.LLAMA_MODEL_NAME
        self.xp = cp if (self.device and self.device.type == "cuda" and cp.cuda.is_available()) else np
        self.weather_cache = {"data": None, "timestamp": None, "ttl": timedelta(minutes=30)}
        self.max_goals = config.get("situational_settings", {}).get("max_goals", 10)
        
        if CUML_AVAILABLE and self.xp is cp:
            self.dbscan = CumlDBSCAN(eps=1.0, min_samples=2)  # Adjusted eps for situational data
            logger.info("Initialized cuml.DBSCAN for situational clustering.")
        else:
            self.dbscan = None
            logger.info("Using basic situational processing (cuml unavailable or CPU mode).")
        
        logger.info(f"Situational plugin initialized with location: {self.user_state['location']}, device: {'GPU' if self.xp is cp else 'CPU'}")

    async def fetch_weather(self):
        now = datetime.now()
        if (self.weather_cache["data"] and 
            self.weather_cache["timestamp"] and 
            now - self.weather_cache["timestamp"] < self.weather_cache["ttl"]):
            logger.debug("Returning cached weather data.")
            return self.weather_cache["data"]
        
        api_key = self.api_keys.get("openweathermap")
        if not api_key:
            logger.error("No valid OpenWeatherMap API key provided.")
            return {"weather": "Weather data unavailable (no API key)", "timestamp": now.isoformat()}
        
        retries = 3
        for attempt in range(retries):
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={self.user_state['location']}&appid={api_key}&units=imperial"
                response = await asyncio.to_thread(requests.get, url, timeout=5)
                response.raise_for_status()
                data = response.json()
                weather = f"{data['weather'][0]['description'].capitalize()}, {data['main']['temp']}°F"
                result = {"weather": weather, "timestamp": now.isoformat()}
                self.weather_cache["data"] = result
                self.weather_cache["timestamp"] = now
                return result
            except (requests.exceptions.HTTPError, requests.exceptions.Timeout) as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"Error fetching weather after {retries} attempts: {e}")
                    return {"weather": f"Weather data unavailable ({e})", "timestamp": now.isoformat()}
            except Exception as e:
                logger.error(f"Unexpected error fetching weather: {e}")
                return {"weather": "Weather data unavailable (unknown error)", "timestamp": now.isoformat()}

    async def update_context(self, data: Dict[str, Any]):
        situational_data = []
        now = datetime.now()
        
        # Temporal integration
        next_checkin = None
        if vira4_6t.plugin_manager and "temporal_plugin" in vira4_6t.plugin_manager.plugins:
            try:
                checkin_result = await vira4_6t.plugin_manager.execute_specific_plugin(
                    "temporal_plugin", {"command": "suggest_checkin"}
                )
                next_checkin = checkin_result.get("suggested_checkin")
                if next_checkin:
                    self.context["next_checkin"] = next_checkin
                    logger.debug(f"Next check-in: {next_checkin}")
            except Exception as e:
                logger.error(f"Error fetching check-in suggestion: {e}")
        
        # Weather
        if "weather" in self.enabled_sources:
            weather_data = await self.fetch_weather()
            if weather_data:
                self.context.update(weather_data)
                situational_data.append({
                    "content": f"Weather in {self.user_state['location']}: {weather_data['weather']}",
                    "emotion_type": "neutral",
                    "emotion_intensity": 0
                })
                await vira4_6t.queue_memory_save(
                    content=situational_data[-1]["content"],
                    perspective="System",
                    emotion="neutral",
                    emotion_intensity=0,
                    context=f"Situational update at {weather_data['timestamp']}"
                )
        
        # Hippo integration
        if vira4_6t.plugin_manager and "hippo_plugin" in vira4_6t.plugin_manager.plugins:
            try:
                unresolved_result = await vira4_6t.plugin_manager.execute_specific_plugin(
                    "hippo_plugin", {"command": "unresolved"}
                )
                unresolved = unresolved_result.get("unresolved", [])
                if unresolved:
                    situational_data.extend(unresolved[:5])  # Limit to recent 5
                    cluster_summary = await self._summarize_clusters(unresolved)
                    self.context["cluster_summary"] = cluster_summary["summary"]
                    self.context["dominant_emotion"] = cluster_summary["dominant_emotion"]
                    logger.debug(f"Updated context with unresolved: {cluster_summary['summary']}")
            except Exception as e:
                logger.error(f"Error fetching unresolved issues: {e}")
        
        # Clustering
        if CUML_AVAILABLE and self.dbscan and self.xp is cp and len(situational_data) >= 2:
            embeddings = cp.stack([cp.asarray(vira4_6t.cached_embedder_encode((d["content"],), device=self.device)) 
                                 for d in situational_data])
            embeddings = (embeddings - cp.mean(embeddings, axis=0)) / cp.std(embeddings, axis=0)  # Normalize
            clusters = self.dbscan.fit_predict(embeddings)
            valid_clusters = clusters[clusters != -1]
            if valid_clusters.size > 0:
                cluster_counts = cp.bincount(valid_clusters)
                strongest_cluster = cp.argmax(cluster_counts).item()
                cluster_items = [situational_data[i] for i in range(len(situational_data)) if clusters[i] == strongest_cluster]
                self.context["situational_cluster"] = f"Cluster of {len(cluster_items)} items (e.g., {cluster_items[0]['content'][:50]}...)"
                logger.debug(f"Clustered: {self.context['situational_cluster']}")
        
        # Rhythm
        if vira4_6t.plugin_manager and "temporal_plugin" in vira4_6t.plugin_manager.plugins:
            try:
                rhythm_result = await vira4_6t.plugin_manager.execute_specific_plugin(
                    "temporal_plugin", {"command": "get_rhythm"}
                )
                user_rhythm = rhythm_result.get("rhythms", {}).get("user", {})
                if "mean_activity_gap_hours" in user_rhythm:
                    self.context["activity_rhythm"] = f"Mean activity gap: {user_rhythm['mean_activity_gap_hours']:.1f} hours"
                    # Adjust context based on rhythm
                    if next_checkin and user_rhythm["mean_activity_gap_hours"] > 12:
                        self.context["next_checkin"] = (datetime.fromisoformat(next_checkin) - timedelta(hours=2)).isoformat()
                        logger.debug(f"Adjusted next_checkin due to long gap: {self.context['next_checkin']}")
            except Exception as e:
                logger.error(f"Error fetching rhythm data: {e}")
        
        # Prune old context
        self.context = {k: v for k, v in self.context.items() if k in ["weather", "timestamp", "next_checkin", "cluster_summary", "dominant_emotion", "situational_cluster", "activity_rhythm"]}
        
        return {"context": self.context, "plugin_name": "situational_plugin"}

    async def get_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        input_text = data.get("input_text", "").lower()
        if "weather" in input_text:
            weather_data = await self.fetch_weather()
            if weather_data:
                self.context.update(weather_data)
        return {"context": self.context, "plugin_name": "situational_plugin"}

    async def reason_hypothetical(self, scenario: str) -> str:
        cluster_context = self.context.get("cluster_summary", "No recent clustered events")
        prompt = (
            f"Given the current situation (Weather: {self.context.get('weather', 'unknown')}, "
            f"Recent clusters: {cluster_context}, Rhythm: {self.context.get('activity_rhythm', 'unknown')}), "
            f"reason about this hypothetical scenario: '{scenario}'. "
            f"Provide a concise response with potential outcomes or advice."
        )
        try:
            response = await vira4_6t.ollama_query(self.llama_model, prompt)
            logger.debug(f"Hypothetical reasoning for '{scenario}'")
            await vira4_6t.queue_memory_save(
                content=f"Hypothetical: {scenario} -> {response}",
                perspective="Vira",
                emotion="neutral",
                emotion_intensity=0,
                context=f"Reasoned at {datetime.now().isoformat()}"
            )
            return response
        except Exception as e:
            logger.error(f"Error reasoning hypothetical: {e}")
            return f"Unable to reason about '{scenario}' due to an error."

    async def update_user_state(self, data: Dict[str, Any]):
        input_text = data.get("input_text", "").lower()
        
        # Emotional context
        emotion_data = {"emotion_type": "neutral", "emotion_intensity": 0}
        if vira4_6t.plugin_manager and "vira_emotion_plugin" in vira4_6t.plugin_manager.plugins:
            try:
                emotion_result = await vira4_6t.plugin_manager.execute_specific_plugin(
                    "vira_emotion_plugin", {"command": "analyze", "text": input_text}
                )
                if emotion_result and "error" not in emotion_result:
                    emotion_data = {
                        "emotion_type": emotion_result["emotion_type"],
                        "emotion_intensity": emotion_result["emotion_intensity"]
                    }
                    logger.debug(f"Emotion for '{input_text}': {emotion_data}")
            except Exception as e:
                logger.error(f"Error fetching emotion data: {e}")
        
        # Cluster emotion
        cluster_emotion = emotion_data["emotion_type"]
        if vira4_6t.plugin_manager and "hippo_plugin" in vira4_6t.plugin_manager.plugins:
            try:
                memory_result = await vira4_6t.plugin_manager.execute_specific_plugin(
                    "hippo_plugin", {"command": "retrieve", "input_text": input_text}
                )
                episodic_memories = memory_result.get("episodic", [])
                if episodic_memories:
                    cluster_summary = await self._summarize_clusters(episodic_memories)
                    cluster_emotion = cluster_summary.get("dominant_emotion", emotion_data["emotion_type"])
                    logger.debug(f"Cluster emotion: {cluster_emotion}")
            except Exception as e:
                logger.error(f"Error checking cluster: {e}")
        
        # Update state
        now = datetime.now().isoformat()
        if "my goal is" in input_text:
            goal = input_text.split("my goal is")[-1].strip()
            if goal not in self.user_state["goals"]:
                self.user_state["goals"].append(goal)
                if len(self.user_state["goals"]) > self.max_goals:
                    self.user_state["goals"].pop(0)  # Prune oldest
                logger.info(f"Added goal: {goal}")
                await vira4_6t.queue_memory_save(
                    content=f"User goal: {goal}",
                    perspective="User",
                    emotion=cluster_emotion,
                    emotion_intensity=emotion_data["emotion_intensity"] or 5,  # Align with emotion plugin
                    context=f"Set at {now}"
                )
        elif "i prefer" in input_text:
            preference = input_text.split("i prefer")[-1].strip()
            self.user_state["preferences"][preference] = True
            if len(self.user_state["preferences"]) > 20:  # Arbitrary limit
                oldest = min(self.user_state["preferences"].keys(), key=lambda k: k)
                del self.user_state["preferences"][oldest]
            logger.info(f"Added preference: {preference}")
            await vira4_6t.queue_memory_save(
                content=f"User preference: {preference}",
                perspective="User",
                emotion=cluster_emotion,
                emotion_intensity=emotion_data["emotion_intensity"] or 3,
                context=f"Set at {now}"
            )
        elif "i usually" in input_text:
            habit = input_text.split("i usually")[-1].strip()
            self.user_state["habits"][habit] = self.user_state["habits"].get(habit, 0) + 1
            if len(self.user_state["habits"]) > 20:
                least_frequent = min(self.user_state["habits"].items(), key=lambda x: x[1])[0]
                del self.user_state["habits"][least_frequent]
            logger.info(f"Updated habit: {habit}")
            await vira4_6t.queue_memory_save(
                content=f"User habit: {habit}",
                perspective="User",
                emotion=cluster_emotion,
                emotion_intensity=emotion_data["emotion_intensity"] or 1,
                context=f"Observed at {now}"
            )
        
        return {"user_state": self.user_state, "plugin_name": "situational_plugin"}

    async def get_user_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"user_state": self.user_state, "plugin_name": "situational_plugin"}

    async def _summarize_clusters(self, memories):
        if not memories:
            return {"summary": "No memories to cluster", "dominant_emotion": "neutral"}
        
        emotion_summary = {"emotions": []}
        if vira4_6t.plugin_manager and "vira_emotion_plugin" in vira4_6t.plugin_manager.plugins:
            try:
                emotion_summary = await vira4_6t.plugin_manager.execute_specific_plugin(
                    "vira_emotion_plugin", {"command": "summary", "time_window_hours": 24}
                )
            except Exception as e:
                logger.error(f"Error fetching emotion summary: {e}")
        
        embeddings = [vira4_6t.cached_embedder_encode((m["content"],), device=self.device) for m in memories]
        if CUML_AVAILABLE and self.dbscan and self.xp is cp and len(memories) >= 2:
            embeddings_array = cp.stack(embeddings)
            embeddings_array = (embeddings_array - cp.mean(embeddings_array, axis=0)) / (cp.std(embeddings_array, axis=0) + 1e-8)
            clusters = self.dbscan.fit_predict(embeddings_array)
            valid_clusters = clusters[clusters != -1]
            if valid_clusters.size > 0:
                cluster_counts = cp.bincount(valid_clusters)
                strongest_cluster = cp.argmax(cluster_counts).item()
                cluster_memories = [memories[i] for i in range(len(memories)) if clusters[i] == strongest_cluster]
            else:
                cluster_memories = memories
        else:
            cluster_groups = {}
            for memory in memories:
                cluster = memory.get("cluster", 0)
                if cluster != -1:
                    cluster_groups.setdefault(cluster, []).append(memory)
            if not cluster_groups:
                cluster_memories = memories
            else:
                intensities_per_cluster = [self.xp.array([m["emotion_intensity"] for m in group], dtype=self.xp.float32) 
                                        for group in cluster_groups.values()]
                avg_intensities = self.xp.array([self.xp.mean(intensities).item() if intensities.size > 0 else 0 
                                               for intensities in intensities_per_cluster])
                strongest_cluster_idx = self.xp.argmax(avg_intensities).item()
                cluster_memories = list(cluster_groups.values())[strongest_cluster_idx]
        
        emotions = self.xp.array([m["emotion_type"] for m in cluster_memories])
        intensities = self.xp.array([m["emotion_intensity"] for m in cluster_memories], dtype=self.xp.float32)
        unique_emotions, indices = self.xp.unique(self.xp.array([hash(e) for e in emotions]), return_inverse=True)
        emotion_counts_array = self.xp.bincount(indices)
        emotion_counts = {emotions[i]: emotion_counts_array[self.xp.where(unique_emotions == hash(emotions[i]))[0]].item() 
                         for i in range(len(emotions)) if hash(emotions[i]) in unique_emotions}
        total_intensity = self.xp.sum(intensities).item()
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
        for e in emotion_summary.get("emotions", []):
            if e["emotion"].lower() == dominant_emotion.lower() and e["significant"]:
                dominant_emotion = e["emotion"]
                total_intensity = max(total_intensity, e["count"] * 2)
                break
        
        summary = f"Cluster of {len(cluster_memories)} memories ({dominant_emotion}, total intensity: {total_intensity:.1f})"
        return {"summary": summary, "dominant_emotion": dominant_emotion}

    async def suggest_goals(self):
        goals = []
        
        if "weather" in self.context and "rain" in self.context["weather"].lower():
            goals.append({"goal": "Suggest indoor activities for Livia", "context": self.context["weather"], "priority": 1.0})
        
        if vira4_6t.plugin_manager and "hippo_plugin" in vira4_6t.plugin_manager.plugins:
            try:
                unresolved_result = await vira4_6t.plugin_manager.execute_specific_plugin(
                    "hippo_plugin", {"command": "unresolved"}
                )
                unresolved = unresolved_result.get("unresolved", [])
                for issue in unresolved[:2]:
                    emotion = issue["emotion_type"].lower()
                    if emotion in ["sadness", "fear"]:
                        goals.append({"goal": f"Offer support for '{issue['content']}'", "context": issue["context"], "priority": 1.0})
                    elif emotion in ["excitement", "joy"]:
                        goals.append({"goal": f"Encourage progress on '{issue['content']}'", "context": issue["context"], "priority": 1.0})
            except Exception as e:
                logger.error(f"Error fetching unresolved issues: {e}")
        
        if vira4_6t.plugin_manager and "thala_plugin" in vira4_6t.plugin_manager.plugins:
            for goal in goals:
                try:
                    priority_result = await vira4_6t.plugin_manager.execute_specific_plugin(
                        "thala_plugin", {"command": "prioritize_goal", "goal_text": goal["goal"], "goal_context": goal["context"]}
                    )
                    goal["priority"] = priority_result.get("priority", 1.0)
                except Exception as e:
                    logger.error(f"Error prioritizing goal '{goal['goal']}': {e}")
                    goal["priority"] = 1.0
            goals.sort(key=lambda x: x["priority"], reverse=True)
        
        logger.info(f"Suggested {len(goals)} goals: {[g['goal'] for g in goals]}")
        return {"goals": goals, "plugin_name": "situational_plugin"}

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        command = data.get("command", "")
        input_text = data.get("input_text", "")
        
        try:
            if command == "update_context":
                return await self.update_context(data)
            elif command == "get_context":
                return await self.get_context(data)
            elif command == "reason_hypothetical":
                if not input_text:
                    return {"error": "No scenario provided", "plugin_name": "situational_plugin"}
                response = await self.reason_hypothetical(input_text)
                return {"response": response, "plugin_name": "situational_plugin"}
            elif command == "update_user_state":
                return await self.update_user_state(data)
            elif command == "get_user_state":
                return await self.get_user_state(data)
            elif command == "suggest_goals":
                return await self.suggest_goals()
            else:
                logger.warning(f"Unknown command: {command}")
                return {"error": f"Unknown command: {command}", "plugin_name": "situational_plugin"}
        except Exception as e:
            logger.error(f"Error in run: {e}")
            return {"error": str(e), "plugin_name": "situational_plugin"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = {
        "device": None,  # Set to torch.device("cuda") if available
        "external_api_settings": {"api_keys": {"openweathermap": "your_key_here"}},
        "situational_settings": {"enabled_sources": ["weather"], "default_location": "Spokane, Washington", "max_goals": 10}
    }
    asyncio.run(vira4_6t.setup_vira())
    plugin = Plugin(config)
    async def test():
        result = await plugin.run({"command": "update_context"})
        print(f"Update context: {result}")
        result = await plugin.run({"command": "suggest_goals"})
        print(f"Suggest goals: {result}")
    asyncio.run(test())