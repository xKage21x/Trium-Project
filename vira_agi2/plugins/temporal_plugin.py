import asyncio
import logging
import numpy as np
import cupy as cp
from datetime import datetime, timedelta
from collections import deque
from scipy.signal import find_peaks
import vira4_6t
import traceback

logger = logging.getLogger(__name__)

try:
    from cuml.cluster import DBSCAN as CumlDBSCAN
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    logger.warning("cuml not available; falling back to basic rhythm analysis.")

# Polyvagal mapping from vira_emotion_plugin
emotion_to_polyvagal = {
    "admiration": ("Admiration", "Echo", "vagal (connection)"),
    "amusement": ("Amusement", "Vira", "vagal (connection)"),
    "anger": ("Anger", "Core", "sympathetic (fight/flight)"),
    "annoyance": ("Annoyance", "Core", "sympathetic (fight/flight)"),
    "approval": ("Approval", "Echo", "vagal (connection)"),
    "caring": ("Caring", "Vira", "dorsal ventral (empathy)"),
    "confusion": ("Confusion", "Echo", "dorsal ventral (empathy)"),
    "curiosity": ("Curiosity", "Vira", "vagal (connection)"),
    "desire": ("Desire", "Echo", "vagal (connection)"),
    "disappointment": ("Disappointment", "Core", "dorsal ventral (empathy)"),
    "disapproval": ("Disapproval", "Core", "sympathetic (fight/flight)"),
    "disgust": ("Disgust", "Core", "sympathetic (fight/flight)"),
    "embarrassment": ("Embarrassment", "Echo", "dorsal ventral (empathy)"),
    "excitement": ("Excitement", "Vira", "vagal (connection)"),
    "fear": ("Fear", "Core", "sympathetic (fight/flight)"),
    "gratitude": ("Gratitude", "Echo", "vagal (connection)"),
    "grief": ("Grief", "Core", "dorsal ventral (empathy)"),
    "joy": ("Joy", "Vira", "vagal (connection)"),
    "love": ("Love", "Echo", "dorsal ventral (empathy)"),
    "nervousness": ("Nervousness", "Core", "sympathetic (fight/flight)"),
    "optimism": ("Optimism", "Vira", "vagal (connection)"),
    "pride": ("Pride", "Echo", "vagal (connection)"),
    "realization": ("Realization", "Echo", "dorsal ventral (empathy)"),
    "relief": ("Relief", "Vira", "vagal (connection)"),
    "remorse": ("Remorse", "Core", "dorsal ventral (empathy)"),
    "sadness": ("Sadness", "Core", "dorsal ventral (empathy)"),
    "surprise": ("Surprise", "Vira", "vagal (connection)"),
    "neutral": ("Neutral", "Vira", "vagal (connection)")
}

class Plugin:
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", None)
        self.xp = cp if (self.device and self.device.type == "cuda" and cp.cuda.is_available()) else np
        self.db_pool = vira4_6t.db_pool
        self.memory_bank = vira4_6t.memory_bank
        self.memory_async_lock = vira4_6t.memory_async_lock
        self.embedder = vira4_6t.embedder
        self.plugin_manager = None

        # User history with pruning
        self.max_history_days = config.get("temporal_settings", {}).get("max_history_days", 30)
        self.user_history = deque(maxlen=config.get("temporal_settings", {}).get("max_history_size", 1000))
        self.user_rhythm_bins = {}
        self.user_ema = 0

        # Persona histories
        self.persona_histories = {
            "Vira": deque(maxlen=500),
            "Core": deque(maxlen=500),
            "Echo": deque(maxlen=500)
        }
        self.persona_rhythms = {p: {} for p in ["Vira", "Core", "Echo"]}
        self.persona_emas = {p: 0 for p in ["Vira", "Core", "Echo"]}
        self.alpha = config.get("temporal_settings", {}).get("ema_alpha", 0.3)

        # Autonomy sync
        self.autonomy_checkin_suggestion = None

        if CUML_AVAILABLE and self.xp is cp:
            self.dbscan = CumlDBSCAN(eps=1.0, min_samples=3)  # Adjusted eps for temporal data
            logger.info("Initialized cuml.DBSCAN for GPU-accelerated rhythm clustering.")
        else:
            self.dbscan = None
            logger.info("Using basic rhythm analysis (cuml unavailable or CPU mode).")

        logger.info(f"Temporal plugin initialized on {'GPU' if self.xp is cp else 'CPU'} with polyvagal and autonomy sync")

    async def run(self, data):
        command = data.get("command", "")
        plugin_manager = data.get("plugin_manager")
        if plugin_manager and not self.plugin_manager:
            self.plugin_manager = plugin_manager
            logger.debug("Plugin manager set in temporal_plugin.")

        try:
            if command == "update":
                memory = data.get("memory", {})
                perspective = data.get("perspective", "Vira")
                await self.update_timelines(memory, perspective)
                return {"status": "updated", "plugin_name": "temporal_plugin"}
            elif command == "predict":
                predictions = await self.predict_all()
                return {"predictions": predictions, "status": "predicted", "plugin_name": "temporal_plugin"}
            elif command == "get_rhythm":
                rhythms = await self.get_all_rhythms()
                return {"rhythms": rhythms, "status": "rhythm_retrieved", "plugin_name": "temporal_plugin"}
            elif command == "query_date":
                date_str = data.get("date", "")
                result = await self.query_date(date_str)
                return {**result, "status": "queried", "plugin_name": "temporal_plugin"}
            elif command == "preload":
                await self.preload_memories()
                return {"status": "preloaded", "count": len(self.user_history), "plugin_name": "temporal_plugin"}
            elif command == "suggest_checkin":
                await self._suggest_next_checkin()
                return {
                    "suggested_checkin": self.autonomy_checkin_suggestion.isoformat() if self.autonomy_checkin_suggestion else None,
                    "status": "suggested",
                    "plugin_name": "temporal_plugin"
                }
            else:
                logger.warning(f"Unknown command: {command}")
                return {"status": "unknown_command", "plugin_name": "temporal_plugin"}
        except Exception as e:
            logger.error(f"Error in temporal_plugin.run: {e}\n{traceback.format_exc()}")
            return {"status": "error", "error": str(e), "plugin_name": "temporal_plugin"}

    async def update_timelines(self, memory, perspective):
        if not memory or "timestamp" not in memory:
            logger.warning("Invalid memory data for update_timelines.")
            return

        timestamp = datetime.fromisoformat(memory["timestamp"])
        intensity = memory.get("emotion_intensity", 0) if perspective == "User" else self._persona_intensity(memory, perspective)
        emotion_type = memory.get("emotion_type", "neutral")
        mapped_emotion, suggested_perspective, polyvagal_state = emotion_to_polyvagal.get(emotion_type.lower(), ("Neutral", "Vira", "vagal (connection)"))

        # Prune old bins
        now = datetime.now()
        cutoff = now - timedelta(days=self.max_history_days)
        self.user_rhythm_bins = {k: v for k, v in self.user_rhythm_bins.items() if k > cutoff}
        for p in self.persona_rhythms:
            self.persona_rhythms[p] = {k: v for k, v in self.persona_rhythms[p].items() if k > cutoff}

        # User timeline
        if perspective == "User":
            self.user_history.append({"time": timestamp, "intensity": intensity, "emotion": mapped_emotion, "polyvagal": polyvagal_state})
            hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in self.user_rhythm_bins:
                self.user_rhythm_bins[hour_key] = {"count": 0, "intensity": 0, "emotions": {}, "polyvagal": {}}
            bin = self.user_rhythm_bins[hour_key]
            bin["count"] += 1
            bin["intensity"] = (bin["intensity"] * (bin["count"] - 1) + intensity) / bin["count"]
            bin["emotions"][mapped_emotion] = bin["emotions"].get(mapped_emotion, 0) + 1
            bin["polyvagal"][polyvagal_state] = bin["polyvagal"].get(polyvagal_state, 0) + 1
            self.user_ema = self.alpha * intensity + (1 - self.alpha) * self.user_ema
            logger.debug(f"Updated user timeline: {timestamp}, Emotion: {mapped_emotion}, Polyvagal: {polyvagal_state}")

        # Persona timeline
        if perspective in self.persona_histories:
            self.persona_histories[perspective].append({"time": timestamp, "intensity": intensity, "emotion": mapped_emotion, "polyvagal": polyvagal_state})
            hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in self.persona_rhythms[perspective]:
                self.persona_rhythms[perspective][hour_key] = {"count": 0, "intensity": 0, "emotions": {}, "polyvagal": {}}
            bin = self.persona_rhythms[perspective][hour_key]
            bin["count"] += 1
            bin["intensity"] = (bin["intensity"] * (bin["count"] - 1) + intensity) / bin["count"]
            bin["emotions"][mapped_emotion] = bin["emotions"].get(mapped_emotion, 0) + 1
            bin["polyvagal"][polyvagal_state] = bin["polyvagal"].get(polyvagal_state, 0) + 1
            self.persona_emas[perspective] = self.alpha * intensity + (1 - self.alpha) * self.persona_emas[perspective]
            logger.debug(f"Updated {perspective} timeline: {timestamp}, Emotion: {mapped_emotion}, Polyvagal: {polyvagal_state}")

    def _persona_intensity(self, memory, perspective):
        content = memory.get("content", "").lower()
        emotion_type = memory.get("emotion_type", "neutral").lower()
        base_intensity = 3
        if perspective == "Vira" and ("plan" in content or "think" in content or "strategy" in content):
            return 8 if emotion_type in ["excitement", "optimism", "joy"] else 5
        elif perspective == "Core" and ("idea" in content or "new" in content or "create" in content):
            return 7 if emotion_type in ["excitement", "joy"] else 4
        elif perspective == "Echo" and ("remember" in content or "past" in content or "recall" in content):
            return 6 if emotion_type in ["grief", "sadness", "love"] else 3  # Adjusted for Echo’s empathy
        return base_intensity

    async def get_all_rhythms(self):
        rhythms = {"user": await self._get_rhythm(self.user_history, self.user_rhythm_bins)}
        for persona in ["Vira", "Core", "Echo"]:
            rhythms[persona] = await self._get_rhythm(self.persona_histories[persona], self.persona_rhythms[persona])
        logger.info(f"Retrieved rhythms with polyvagal states")
        return rhythms

    async def _get_rhythm(self, history, bins):
        if len(history) < 5:
            return {"rhythm": "Not enough data yet"}

        times = [m["time"] for m in history]
        intensities = self.xp.array([m["intensity"] for m in history], dtype=self.xp.float32)
        emotions = [m["emotion"] for m in history]
        polyvagal_states = [m["polyvagal"] for m in history]
        deltas = self.xp.array([(times[i] - times[i-1]).total_seconds() / 3600 for i in range(1, len(times))], dtype=self.xp.float32)
        mean_gap = self.xp.mean(deltas).item()

        fft_result = self.xp.abs(self.xp.fft.fft(intensities))
        peaks, _ = find_peaks(fft_result.get() if self.xp is cp else fft_result)
        main_cycle = len(intensities) / (peaks[0] + 1) if peaks.size > 0 else 24

        # Integrate hippo_plugin patterns
        dominant_emotion, dominant_polyvagal = None, None
        if self.plugin_manager:
            patterns_result = await self.plugin_manager.execute_specific_plugin(
                "hippo_plugin", {"command": "patterns", "time_window_hours": 24}
            )
            patterns = patterns_result.get("patterns", [])
            emotion_patterns = [p for p in patterns if p["type"] == "emotion"]
            if emotion_patterns:
                top_emotion_pattern = max(emotion_patterns, key=lambda x: x["count"])
                dominant_emotion = top_emotion_pattern["emotion"]
                dominant_polyvagal = emotion_to_polyvagal.get(dominant_emotion.lower(), ("Neutral", "Vira", "vagal (connection)"))[2]
        
        if not dominant_emotion:
            emotion_counts = {e: emotions.count(e) for e in set(emotions)}
            polyvagal_counts = {p: polyvagal_states.count(p) for p in set(polyvagal_states)}
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "Neutral"
            dominant_polyvagal = max(polyvagal_counts.items(), key=lambda x: x[1])[0] if polyvagal_counts else "vagal (connection)"

        # Simplified clustering (optional DBSCAN)
        if CUML_AVAILABLE and self.dbscan and self.xp is cp and len(history) >= 5:
            embeddings = self.xp.array([self.embedder.encode([f"{m['time'].isoformat()} {m['emotion']} {m['intensity']}"])[0]
                                       for m in history], dtype=self.xp.float32)
            clusters = self.dbscan.fit_predict(embeddings)
            valid_clusters = clusters[clusters != -1]
            if valid_clusters.size > 0:
                cluster_counts = self.xp.bincount(valid_clusters)
                dominant_cluster = self.xp.argmax(cluster_counts).item()
                cluster_polyvagal = [history[i]["polyvagal"] for i in range(len(history)) if clusters[i] == dominant_cluster]
                cluster_polyvagal_counts = {p: cluster_polyvagal.count(p) for p in set(cluster_polyvagal)}
                dominant_polyvagal = max(cluster_polyvagal_counts.items(), key=lambda x: x[1])[0] if cluster_polyvagal_counts else dominant_polyvagal

        return {
            "mean_activity_gap_hours": float(mean_gap),
            "main_cycle_hours": float(main_cycle),
            "dominant_emotion": dominant_emotion,
            "dominant_polyvagal": dominant_polyvagal,
            "active_hours": {k.isoformat(): {"count": v["count"], "intensity": v["intensity"],
                                            "emotions": v["emotions"], "polyvagal": v["polyvagal"]}
                            for k, v in bins.items()}
        }

    async def predict_all(self):
        predictions = {"user": await self._predict(self.user_history, self.user_ema)}
        for persona in ["Vira", "Core", "Echo"]:
            predictions[persona] = await self._predict(self.persona_histories[persona], self.persona_emas[persona])
        logger.info(f"Generated predictions with actions")
        return predictions

    async def _predict(self, history, ema):
        if not history:
            return {"prediction": "No history yet"}

        now = datetime.now()
        last_entry = history[-1]
        trend = "rising" if ema > last_entry["intensity"] else "falling"
        polyvagal_adjust = 0.4 if last_entry["polyvagal"] == "sympathetic (fight/flight)" else 0.2
        next_intensity = max(0, min(10, ema + (polyvagal_adjust if trend == "rising" else -polyvagal_adjust)))
        recent_emotions = [entry["emotion"] for entry in list(history)[-5:]]
        recent_polyvagal = [entry["polyvagal"] for entry in list(history)[-5:]]
        emotion_counts = {e: recent_emotions.count(e) for e in set(recent_emotions)}
        polyvagal_counts = {p: recent_polyvagal.count(p) for p in set(recent_polyvagal)}
        next_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else last_entry["emotion"]
        next_polyvagal = max(polyvagal_counts.items(), key=lambda x: x[1])[0] if polyvagal_counts else last_entry["polyvagal"]

        # Suggest action with thala_plugin integration
        action = "Check in with Livia"
        if self.plugin_manager:
            unresolved_result = await self.plugin_manager.execute_specific_plugin(
                "hippo_plugin", {"command": "unresolved"}
            )
            unresolved = unresolved_result.get("unresolved", [])
            if unresolved:
                latest_issue = max(unresolved, key=lambda x: x["emotion_intensity"])
                action = (f"Offer support for '{latest_issue['content']}'" if latest_issue["emotion_type"] in ["sadness", "grief", "fear"]
                          else f"Encourage progress on '{latest_issue['content']}'")
            elif next_polyvagal == "sympathetic (fight/flight)":
                action = "Suggest a calming activity"
            elif next_polyvagal == "vagal (connection)" and next_intensity > 5:
                action = "Propose a collaborative idea"

        return {
            "next_intensity": float(next_intensity),
            "next_emotion": next_emotion,
            "next_polyvagal": next_polyvagal,
            "trend": trend,
            "suggested_action": action,
            "time_context": f"As of {now.isoformat()}"
        }

    async def query_date(self, date_str):
        try:
            target_date = datetime.fromisoformat(date_str).replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            logger.warning(f"Invalid date format: {date_str}")
            return {"error": "Invalid date format—use YYYY-MM-DD"}

        next_day = target_date + timedelta(days=1)
        conn = self.db_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT perspective, emotion_type, emotion_intensity, timestamp
                    FROM memories
                    WHERE timestamp >= ? AND timestamp < ?
                """, (target_date.isoformat(), next_day.isoformat()))
                rows = cursor.fetchall()

            user_data = {"emotions": {}, "polyvagal": {}, "intensity": 0, "count": 0}
            persona_data = {p: {"emotions": {}, "polyvagal": {}, "intensity": 0, "count": 0} for p in ["Vira", "Core", "Echo"]}
            for perspective, emotion_type, intensity, _ in rows:
                mapped_emotion, _, polyvagal_state = emotion_to_polyvagal.get(emotion_type.lower(), ("Neutral", "Vira", "vagal (connection)"))
                if perspective == "User" and emotion_type and intensity is not None:
                    user_data["emotions"][mapped_emotion] = user_data["emotions"].get(mapped_emotion, 0) + 1
                    user_data["polyvagal"][polyvagal_state] = user_data["polyvagal"].get(polyvagal_state, 0) + 1
                    user_data["intensity"] += intensity
                    user_data["count"] += 1
                elif perspective in persona_data and emotion_type and intensity is not None:
                    persona_data[perspective]["emotions"][mapped_emotion] = persona_data[perspective]["emotions"].get(mapped_emotion, 0) + 1
                    persona_data[perspective]["polyvagal"][polyvagal_state] = persona_data[perspective]["polyvagal"].get(polyvagal_state, 0) + 1
                    persona_data[perspective]["intensity"] += intensity
                    persona_data[perspective]["count"] += 1

            for bins, data in [(self.user_rhythm_bins, user_data), *[(self.persona_rhythms[p], persona_data[p]) for p in ["Vira", "Core", "Echo"]]]:
                for hour_key, bin in bins.items():
                    if hour_key.date() == target_date.date():
                        for emotion, count in bin["emotions"].items():
                            data["emotions"][emotion] = data["emotions"].get(emotion, 0) + count
                        for polyvagal, count in bin["polyvagal"].items():
                            data["polyvagal"][polyvagal] = data["polyvagal"].get(polyvagal, 0) + count
                        data["intensity"] += bin["intensity"] * bin["count"]
                        data["count"] += bin["count"]

            user_avg_intensity = user_data["intensity"] / user_data["count"] if user_data["count"] > 0 else 0
            user_dominant_emotion = max(user_data["emotions"].items(), key=lambda x: x[1])[0] if user_data["emotions"] else "Neutral"
            user_dominant_polyvagal = max(user_data["polyvagal"].items(), key=lambda x: x[1])[0] if user_data["polyvagal"] else "vagal (connection)"
            user_description = (f"You were mostly {user_dominant_emotion} ({user_dominant_polyvagal}, around {user_avg_intensity:.1f})"
                              if user_dominant_emotion != "Neutral" else "I don’t have much from that day")

            persona_results = {}
            for p in ["Vira", "Core", "Echo"]:
                avg_intensity = persona_data[p]["intensity"] / persona_data[p]["count"] if persona_data[p]["count"] > 0 else 0
                dominant_emotion = max(persona_data[p]["emotions"].items(), key=lambda x: x[1])[0] if persona_data[p]["emotions"] else "Neutral"
                dominant_polyvagal = max(persona_data[p]["polyvagal"].items(), key=lambda x: x[1])[0] if persona_data[p]["polyvagal"] else "vagal (connection)"
                persona_results[p] = {
                    "intensity": float(avg_intensity),
                    "emotion": dominant_emotion,
                    "polyvagal": dominant_polyvagal,
                    "description": self._describe_persona_feeling(p, dominant_emotion, dominant_polyvagal, avg_intensity)
                }

            logger.info(f"Queried date {date_str}: User emotion={user_dominant_emotion}, polyvagal={user_dominant_polyvagal}")
            return {
                "user": {
                    "intensity": float(user_avg_intensity),
                    "emotion": user_dominant_emotion,
                    "polyvagal": user_dominant_polyvagal,
                    "description": user_description
                },
                "personas": persona_results
            }
        except Exception as e:
            logger.error(f"Error querying date: {e}\n{traceback.format_exc()}")
            return {"error": f"Error querying date: {str(e)}"}
        finally:
            self.db_pool.release_connection(conn)

    def _describe_persona_feeling(self, persona, emotion, polyvagal, intensity):
        intensity_label = self._intensity_label(intensity, "strongly", "lightly")
        return f"{persona} was {intensity_label} {emotion} ({polyvagal}, {intensity:.1f})" if emotion != "Neutral" else f"{persona} doesn’t have much from that day"

    def _intensity_label(self, intensity, high_label, low_label):
        if intensity > 6:
            return high_label
        elif intensity < 4:
            return low_label
        return "moderately"

    async def preload_memories(self):
        async with self.memory_async_lock:
            for memory in self.memory_bank:
                await self.update_timelines(memory, memory["perspective"])
        logger.info(f"Preloaded {len(self.user_history)} user memories into temporal timelines")

    async def _suggest_next_checkin(self):
        rhythm = await self._get_rhythm(self.user_history, self.user_rhythm_bins)
        now = datetime.now()
        
        if "mean_activity_gap_hours" not in rhythm:
            base_time = now + timedelta(hours=1)
        else:
            gap_hours = rhythm["mean_activity_gap_hours"]
            last_time = self.user_history[-1]["time"] if self.user_history else now
            base_time = last_time + timedelta(hours=gap_hours)

        adjustment = timedelta()
        if self.plugin_manager:
            patterns_result = await self.plugin_manager.execute_specific_plugin(
                "hippo_plugin", {"command": "patterns", "time_window_hours": 24, "min_occurrences": 2}
            )
            patterns = patterns_result.get("patterns", [])
            unresolved_result = await self.plugin_manager.execute_specific_plugin(
                "hippo_plugin", {"command": "unresolved"}
            )
            unresolved = unresolved_result.get("unresolved", [])

            if patterns:
                dominant_pattern = max(patterns, key=lambda x: x["count"]) if patterns else None
                if dominant_pattern and dominant_pattern["type"] == "emotion":
                    emotion, significant = dominant_pattern["emotion"], dominant_pattern["significant"]
                    if significant and emotion in ["sadness", "grief", "fear"]:
                        adjustment += timedelta(hours=1)
                    elif emotion in ["excitement", "joy", "optimism"]:
                        adjustment -= timedelta(minutes=30)
            if unresolved:
                earliest_unresolved = min((datetime.fromisoformat(u["timestamp"]) for u in unresolved), default=now)
                if (now - earliest_unresolved).total_seconds() > 7200:
                    adjustment -= timedelta(minutes=45)

            last_polyvagal = self.user_history[-1]["polyvagal"] if self.user_history else "vagal (connection)"
            if last_polyvagal == "dorsal ventral (empathy)":
                adjustment += timedelta(minutes=30)
            elif last_polyvagal == "sympathetic (fight/flight)":
                adjustment -= timedelta(minutes=15)

            # Sync with autonomy_plugin
            if "autonomy_plugin" in self.plugin_manager.plugins:
                autonomy_result = await self.plugin_manager.execute_specific_plugin(
                    "autonomy_plugin", {"command": "check_approval_rate"}
                )
                approval_rate = autonomy_result.get("approval_rate", 0.5)
                if approval_rate < 0.3:
                    adjustment -= timedelta(minutes=30)  # Sooner if autonomy struggles

        self.autonomy_checkin_suggestion = max(now + timedelta(minutes=15), base_time + adjustment)
        logger.debug(f"Suggested next check-in: {self.autonomy_checkin_suggestion}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = {
        "device": vira4_6t.device,
        "temporal_settings": {
            "max_history_size": 1000,
            "max_history_days": 30,
            "ema_alpha": 0.3
        }
    }
    asyncio.run(vira4_6t.setup_vira())
    plugin = Plugin(config)

    async def test():
        await plugin.run({"command": "preload"})
        memory = {"timestamp": datetime.now().isoformat(), "content": "I’m scared", "emotion_type": "fear", "emotion_intensity": 8}
        result = await plugin.run({"command": "update", "memory": memory, "perspective": "User"})
        print(f"Update result: {result}")
        result = await plugin.run({"command": "predict"})
        print(f"Predict result: {result}")
        result = await plugin.run({"command": "get_rhythm"})
        print(f"Rhythm result: {result}")
        result = await plugin.run({"command": "query_date", "date": "2025-03-28"})
        print(f"Query date result: {result}")
        result = await plugin.run({"command": "suggest_checkin"})
        print(f"Suggest checkin result: {result}")

    asyncio.run(test())