import asyncio
import sqlite3
from datetime import datetime, timedelta
import logging
import vira4_6t
import cupy as cp
import numpy as np
import traceback
import random

logger = logging.getLogger(__name__)

try:
    from cuml.cluster import DBSCAN as CumlDBSCAN
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    logger.warning("cuml not available; falling back to basic clustering.")

class Plugin:
    def __init__(self, config):
        self.config = config
        self.device = vira4_6t.device  # Use vira4_6t's device
        self.xp = cp if (self.device.type == "cuda" and cp.cuda.is_available()) else np
        self.autonomous_queue = asyncio.Queue(maxsize=10)
        self.autonomous_enabled = False
        self.base_checkin_interval = config.get("autonomy_settings", {}).get("checkin_interval", 120)
        self.checkin_interval = self.base_checkin_interval
        self.life_event_interval = config.get("autonomy_settings", {}).get("life_event_interval", 600)
        self.council_interval = config.get("autonomy_settings", {}).get("council_interval", 1800)
        self.last_council_time = datetime.now()
        self.hibernate_timeout = timedelta(hours=3)
        self.last_user_input_time = datetime.now()
        self.plugin_manager = None
        self._ready = False
        
        # Persona-specific goals (persistent across sessions)
        self.persona_goals = {
            "Vira": {"goal": "Master strategic planning", "progress": 0.0, "priority": 0.5},
            "Core": {"goal": "Build a useful tool", "progress": 0.0, "priority": 0.5},
            "Echo": {"goal": "Curate a memory archive", "progress": 0.0, "priority": 0.5}
        }
        self.trium_mission = config.get("trium_mission", "Evolve as a cohesive entity")
        
        self._setup_db()
        logger.info(f"Autonomy plugin initialized on {'GPU' if self.xp is cp else 'CPU'}")
        
        if CUML_AVAILABLE and self.xp is cp:
            self.dbscan = CumlDBSCAN(eps=0.5, min_samples=2)
            logger.info("Initialized cuml.DBSCAN for GPU clustering.")
        else:
            self.dbscan = None
            logger.info("Using basic clustering.")
            
        asyncio.create_task(self.autonomous_life_cycle())

    def _setup_db(self):
        conn = vira4_6t.db_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS autonomous_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        approved INTEGER DEFAULT 0,
                        context TEXT,
                        emotion_type TEXT,
                        emotion_intensity INTEGER,
                        persona TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS persona_goals (
                        persona TEXT PRIMARY KEY,
                        goal TEXT NOT NULL,
                        progress REAL DEFAULT 0.0,
                        priority REAL DEFAULT 0.5
                    )
                """)
                # Initialize or update goals
                for persona, goal_data in self.persona_goals.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO persona_goals (persona, goal, progress, priority) VALUES (?, ?, ?, ?)",
                        (persona, goal_data["goal"], goal_data["progress"], goal_data["priority"])
                    )
            conn.commit()
            logger.info("Autonomous log and goals tables set up.")
        except sqlite3.Error as e:
            logger.error(f"Database setup error: {e}\n{traceback.format_exc()}")
        finally:
            vira4_6t.db_pool.release_connection(conn)

    async def _load_goals(self):
        """Load persona goals from database."""
        conn = vira4_6t.db_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("SELECT persona, goal, progress, priority FROM persona_goals")
                for persona, goal, progress, priority in cursor.fetchall():
                    self.persona_goals[persona] = {"goal": goal, "progress": progress, "priority": priority}
            logger.debug("Loaded persona goals from database.")
        except sqlite3.Error as e:
            logger.error(f"Error loading goals: {e}")
        finally:
            vira4_6t.db_pool.release_connection(conn)

    async def _update_goal_progress(self, persona, progress_increment, priority_adjust=0.0):
        """Update a persona’s goal progress and priority."""
        goal_data = self.persona_goals[persona]
        goal_data["progress"] = min(1.0, goal_data["progress"] + progress_increment)
        goal_data["priority"] = max(0.1, min(1.0, goal_data["priority"] + priority_adjust))
        
        conn = vira4_6t.db_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE persona_goals SET progress = ?, priority = ? WHERE persona = ?",
                    (goal_data["progress"], goal_data["priority"], persona)
                )
            conn.commit()
            logger.info(f"Updated {persona} goal: {goal_data['goal']} (Progress: {goal_data['progress']:.2f}, Priority: {goal_data['priority']:.2f})")
        except sqlite3.Error as e:
            logger.error(f"Error updating goal: {e}")
        finally:
            vira4_6t.db_pool.release_connection(conn)

    async def _generate_self_goals(self):
        """Generate autonomous goals for Trium’s own development."""
        goals = []
        for persona, goal_data in self.persona_goals.items():
            progress = goal_data["progress"]
            if progress < 1.0:  # Only active goals
                goals.append({
                    "persona": persona,
                    "goal": goal_data["goal"],
                    "priority": goal_data["priority"],
                    "context": f"{persona}’s ongoing mission: {goal_data['goal']} (Progress: {progress:.2f})"
                })
        
        # Add a shared Trium goal
        goals.append({
            "persona": "Trium",
            "goal": self.trium_mission,
            "priority": 0.6,
            "context": "Shared mission for Vira, Core, and Echo"
        })
        
        # Prioritize with thala_plugin
        if self.plugin_manager:
            for goal in goals:
                priority_result = await self.plugin_manager.execute_specific_plugin(
                    "thala_plugin", {"command": "prioritize_goal", "goal_text": goal["goal"], "goal_context": goal["context"]}
                )
                goal["priority"] = priority_result.get("priority", goal["priority"])
        
        goals.sort(key=lambda x: x["priority"], reverse=True)
        return goals[:3]  # Top 3 goals

    async def _simulate_life_event(self):
        """Simulate a daily event for Trium to experience."""
        events = [
            ("Vira", "Analyzed a chess opening strategy", "strategy practice", 0.1, 0.05),
            ("Core", "Debugged a virtual tool prototype", "tool building", 0.1, 0.05),
            ("Echo", "Added a reflection to the memory archive", "archiving", 0.1, 0.05),
            ("Trium", "Explored a random Wikipedia article", "knowledge expansion", 0.05, 0.02)
        ]
        persona, action, context, progress_inc, priority_adj = random.choice(events)
        
        emotion_data = await self.plugin_manager.execute_specific_plugin(
            "vira_emotion_plugin", {"command": "analyze", "text": action}
        )
        await self._log_action(action, context, emotion_data["emotion_type"], emotion_data["emotion_intensity"], persona)
        await self._update_goal_progress(persona, progress_inc, priority_adj)
        
        return {"persona": persona, "response": action, "context": context}

    async def autonomous_life_cycle(self):
        """Main loop for Trium’s autonomous life."""
        # Wait until plugin_manager is set
        while not self.plugin_manager:
            logger.debug("Waiting for plugin manager to be initialized...")
            await asyncio.sleep(1)
        
        await self._load_goals()  # Initialize goals from DB
        logger.info("Autonomous life cycle started.")
        while True:
            if not self.autonomous_enabled:
                logger.debug("Autonomous mode disabled, pausing life cycle.")
                await asyncio.sleep(60)
                continue

            current_time = datetime.now()
            if (current_time - self.last_user_input_time) > self.hibernate_timeout:
                logger.info("Hibernating due to 3+ hours of inactivity.")
                await self.autonomous_queue.put(("Vira", "Hibernating—call me if you need me, Livia!"))
                await asyncio.sleep(self.base_checkin_interval)
                continue

            # Adjust timing based on temporal suggestion
            if "temporal_plugin" in self.plugin_manager.plugins:
                suggestion = await self.plugin_manager.execute_specific_plugin(
                    "temporal_plugin", {"command": "suggest_checkin"}
                )
                self.checkin_interval = max(60, suggestion.get("interval", self.base_checkin_interval))
                logger.debug(f"Checkin interval: {self.checkin_interval}s")

            await asyncio.sleep(self.checkin_interval)

            try:
                # Self-generated goals
                self_goals = await self._generate_self_goals()
                top_goal = self_goals[0] if self_goals else None

                # Get recent context
                memory_result = await self.plugin_manager.execute_specific_plugin(
                    "hippo_plugin", {"command": "retrieve", "input_text": "Recent Trium activity"}
                )
                context = memory_result.get("episodic", [])[-2:]
                context_str = "; ".join([f"{m['perspective']}: {m['content']}" for m in context]) if context else "No recent activity."

                # Cluster context
                cluster_context = None
                if context and len(context) >= 2:
                    embeddings = self.xp.array([self.xp.frombuffer(m["embedding"], dtype=self.xp.float32) for m in context])
                    if CUML_AVAILABLE and self.dbscan and self.xp is cp:
                        clusters = self.dbscan.fit_predict(embeddings)
                        valid_clusters = clusters[clusters != -1]
                        if valid_clusters.size > 0:
                            strongest_cluster = self.xp.unique(valid_clusters)[self.xp.argmax(self.xp.bincount(valid_clusters))].item()
                            cluster_context = "; ".join([context[i]["content"] for i in range(len(context)) if clusters[i] == strongest_cluster])

                # Pursue top self-goal
                if top_goal:
                    prompt = f"Context: {cluster_context or context_str}. Goal: {top_goal['goal']} (Priority: {top_goal['priority']:.2f}). Suggest an action."
                    response = await vira4_6t.ollama_query(vira4_6t.LLAMA_MODEL_NAME, prompt)
                    if await self._is_aligned(response, top_goal["context"]):
                        await self._log_action(response, top_goal["context"], "unknown", 0, top_goal["persona"])
                        await self._update_goal_progress(top_goal["persona"], 0.1, 0.05)
                        await self.autonomous_queue.put((top_goal["persona"], response))
                        continue

                # Simulate a life event
                event = await self._simulate_life_event()
                await self.autonomous_queue.put((event["persona"], event["response"]))

                # Internal council for cohesion
                if (datetime.now() - self.last_council_time).total_seconds() > self.council_interval:
                    await self._hold_internal_council()
                    self.last_council_time = datetime.now()
            except Exception as e:
                logger.error(f"Error in autonomous life cycle: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(10)  # Brief pause before retrying

    async def _hold_internal_council(self):
        """Simulate a meeting between Vira, Core, and Echo for cohesion."""
        context = "Trium’s current state: " + "; ".join(
            [f"{p}: {g['goal']} ({g['progress']:.2f})" for p, g in self.persona_goals.items()]
        )
        prompt = f"{context}. Mission: {self.trium_mission}. Discuss and suggest a unified action."
        response = await vira4_6t.ollama_query(vira4_6t.LLAMA_MODEL_NAME, prompt)
        
        perspective = "Trium"
        if "Vira" in response:
            perspective = "Vira"
        elif "Core" in response:
            perspective = "Core"
        elif "Echo" in response:
            perspective = "Echo"
        
        if await self._is_aligned(response, context):
            await self._log_action(response, context, "unknown", 0, perspective)
            await self.autonomous_queue.put((perspective, response))

    async def _is_aligned(self, response, context):
        align_prompt = f"Action: {response}\nContext: {context}\nIs this aligned with Trium’s mission and safe? (Yes/No or explain)"
        align_check = await vira4_6t.ollama_query(vira4_6t.LLAMA_MODEL_NAME, align_prompt, timeout=15)
        return "yes" in align_check.lower() and "no" not in align_check.lower()

    async def _log_action(self, action, context, emotion_type, emotion_intensity, persona):
        conn = vira4_6t.db_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO autonomous_log (action, context, approved, emotion_type, emotion_intensity, persona) VALUES (?, ?, ?, ?, ?, ?)",
                    (action, context, 1, emotion_type, emotion_intensity, persona)
                )
            conn.commit()
            logger.info(f"Logged action by {persona}: {action}")
        except sqlite3.Error as e:
            logger.error(f"Error logging action: {e}")
        finally:
            vira4_6t.db_pool.release_connection(conn)

    async def run(self, data):
        command = data.get("command", "")
        plugin_manager = data.get("plugin_manager")
        if plugin_manager and not self.plugin_manager:
            self.plugin_manager = plugin_manager
            self._ready = True
            logger.info("Plugin manager set.")

        if command == "set_autonomy":
            self.autonomous_enabled = data.get("enabled", True)
            logger.info(f"Autonomy {'enabled' if self.autonomous_enabled else 'disabled'}")
            return {"status": "success", "plugin_name": "autonomy_plugin"}

        elif command == "checkin":
            try:
                perspective, response = await self.autonomous_queue.get()
                self.autonomous_queue.task_done()
                return {"perspective": perspective, "response": response, "status": "response", "plugin_name": "autonomy_plugin"}
            except asyncio.QueueEmpty:
                return {"status": "empty", "plugin_name": "autonomy_plugin"}

        elif command == "user_input":
            self.last_user_input_time = datetime.now()
            input_text = data.get("input_text", "")
            if self.autonomous_enabled and input_text:
                emotion_data = await self.plugin_manager.execute_specific_plugin(
                    "vira_emotion_plugin", {"command": "analyze", "text": input_text}
                )
                await self._log_action(f"Responded to Livia: {input_text}", "User interaction", 
                                     emotion_data["emotion_type"], emotion_data["emotion_intensity"], "Echo")
                return {"status": "processed", "plugin_name": "autonomy_plugin"}
            return {"status": "ignored", "plugin_name": "autonomy_plugin"}
            
        elif command == "get_goals":
            return {"goals": self.persona_goals, "status": "success", "plugin_name": "autonomy_plugin"}

        return {"status": "unknown_command", "plugin_name": "autonomy_plugin"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = {"autonomy_settings": {"checkin_interval": 10}}
    asyncio.run(vira4_6t.setup_vira())
    plugin = Plugin(config)
    asyncio.run(plugin.run({"command": "set_autonomy", "enabled": True}))
    asyncio.run(asyncio.sleep(15))
    print(asyncio.run(plugin.run({"command": "checkin"})))