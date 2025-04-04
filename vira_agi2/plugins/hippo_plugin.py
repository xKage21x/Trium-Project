import logging
import sqlite3
import asyncio
import numpy as np
import cupy as cp
from langchain_ollama import OllamaLLM
import vira4_6t
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, END
from typing import Dict, Any
import hdbscan
import traceback

logger = logging.getLogger(__name__)

try:
    from cuml.cluster import HDBSCAN as CumlHDBSCAN
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    logger.warning("cuml not available; falling back to hdbscan for clustering.")

class MemoryState:
    def __init__(self):
        self.summary = ""
        self.messages = []

    def update_summary(self, input_text: str, response: str, emotion_type: str = "unknown", emotion_intensity: int = 0) -> None:
        message_key = (input_text, response, emotion_type, emotion_intensity)
        message_dict = {m["key"]: m for m in self.messages if "key" in m}
        if message_key not in message_dict:
            self.messages.append({
                "input": input_text,
                "response": response,
                "emotion_type": emotion_type,
                "emotion_intensity": emotion_intensity,
                "key": message_key
            })
        self.summary = f"Summary: {len(self.messages)} turns, latest: {response} (Emotion: {emotion_type}, Intensity: {emotion_intensity})"

def reduce_memory_state(state: MemoryState, update: Dict[str, Any]) -> MemoryState:
    state.update_summary(
        update["input"], 
        update["response"], 
        update.get("emotion_type", "unknown"), 
        update.get("emotion_intensity", 0)
    )
    return state

class Plugin:
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", None)
        self.xp = cp if (self.device and self.device.type == "cuda" and cp.cuda.is_available()) else np
        self.llm = OllamaLLM(model=vira4_6t.LLAMA_MODEL_NAME)
        self.graph = StateGraph(MemoryState)
        self.graph.add_node("memory", lambda state: state)
        self.graph.set_entry_point("memory")
        self.graph.add_edge("memory", END)
        self.memory = self.graph.compile()
        self.checkpoint = MemoryState()
        
        self.faiss_index = vira4_6t.index
        self.memory_bank = vira4_6t.memory_bank
        self.memory_async_lock = vira4_6t.memory_async_lock
        self.embedder = vira4_6t.embedder
        self.db_pool = vira4_6t.db_pool
        self.max_memory_bank_size = vira4_6t.MAX_MEMORY_BANK_SIZE
        self.relevance_threshold = vira4_6t.MEMORY_RELEVANCE_THRESHOLD
        self.startup_memory_turns = vira4_6t.STARTUP_MEMORY_TURNS
        self.intensity_threshold = config.get("emotion_settings", {}).get("intensity_threshold", 3)
        self.memory_set = set()
        self.plugin_manager = None
        
        clustering_config = config.get("memory_settings", {}).get("clustering", {})
        self.clustering_enabled = clustering_config.get("enabled", True)
        if self.clustering_enabled:
            if CUML_AVAILABLE and self.xp is cp:
                self.clusterer = CumlHDBSCAN(
                    min_cluster_size=clustering_config.get("min_cluster_size", 3),
                    min_samples=clustering_config.get("min_samples", 2),
                    cluster_selection_method=clustering_config.get("cluster_selection_method", "eom")
                )
                logger.info(f"Using cuml.HDBSCAN for GPU-accelerated clustering with min_cluster_size={self.clusterer.min_cluster_size}")
            else:
                self.clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=clustering_config.get("min_cluster_size", 3),
                    min_samples=clustering_config.get("min_samples", 2),
                    cluster_selection_method=clustering_config.get("cluster_selection_method", "eom")
                )
                logger.info(f"Using hdbscan.HDBSCAN for clustering (CPU or cuml unavailable)")
            self.cluster_interval = clustering_config.get("cluster_interval", 30)
            self.last_cluster_time = None
            asyncio.create_task(self._periodic_clustering())
        else:
            self.clusterer = None
        
        logger.info(f"Hippocampus plugin initialized with FAISS on {'GPU' if self.xp is cp else 'CPU'}")
        
        # Preload memory_set
        conn = self.db_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("SELECT content, perspective, emotion_type, emotion_intensity FROM memories")
                for content, perspective, emotion_type, emotion_intensity in cursor.fetchall():
                    self.memory_set.add((content, perspective, emotion_type or "unknown", emotion_intensity or 0))
        except sqlite3.Error as e:
            logger.error(f"Error preloading memory_set: {e}\n{traceback.format_exc()}")
        finally:
            self.db_pool.release_connection(conn)

    async def retrieve_emotional_patterns(self, time_window_hours=24, min_occurrences=2):
        """Retrieve emotional patterns or recurring themes from recent memories."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        conn = self.db_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT content, emotion_type, emotion_intensity FROM memories WHERE timestamp > ?",
                    (cutoff_time.isoformat(),)
                )
                recent_memories = [{"content": r[0], "emotion_type": r[1], "emotion_intensity": r[2]} for r in cursor.fetchall()]
        
            if not recent_memories:
                logger.debug(f"No memories in the last {time_window_hours} hours.")
                return {"patterns": [], "summary": "No recent activity", "plugin_name": "hippo_plugin"}

            # Aggregate emotions and topics
            emotion_counts = {}
            topic_counts = {}
            for memory in recent_memories:
                emotion = memory["emotion_type"]
                intensity = memory["emotion_intensity"]
                content = memory["content"].lower()
                
                emotion_key = (emotion, intensity >= self.intensity_threshold)
                emotion_counts[emotion_key] = emotion_counts.get(emotion_key, 0) + 1
                
                keywords = set(content.split()) & {"sad", "happy", "work", "family", "day", "tough", "great"}
                for keyword in keywords:
                    topic_counts[keyword] = topic_counts.get(keyword, 0) + 1
            
            patterns = []
            for (emotion, significant), count in emotion_counts.items():
                if count >= min_occurrences:
                    patterns.append({
                        "type": "emotion",
                        "emotion": emotion,
                        "significant": significant,
                        "count": count,
                        "description": f"{emotion} {'(significant)' if significant else ''} appeared {count} times"
                    })
            for topic, count in topic_counts.items():
                if count >= min_occurrences:
                    patterns.append({
                        "type": "topic",
                        "topic": topic,
                        "count": count,
                        "description": f"Topic '{topic}' mentioned {count} times"
                    })
            
            summary = f"Found {len(patterns)} patterns in last {time_window_hours}h: " + "; ".join(p["description"] for p in patterns)
            logger.info(summary)
            return {"patterns": patterns, "summary": summary, "plugin_name": "hippo_plugin"}
        finally:
            self.db_pool.release_connection(conn)

    async def retrieve_unresolved_issues(self, intensity_threshold=None):
        """Identify significant memories without follow-up."""
        threshold = intensity_threshold if intensity_threshold is not None else self.intensity_threshold
        conn = self.db_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT timestamp, content, emotion_type, emotion_intensity, context, cluster_label 
                    FROM memories 
                    WHERE emotion_intensity >= ? AND perspective = 'User'
                    ORDER BY timestamp ASC
                    """,
                    (threshold,)
                )
                all_memories = [
                    {
                        "timestamp": r[0],
                        "content": r[1],
                        "emotion_type": r[2],
                        "intensity": r[3],
                        "context": r[4],
                        "cluster": r[5]
                    } for r in cursor.fetchall()
                ]
            
            unresolved = []
            for i, memory in enumerate(all_memories):
                has_followup = any(
                    m["perspective"] != "User" and 
                    datetime.fromisoformat(m["timestamp"]) > datetime.fromisoformat(memory["timestamp"]) and
                    m["content"] != memory["content"]  # Avoid self-reference
                    for m in all_memories[i+1:]
                )
                if not has_followup:
                    unresolved.append(memory)
            
            # Boost unresolved issues with temporal rhythm
            if self.plugin_manager and "temporal_plugin" in self.plugin_manager.plugins:
                rhythm_result = await self.plugin_manager.execute_specific_plugin(
                    "temporal_plugin", {"command": "get_rhythm"}
                )
                user_rhythm = rhythm_result.get("rhythms", {}).get("user", {})
                dominant_emotion = user_rhythm.get("dominant_emotion", "").lower()
                for issue in unresolved:
                    if issue["emotion_type"].lower() == dominant_emotion:
                        issue["priority_boost"] = 0.1
                        logger.debug(f"Temporal boost for unresolved issue '{issue['content']}': {dominant_emotion}")
            
            summary = f"Found {len(unresolved)} unresolved issues above intensity {threshold}"
            logger.info(summary)
            return {"unresolved": unresolved, "summary": summary, "plugin_name": "hippo_plugin"}
        finally:
            self.db_pool.release_connection(conn)

    async def encode_episodic_memory(self, input_text, perspective, emotion_data=None, timestamp=None, is_autonomous=False):
        if not input_text:
            logger.warning("No input text provided for encoding.")
            return {"status": "no_input", "plugin_name": "hippo_plugin"}
        
        timestamp = timestamp or datetime.now().isoformat()
        event_summary = f"{timestamp} - {perspective}: {input_text}"
        
        emotion_type = emotion_data.get("emotion_type", "unknown") if emotion_data else "unknown"
        emotion_intensity = emotion_data.get("emotion_intensity", 0) if emotion_data else 0
        context = emotion_data.get("context", f"Encoded by {perspective} at {timestamp}") if emotion_data else f"Encoded by {perspective} at {timestamp}"
        is_feedback = "Feedback provided" in context
        if is_autonomous:
            context = f"Autonomous action by {perspective} at {timestamp}"
        
        memory_key = (input_text, perspective, emotion_type, emotion_intensity)
        
        # Semantic deduplication
        embedding = self.xp.array(self.embedder.encode([input_text])[0], dtype=self.xp.float32)
        async with self.memory_async_lock:
            for existing_memory in self.memory_bank:
                similarity = self.xp.dot(embedding, existing_memory["embedding"]) / (
                    self.xp.linalg.norm(embedding) * self.xp.linalg.norm(existing_memory["embedding"])
                )
                if similarity > 0.95 and not is_feedback and not is_autonomous:
                    logger.debug(f"Duplicate memory skipped (similarity {similarity:.3f}): {event_summary}")
                    return {"status": "duplicate_skipped", "similarity": float(similarity), "plugin_name": "hippo_plugin"}
        
        if memory_key in self.memory_set and not (is_feedback or is_autonomous):
            logger.debug(f"Duplicate memory skipped: {event_summary}")
            return {"status": "duplicate_skipped", "plugin_name": "hippo_plugin"}
        
        # Enhance context with situational data
        if self.plugin_manager and "situational_plugin" in self.plugin_manager.plugins:
            situational_result = await self.plugin_manager.execute_specific_plugin(
                "situational_plugin", {"command": "get_context"}
            )
            situational_context = situational_result.get("context", {})
            if "weather" in situational_context:
                context += f" | Weather: {situational_context['weather']}"
        
        conn = self.db_pool.get_connection()
        try:
            response = f"{perspective} acknowledged: {input_text}"
            self.checkpoint = reduce_memory_state(self.checkpoint, {
                "input": input_text,
                "response": response,
                "emotion_type": emotion_type,
                "emotion_intensity": emotion_intensity
            })
            
            texts_to_embed = [input_text, vira4_6t.create_emotional_context(input_text, emotion_type, emotion_intensity)]
            embeddings = self.xp.array(self.embedder.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True, batch_size=2), dtype=self.xp.float32)
            embedding, emotion_embedding = embeddings[0], embeddings[1]
            
            async with self.memory_async_lock:
                with conn:
                    conn.execute("BEGIN TRANSACTION")
                    try:
                        cursor = conn.cursor()
                        cluster_label = await self._get_cluster_label(embedding) if self.clustering_enabled else None
                        
                        if is_feedback:
                            cursor.execute(
                                """
                                SELECT id FROM memories 
                                WHERE content = ? AND perspective = ? AND emotion_type != ? AND emotion_intensity != ?
                                """,
                                (input_text, perspective, emotion_type, emotion_intensity)
                            )
                            existing = cursor.fetchone()
                            if existing:
                                cursor.execute(
                                    """
                                    UPDATE memories 
                                    SET emotion_type = ?, emotion_intensity = ?, context = ?, emotion_embedding = ?, timestamp = ?, cluster_label = ?
                                    WHERE id = ?
                                    """,
                                    (emotion_type, emotion_intensity, context, emotion_embedding.tobytes(), timestamp, cluster_label, existing[0])
                                )
                                memory_id = existing[0]
                                logger.info(f"Updated memory with feedback: {event_summary}")
                            else:
                                cursor.execute(
                                    """
                                    INSERT INTO memories (content, perspective, source_type, embedding, creator, timestamp, 
                                                        emotion_type, emotion_intensity, context, emotion_embedding, cluster_label) 
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (input_text, perspective, "text", embedding.tobytes(), 
                                     "Livia" if perspective == "User" else perspective, timestamp, 
                                     emotion_type, emotion_intensity, context, emotion_embedding.tobytes(), cluster_label)
                                )
                                memory_id = cursor.lastrowid
                                logger.info(f"Encoded new feedback memory: {event_summary}")
                        else:
                            cursor.execute(
                                """
                                INSERT INTO memories (content, perspective, source_type, embedding, creator, timestamp, 
                                                    emotion_type, emotion_intensity, context, emotion_embedding, cluster_label) 
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (input_text, perspective, "text", embedding.tobytes(), 
                                 "Livia" if perspective == "User" else perspective, timestamp, 
                                 emotion_type, emotion_intensity, context, emotion_embedding.tobytes(), cluster_label)
                            )
                            memory_id = cursor.lastrowid
                            logger.info(f"Encoded memory: {event_summary} (Cluster: {cluster_label})")
                        
                        new_memory = {
                            "id": memory_id,
                            "content": input_text,
                            "perspective": perspective,
                            "embedding": embedding,
                            "emotion_type": emotion_type,
                            "emotion_intensity": emotion_intensity,
                            "context": context,
                            "emotion_embedding": emotion_embedding,
                            "timestamp": timestamp,
                            "cluster": cluster_label,
                            "is_autonomous": is_autonomous
                        }
                        if is_feedback and existing:
                            for i, mem in enumerate(self.memory_bank):
                                if mem["id"] == memory_id:
                                    self.memory_bank[i] = new_memory
                                    break
                        else:
                            self.memory_bank.append(new_memory)
                            if len(self.memory_bank) > self.max_memory_bank_size:
                                self.memory_bank.pop(0)
                        
                        async with vira4_6t.index_lock:
                            if is_feedback and existing:
                                logger.debug(f"FAISS index unchanged for feedback update: {memory_id}")
                            else:
                                self.faiss_index.add(embedding[None] if self.xp is np else embedding[None].get())
                        
                        self.memory_set.add(memory_key)
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        raise
            return {"status": "encoded", "memory_id": memory_id, "plugin_name": "hippo_plugin", "is_feedback": is_feedback, "cluster": cluster_label}
        
        except sqlite3.Error as e:
            logger.error(f"Database error encoding memory: {e}\n{traceback.format_exc()}")
            return {"status": "error", "error": f"Database error: {str(e)}", "plugin_name": "hippo_plugin"}
        finally:
            self.db_pool.release_connection(conn)

    async def retrieve_action_outcomes(self, limit=5):
        """Retrieve recent autonomous action outcomes for reflection."""
        conn = self.db_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT m.content, m.perspective, m.emotion_type, m.emotion_intensity, m.timestamp, al.approved 
                    FROM memories m 
                    JOIN autonomous_log al ON m.content = al.action 
                    WHERE m.is_autonomous = 1 
                    ORDER BY m.timestamp DESC LIMIT ?
                    """,
                    (limit,)
                )
                outcomes = [
                    {
                        "content": row[0],
                        "perspective": row[1],
                        "emotion_type": row[2],
                        "emotion_intensity": row[3],
                        "timestamp": row[4],
                        "approved": bool(row[5])
                    } for row in cursor.fetchall()
                ]
            logger.info(f"Retrieved {len(outcomes)} autonomous action outcomes")
            return {"outcomes": outcomes, "plugin_name": "hippo_plugin"}
        except sqlite3.Error as e:
            logger.error(f"Error retrieving action outcomes: {e}")
            return {"outcomes": [], "error": str(e), "plugin_name": "hippo_plugin"}
        finally:
            self.db_pool.release_connection(conn)

    async def retrieve_memories(self, query, top_k=3):
        if not query:
            logger.warning("No query provided for memory retrieval.")
            return {"episodic": [], "semantic": "", "plugin_name": "hippo_plugin"}
        
        async with self.memory_async_lock:
            if not self.memory_bank or not self.faiss_index.ntotal:
                logger.warning("Memory bank or FAISS index is empty")
                return {"episodic": [], "semantic": "", "plugin_name": "hippo_plugin"}
        
        query_embedding = self.xp.array(self.embedder.encode([query])[0], dtype=self.xp.float32)
        distances, indices = self.faiss_index.search(query_embedding[None] if self.xp is np else query_embedding[None].get(), top_k * 2)
        
        episodic_memories = []
        seen_content = set()
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.memory_bank):
                memory = self.memory_bank[idx]
                distance = distances[0][i]
                context = memory.get("context", "")
                relevance_boost = 0.1 if "Feedback provided" in context else 0
                cluster_boost = 0.2 if memory.get("cluster") is not None and memory["cluster"] != -1 else 0
                adjusted_distance = distance - relevance_boost - cluster_boost
                if adjusted_distance < self.relevance_threshold and memory["content"] not in seen_content:
                    episodic_memories.append({
                        "timestamp": memory["timestamp"],
                        "perspective": memory["perspective"],
                        "content": memory["content"],
                        "emotion_type": memory["emotion_type"],
                        "emotion_intensity": memory["emotion_intensity"],
                        "context": context,
                        "distance": float(adjusted_distance),
                        "is_feedback": "Feedback provided" in context,
                        "cluster": memory.get("cluster"),
                        "embedding": memory["embedding"]
                    })
                    seen_content.add(memory["content"])
            if len(episodic_memories) >= top_k:
                break
        
        episodic_memories = sorted(
            [m for m in episodic_memories if m["emotion_intensity"] >= self.intensity_threshold],
            key=lambda x: x["distance"]
        )[:top_k]
        
        if episodic_memories:
            memory_texts = "\n".join(
                [f"{m['perspective']}: {m['content']} (Emotion: {m['emotion_type']}, Intensity: {m['emotion_intensity']})" 
                 for m in episodic_memories]
            )
            summary_prompt = f"Provide a concise summary of these memories in response to '{query}':\n{memory_texts}"
            try:
                async with asyncio.timeout(20):
                    semantic_summary = await asyncio.to_thread(self.llm.invoke, summary_prompt)
            except asyncio.TimeoutError:
                semantic_summary = self.checkpoint.summary
                logger.warning("LLM summary timed out")
            except Exception as e:
                semantic_summary = self.checkpoint.summary
                logger.error(f"LLM summary error: {e}")
        else:
            semantic_summary = "No relevant memories found."
        
        logger.info(f"Retrieved {len(episodic_memories)} episodic memories for query: {query}")
        return {"episodic": episodic_memories, "semantic": semantic_summary, "plugin_name": "hippo_plugin"}

    async def preload_memories(self):
        conn = self.db_pool.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, content, perspective, timestamp, emotion_type, emotion_intensity, embedding, context, emotion_embedding, cluster_label, is_autonomous 
                    FROM memories 
                    ORDER BY timestamp DESC
                    """
                )
                rows = cursor.fetchall()
            
            async with self.memory_async_lock:
                self.memory_bank.clear()
                embeddings_to_add = []
                for row in rows:
                    (memory_id, content, perspective, timestamp, emotion_type, emotion_intensity, 
                     embedding_bytes, context, emotion_embedding_bytes, cluster_label, is_autonomous) = row
                    embedding = self.xp.frombuffer(embedding_bytes, dtype=self.xp.float32)
                    emotion_embedding = self.xp.frombuffer(emotion_embedding_bytes, dtype=self.xp.float32)
                    new_memory = {
                        "id": memory_id,
                        "content": content,
                        "perspective": perspective,
                        "embedding": embedding,
                        "emotion_type": emotion_type or "unknown",
                        "emotion_intensity": emotion_intensity or 0,
                        "context": context,
                        "emotion_embedding": emotion_embedding,
                        "timestamp": timestamp,
                        "cluster": cluster_label,
                        "is_autonomous": bool(is_autonomous)
                    }
                    self.memory_bank.append(new_memory)
                    embeddings_to_add.append(embedding)
                
                async with vira4_6t.index_lock:
                    self.faiss_index.reset()
                    if embeddings_to_add:
                        embeddings_array = self.xp.array(embeddings_to_add, dtype=self.xp.float32)
                        self.faiss_index.add(embeddings_array if self.xp is np else embeddings_array.get())
                        logger.info(f"Rebuilt FAISS index with {len(embeddings_to_add)} memories")
            
            logger.info(f"Preloaded {len(rows)} memories into memory_bank and FAISS")
            return {"status": "preloaded", "count": len(rows), "plugin_name": "hippo_plugin"}
        
        except sqlite3.Error as e:
            logger.error(f"Database error preloading memories: {e}\n{traceback.format_exc()}")
            return {"status": "error", "error": f"Database error: {str(e)}", "plugin_name": "hippo_plugin"}
        finally:
            self.db_pool.release_connection(conn)

    async def _periodic_clustering(self):
        while self.clustering_enabled:
            await asyncio.sleep(self.cluster_interval)
            if len(self.memory_bank) < self.clusterer.min_cluster_size:
                logger.debug("Insufficient memories for clustering.")
                continue
            
            try:
                async with asyncio.timeout(30):
                    async with self.memory_async_lock:
                        embeddings = self.xp.array([m["embedding"] for m in self.memory_bank], dtype=self.xp.float32)
                        if embeddings.shape[0] < self.clusterer.min_cluster_size:
                            continue
                        
                        if CUML_AVAILABLE and self.xp is cp:
                            clusters = self.clusterer.fit_predict(embeddings)
                        else:
                            clusters = self.clusterer.fit_predict(embeddings if self.xp is np else embeddings.get())
                        
                        conn = self.db_pool.get_connection()
                        try:
                            with conn:
                                conn.execute("BEGIN TRANSACTION")
                                cursor = conn.cursor()
                                for i, memory in enumerate(self.memory_bank):
                                    cluster_label = int(clusters[i]) if clusters[i] != -1 else None
                                    cursor.execute(
                                        "UPDATE memories SET cluster_label = ? WHERE id = ?",
                                        (cluster_label, memory["id"])
                                    )
                                    memory["cluster"] = cluster_label
                                conn.commit()
                            self.last_cluster_time = datetime.now()
                            logger.info(f"Reclustered {embeddings.shape[0]} memories with {len(set(clusters)) - (-1 in clusters)} clusters")
                        finally:
                            self.db_pool.release_connection(conn)
            except asyncio.TimeoutError:
                logger.error("Clustering timed out")
            except Exception as e:
                logger.error(f"Error during periodic clustering: {e}\n{traceback.format_exc()}")

    async def _get_cluster_label(self, embedding):
        if not self.clustering_enabled or not self.memory_bank:
            return None
        
        async with self.memory_async_lock:
            embeddings = self.xp.array([m["embedding"] for m in self.memory_bank], dtype=self.xp.float32)
            if len(embeddings) < 3:
                return None
            distances = self.xp.linalg.norm(embeddings - embedding, axis=1)
            nearest_idx = self.xp.argmin(distances)
            if distances[nearest_idx] < self.relevance_threshold:
                return self.memory_bank[nearest_idx]["cluster"]
        return None

    async def run(self, data):
        command = data.get("command", "")
        input_text = data.get("input_text", "")
        perspective = data.get("perspective", "Vira")
        emotion_data = data.get("emotion_data")
        plugin_manager = data.get("plugin_manager")
        
        if plugin_manager and not self.plugin_manager:
            self.plugin_manager = plugin_manager
            logger.debug("Plugin manager set in hippo_plugin")
        
        if command == "encode":
            is_autonomous = data.get("is_autonomous", False)
            return await self.encode_episodic_memory(input_text, perspective, emotion_data, is_autonomous=is_autonomous)
        elif command == "retrieve":
            return await self.retrieve_memories(input_text)
        elif command == "preload":
            return await self.preload_memories()
        elif command == "patterns":
            time_window = data.get("time_window_hours", 24)
            min_occurrences = data.get("min_occurrences", 2)
            return await self.retrieve_emotional_patterns(time_window, min_occurrences)
        elif command == "unresolved":
            intensity_threshold = data.get("intensity_threshold")
            return await self.retrieve_unresolved_issues(intensity_threshold)
        elif command == "outcomes":
            limit = data.get("limit", 5)
            return await self.retrieve_action_outcomes(limit)
        else:
            logger.warning(f"Unknown command: {command}")
            return {"status": "unknown_command", "plugin_name": "hippo_plugin"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = {
        "memory_settings": {
            "database_name": "chat_history_populated.db",
            "max_memory_bank_size": 2000,
            "memory_relevance_threshold": 0.5,
            "startup_memory_turns": 20,
            "clustering": {
                "enabled": True,
                "min_cluster_size": 3,
                "min_samples": 2,
                "cluster_selection_method": "eom",
                "cluster_interval": 300
            }
        },
        "emotion_settings": {"intensity_threshold": 3},
        "device": vira4_6t.device
    }
    asyncio.run(vira4_6t.setup_vira())
    plugin = Plugin(config)
    
    async def test():
        emotion_data = {"emotion_type": "sad", "emotion_intensity": 8, "context": "Test context"}
        result = await plugin.run({"command": "encode", "input_text": "I had a tough day", "perspective": "User", "emotion_data": emotion_data})
        print(f"Encode result: {result}")
        result = await plugin.run({"command": "encode", "input_text": "Take it easy tonight", "perspective": "Core", "emotion_data": {"emotion_type": "happy", "emotion_intensity": 6}})
        print(f"Encode result: {result}")
        result = await plugin.run({"command": "retrieve", "input_text": "How’s my day going?"})
        print(f"Retrieve result: {result}")
        result = await plugin.run({"command": "preload"})
        print(f"Preload result: {result}")
    
    asyncio.run(test())