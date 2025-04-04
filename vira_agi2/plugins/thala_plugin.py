import logging
import numpy as np
import cupy as cp
import sqlite3
import vira4_6t
import traceback
from datetime import datetime
import torch
import asyncio

logger = logging.getLogger(__name__)

try:
    from cuml.manifold import UMAP as CumlUMAP
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    logger.warning("cuml not available; falling back to basic weighting.")

class Plugin:    
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", None)
        self.default_weights = config.get("priority_settings", {}).get("weights", {
            "fear": 0.9, "angry": 0.8, "sad": 0.7, "excited": 0.5, "happy": 0.4, "unknown": 0.3
        })
        self.memory_window_size = config.get("priority_settings", {}).get("memory_window_size", 10)
        self.memory_cache = None
        self.cache_timestamp = None
        self.cache_refresh_interval = 60
        self.plugin_manager = None
        
        # Use CuPy if CUDA is available, else fall back to PyTorch
        self.xp = cp if (self.device == "cuda" and cp.cuda.is_available()) else torch
        logger.info(f"Thalamus plugin initialized for input prioritization with dynamic weights, device: {'GPU' if self.device == 'cuda' else 'CPU'}")
        
        if CUML_AVAILABLE and self.xp is cp:
            self.umap = CumlUMAP(n_components=2, n_neighbors=5, random_state=42)
            logger.info("Initialized cuml.UMAP for GPU-accelerated embedding reduction.")
        else:
            self.umap = None
            logger.info("Using basic weighting (cuml unavailable or CPU mode).")
        
        conn = vira4_6t.db_pool.get_connection()
        try:
            with conn:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)")
                logger.debug("Ensured index on memories.timestamp for query performance.")
        except sqlite3.Error as e:
            logger.error(f"Error creating index on memories.timestamp: {e}\n{traceback.format_exc()}")
        finally:
            vira4_6t.db_pool.release_connection(conn)

    async def _update_dynamic_weights(self):  # Moved out of __init__ and made async
        current_time = datetime.now()
        
        if (self.memory_cache is not None and self.cache_timestamp and 
            (current_time - self.cache_timestamp).total_seconds() < self.cache_refresh_interval):
            recent_memories = self.memory_cache
        else:
            conn = vira4_6t.db_pool.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT emotion_type, emotion_intensity, cluster_label, embedding 
                    FROM memories 
                    WHERE emotion_type IS NOT NULL 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (self.memory_window_size,))
                recent_memories = cursor.fetchall()
                self.memory_cache = recent_memories
                self.cache_timestamp = current_time
                logger.debug(f"Updated memory cache with {len(recent_memories)} recent memories including embeddings.")
            except sqlite3.Error as e:
                logger.error(f"Error fetching recent memories for dynamic weights: {e}\n{traceback.format_exc()}")
                vira4_6t.db_pool.release_connection(conn)
                return self.default_weights.copy()
            finally:
                vira4_6t.db_pool.release_connection(conn)
        
        if not recent_memories:
            return self.default_weights.copy()
        
        # Use CuPy for GPU acceleration
        emotions = [m[0] for m in recent_memories]
        intensities = self.xp.array([m[1] or 0 for m in recent_memories], dtype=self.xp.float32)
        clusters = [m[2] for m in recent_memories]
        embeddings = [self.xp.frombuffer(m[3], dtype=self.xp.float32) for m in recent_memories]
        
        total_memories = len(recent_memories)
        # Unique emotions and counts with CuPy
        unique_emotions, indices = self.xp.unique(self.xp.array([hash(e) for e in emotions]), return_inverse=True)
        emotion_counts_array = self.xp.bincount(indices)
        emotion_counts = {emotions[i]: emotion_counts_array[self.xp.where(unique_emotions == hash(emotions[i]))[0]].item() 
                         for i in range(len(emotions)) if hash(emotions[i]) in unique_emotions}
        
        # Intensity sums per emotion with CuPy
        emotion_intensities = {}
        for e in set(emotions):
            mask = self.xp.array([1 if emo == e else 0 for emo in emotions], dtype=self.xp.int32)
            emotion_intensities[e] = self.xp.sum(intensities * mask).item()
        
        # Cluster counts with CuPy
        valid_clusters = [c for c in clusters if c is not None and c != -1]
        cluster_counts = {}
        if valid_clusters:
            unique_clusters, cluster_indices = self.xp.unique(self.xp.array(valid_clusters), return_inverse=True)
            cluster_counts_array = self.xp.bincount(cluster_indices)
            cluster_counts = {unique_clusters[i].item(): cluster_counts_array[i].item() for i in range(len(unique_clusters))}
        
        dynamic_weights = self.default_weights.copy()
        
        # UMAP reduction for embedding patterns if cuml is available
        umap_boost = {}
        if CUML_AVAILABLE and self.umap and self.xp is cp and len(embeddings) >= 5:
            try:
                embedding_array = self.xp.stack(embeddings)
                reduced_embeddings = self.umap.fit_transform(embedding_array)
                centroid = self.xp.mean(reduced_embeddings, axis=0)
                distances = self.xp.linalg.norm(reduced_embeddings - centroid, axis=1)
                max_distance = self.xp.max(distances)
                if max_distance > 0:
                    umap_scores = 1 - (distances / max_distance)
                    for i, emotion in enumerate(emotions):
                        if emotion in dynamic_weights:
                            umap_boost[emotion] = umap_boost.get(emotion, 0) + umap_scores[i].item()
                    for emotion in umap_boost:
                        umap_boost[emotion] /= emotion_counts.get(emotion, 1)
                        umap_boost[emotion] = min(0.1, umap_boost[emotion] * 0.2)
                    logger.debug(f"UMAP boost computed: {umap_boost}")
            except Exception as e:
                logger.error(f"UMAP reduction error: {e}\n{traceback.format_exc()}")

        # Temporal adjustment (inserted here)
        temporal_boost = {}
        if self.plugin_manager:
            try:
                temporal_result = await self.plugin_manager.execute_specific_plugin(
                    "temporal_plugin", {"command": "predict"}
                )
                user_pred = temporal_result.get("predictions", {}).get("user", {})
                if "next_emotion" in user_pred:
                    next_emotion = user_pred["next_emotion"].lower()
                    next_intensity = user_pred["next_intensity"] / 10.0
                    temporal_boost[next_emotion] = min(0.15, next_intensity * 0.3)  # Cap at 0.15
                    logger.debug(f"Temporal boost for {next_emotion}: {temporal_boost[next_emotion]:.2f}")
                
                rhythm_result = await self.plugin_manager.execute_specific_plugin(
                    "temporal_plugin", {"command": "get_rhythm"}
                )
                user_rhythm = rhythm_result.get("rhythms", {}).get("user", {})
                if "dominant_emotion" in user_rhythm:
                    dominant_emotion = user_rhythm["dominant_emotion"].lower()
                    if dominant_emotion in dynamic_weights:
                        temporal_boost[dominant_emotion] = temporal_boost.get(dominant_emotion, 0) + 0.1
                        logger.debug(f"Rhythm boost for {dominant_emotion}: {temporal_boost[dominant_emotion]:.2f}")
            except Exception as e:
                logger.error(f"Error fetching temporal data: {e}\n{traceback.format_exc()}")

        # Apply frequency, intensity, UMAP, and temporal adjustments
        for emotion in emotion_counts:
            if emotion in dynamic_weights:
                frequency = emotion_counts[emotion] / total_memories
                avg_intensity = emotion_intensities[emotion] / emotion_counts[emotion] / 10.0
                adjustment = (frequency * 0.5 + avg_intensity * 0.5) * 0.2
                dynamic_weights[emotion] = min(1.0, dynamic_weights[emotion] * (1 + adjustment))
                if emotion in umap_boost:
                    dynamic_weights[emotion] = min(1.0, dynamic_weights[emotion] + umap_boost[emotion])
                if emotion in temporal_boost:
                    dynamic_weights[emotion] = min(1.0, dynamic_weights[emotion] + temporal_boost[emotion])
                logger.debug(f"Dynamic weight for {emotion}: {dynamic_weights[emotion]:.2f} (Freq: {frequency:.2f}, Avg Intensity: {avg_intensity:.2f}, UMAP: {umap_boost.get(emotion, 0):.2f}, Temporal: {temporal_boost.get(emotion, 0):.2f})")
        
        # Apply cluster boosts
        if cluster_counts:
            dominant_cluster = max(cluster_counts, key=cluster_counts.get)
            cluster_memories = [m for m in recent_memories if m[2] == dominant_cluster]
            cluster_emotion_counts = {}
            for emotion_type, _, _, _ in cluster_memories:
                cluster_emotion_counts[emotion_type] = cluster_emotion_counts.get(emotion_type, 0) + 1
            
            cluster_size = len(cluster_memories)
            for emotion in cluster_emotion_counts:
                if emotion in dynamic_weights:
                    cluster_frequency = cluster_emotion_counts[emotion] / cluster_size
                    cluster_boost = cluster_frequency * 0.1
                    dynamic_weights[emotion] = min(1.0, dynamic_weights[emotion] + cluster_boost)
                    logger.debug(f"Cluster boost for {emotion} in cluster {dominant_cluster}: +{cluster_boost:.2f}, New weight: {dynamic_weights[emotion]:.2f}")
        
        logger.debug(f"Dynamic weights computed on {'GPU' if self.xp is cp else 'CPU'} with {'cuml.UMAP' if umap_boost else 'no UMAP'}")
        return dynamic_weights

    async def _calculate_priority(self, input_text, emotion_data=None):
        if not input_text:
            logger.warning("No input text provided for priority calculation.")
            return 0.3
        
        weights = await self._update_dynamic_weights()
        
        if not emotion_data or "emotion_type" not in emotion_data:
            logger.warning(f"No valid emotion data for: {input_text}. Attempting direct analysis.")
            if self.plugin_manager:
                try:
                    emotion_result = await self.plugin_manager.execute_specific_plugin(
                        "vira_emotion_plugin", {"command": "analyze", "text": input_text}
                    )
                    if emotion_result and "error" not in emotion_result:
                        emotion_data = {
                            "emotion_type": emotion_result.get("emotion_type", "unknown"),
                            "emotion_intensity": emotion_result.get("emotion_intensity", 0)
                        }
                        logger.info(f"Direct emotion analysis for '{input_text}': {emotion_data}")
                    else:
                        logger.warning(f"Emotion analysis failed: {emotion_result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error analyzing emotion directly: {e}\n{traceback.format_exc()}")
            if not emotion_data or "emotion_type" not in emotion_data:
                logger.info(f"No emotion data available for '{input_text}'. Defaulting to unknown.")
                return weights["unknown"]
        
        emotion_type = emotion_data["emotion_type"].lower()
        emotion_intensity = emotion_data.get("emotion_intensity", 0) / 10.0
        
        # Emotional summary boost
        summary_boost = 0.0
        if self.plugin_manager:
            try:
                summary_result = await self.plugin_manager.execute_specific_plugin(
                    "vira_emotion_plugin", {"command": "summary", "time_window_hours": 24}
                )
                emotions = summary_result.get("emotions", [])
                for e in emotions:
                    if e["emotion"].lower() == emotion_type and e["significant"]:
                        summary_boost = min(0.15, e["count"] * 0.05)
                        logger.debug(f"Emotional summary boost for {emotion_type}: {summary_boost:.2f}")
                        break
            except Exception as e:
                logger.error(f"Error fetching emotional summary: {e}\n{traceback.format_exc()}")
    
        # Cluster and unresolved issues boost
        cluster_boost = 0.0
        unresolved_boost = 0.0
        if self.plugin_manager:
            try:
                memory_result = await self.plugin_manager.execute_specific_plugin(
                    "hippo_plugin", {"command": "retrieve", "input_text": input_text}
                )
                episodic_memories = memory_result.get("episodic", [])
                if episodic_memories:
                    cluster_labels = {m["cluster"] for m in episodic_memories if m.get("cluster") is not None and m["cluster"] != -1}
                    if cluster_labels:
                        dominant_cluster = max(
                            cluster_labels,
                            key=lambda c: sum(1 for m in episodic_memories if m["cluster"] == c and m["emotion_type"].lower() == emotion_type),
                            default=None
                        )
                        if dominant_cluster:
                            cluster_size = sum(1 for m in episodic_memories if m["cluster"] == dominant_cluster)
                            cluster_boost = min(0.2, cluster_size * 0.05)
                            logger.debug(f"Cluster {dominant_cluster} boost: {cluster_boost:.2f}")
    
                unresolved_result = await self.plugin_manager.execute_specific_plugin(
                    "hippo_plugin", {"command": "unresolved"}
                )
                unresolved = unresolved_result.get("unresolved", [])
                if any(u["content"] == input_text and u["emotion_type"].lower() == emotion_type for u in unresolved):
                    unresolved_boost = 0.1
                    logger.debug(f"Unresolved issue boost for '{input_text}': {unresolved_boost:.2f}")
            except Exception as e:
                logger.error(f"Error checking memory context: {e}\n{traceback.format_exc()}")
        
        emotion_weight = weights.get(emotion_type, weights["unknown"])
        priority_score = (emotion_weight * 0.6 + emotion_intensity * 0.3 + 
                          cluster_boost + summary_boost + unresolved_boost)
        
        priority_score = max(0.0, min(1.0, priority_score))
        logger.info(f"Priority for '{input_text}': {priority_score:.2f} (Emotion: {emotion_type}, Intensity: {emotion_intensity:.2f}, Weight: {emotion_weight:.2f}, Cluster: {cluster_boost:.2f}, Summary: {summary_boost:.2f}, Unresolved: {unresolved_boost:.2f})")
        return priority_score
        
    async def prioritize_goal(self, goal_text, goal_context):
        """Calculate priority for an autonomous goal."""
        emotion_data = None
        if self.plugin_manager:
            try:
                emotion_result = await self.plugin_manager.execute_specific_plugin(
                    "vira_emotion_plugin", {"command": "analyze", "text": goal_text}
                )
                if emotion_result and "error" not in emotion_result:
                    emotion_data = {
                        "emotion_type": emotion_result["emotion_type"],
                        "emotion_intensity": emotion_result["emotion_intensity"]
                    }
            except Exception as e:
                logger.error(f"Error analyzing goal emotion: {e}\n{traceback.format_exc()}")
    
        priority = await self._calculate_priority(goal_text, emotion_data)
        # Boost if tied to unresolved issues or high-priority memory
        if "support" in goal_text.lower() or "encourage" in goal_text.lower():
            unresolved_result = await self.plugin_manager.execute_specific_plugin(
                "hippo_plugin", {"command": "unresolved"}
            )
            if any(goal_context in u["context"] for u in unresolved_result.get("unresolved", [])):
                priority = min(1.0, priority + 0.2)
                logger.debug(f"Goal '{goal_text}' boosted by unresolved context: {priority:.2f}")
        
        return priority

    async def run(self, data):
        command = data.get("command", "")
        input_text = data.get("input_text", "")
        emotion_data = data.get("emotion_data")
        plugin_manager = data.get("plugin_manager")
        
        if plugin_manager and not self.plugin_manager:
            self.plugin_manager = plugin_manager
            logger.debug("Plugin manager set in thala_plugin.")
        
        if command == "prioritize" and input_text:
            try:
                priority_score = await self._calculate_priority(input_text, emotion_data)
                return {
                    "priority": priority_score,
                    "input_text": input_text,
                    "status": "prioritized",
                    "plugin_name": "thala_plugin"
                }
            except Exception as e:
                logger.error(f"Error calculating priority for '{input_text}': {e}\n{traceback.format_exc()}")
                return {
                    "priority": 0.3,
                    "input_text": input_text,
                    "status": "error",
                    "error": str(e),
                    "plugin_name": "thala_plugin"
                }
        elif command == "prioritize_goal":
            goal_text = data.get("goal_text", "")
            goal_context = data.get("goal_context", "")
            if not goal_text:
                return {"priority": 0.3, "status": "no_goal_text", "plugin_name": "thala_plugin"}
            try:
                priority = await self.prioritize_goal(goal_text, goal_context)
                return {
                    "priority": priority,
                    "goal_text": goal_text,
                    "status": "goal_prioritized",
                    "plugin_name": "thala_plugin"
                }
            except Exception as e:
                logger.error(f"Error prioritizing goal '{goal_text}': {e}\n{traceback.format_exc()}")
                return {
                    "priority": 0.3,
                    "goal_text": goal_text,
                    "status": "error",
                    "error": str(e),
                    "plugin_name": "thala_plugin"
                }
        else:
            logger.warning(f"Invalid command '{command}' or no input text provided.")
            return {"priority": 0.3, "status": "no_input_or_invalid_command", "plugin_name": "thala_plugin"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = {
        "plugin_execution_order": ["vira_emotion_plugin", "thala_plugin"],
        "priority_settings": {
            "weights": {
                "fear": 0.9, "angry": 0.8, "sad": 0.7, "excited": 0.5, "happy": 0.4, "unknown": 0.3
            }
        }
    }
    
    asyncio.run(vira4_6t.setup_vira())
    plugin = Plugin(config)
    
    async def test():
        test_inputs = [
            {"command": "prioritize", "input_text": "I’m really upset right now", "emotion_data": {"emotion_type": "sad", "emotion_intensity": 8}},
            {"command": "prioritize", "input_text": "Hey, how’s it going?", "emotion_data": {"emotion_type": "happy", "emotion_intensity": 6}},
            {"command": "prioritize", "input_text": "I’m scared about tomorrow", "emotion_data": {"emotion_type": "fear", "emotion_intensity": 9}},
            {"command": "prioritize", "input_text": "I’m really upset right now"},
            {"command": "invalid", "input_text": "Test"},
            {"command": "prioritize", "input_text": ""}
        ]
        
        for test in test_inputs:
            result = await plugin.run(test)
            print(f"Input: {test}\nResult: {result}\n")
    
    asyncio.run(test())