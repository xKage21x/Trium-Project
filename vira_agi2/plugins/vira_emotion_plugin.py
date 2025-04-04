import logging
import asyncio
import torch
import cupy as cp
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import deque
import vira4_6t
import traceback
from functools import lru_cache
import concurrent.futures

logger = logging.getLogger(__name__)

try:
    from cuml.cluster import KMeans as CumlKMeans
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    logger.warning("cuml not available; falling back to manual clustering.")

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

simplified_emotion_map = {
    "Admiration": "happy",
    "Amusement": "happy",
    "Anger": "angry",
    "Annoyance": "angry",
    "Approval": "happy",
    "Caring": "happy",
    "Confusion": "sad",
    "Curiosity": "happy",
    "Desire": "happy",
    "Disappointment": "sad",
    "Disapproval": "angry",
    "Disgust": "angry",
    "Embarrassment": "sad",
    "Excitement": "excited",
    "Fear": "fear",
    "Gratitude": "happy",
    "Grief": "sad",
    "Joy": "happy",
    "Love": "happy",
    "Nervousness": "fear",
    "Optimism": "happy",
    "Pride": "happy",
    "Realization": "happy",
    "Relief": "happy",
    "Remorse": "sad",
    "Sadness": "sad",
    "Surprise": "excited",
    "Neutral": "unknown"
}

class Plugin:
    def __init__(self, config):
        self.config = config
        self.model_name = config.get("model_settings", {}).get("emobert_model", "j-hartmann/emotion-english-distilroberta-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.lock = asyncio.Lock()
        
        self.xp = cp if (self.device.type == "cuda" and cp.cuda.is_available()) else torch
        
        self._initialize_model()
        
        self.context_memory = deque(maxlen=5)
        self.feedback_buffer = deque(maxlen=100)
        self.custom_emotion_map = {}
        self.emotion_clusters = {}
        self.learning_rate = config.get("emotion_settings", {}).get("learning_rate", 1e-5)
        self.cluster_threshold = config.get("emotion_settings", {}).get("cluster_threshold", 0.7)
        self.max_clusters = config.get("emotion_settings", {}).get("max_clusters", 20)
        self.last_cluster_time = 0
        self.cluster_interval = 60
        
        if CUML_AVAILABLE and self.xp is cp:
            self.kmeans = CumlKMeans(n_clusters=min(10, self.max_clusters), random_state=42, init="k-means++", max_iter=100)
            logger.info("Initialized cuml.KMeans for GPU-accelerated clustering.")
        else:
            self.kmeans = None
            logger.info("Using manual clustering (cuml unavailable or CPU mode).")

    def _initialize_model(self):
        try:
            torch.cuda.empty_cache()
            if self.device.type == "cuda":
                torch.cuda.set_per_process_memory_fraction(0.5)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            dummy_input = self.tokenizer("Hello", return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**dummy_input)
            logger.info(f"RoBERTa emotional analyzer ({self.model_name}) loaded on {self.device}")
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM: {e}. Falling back to CPU.")
            self.device = torch.device("cpu")
            torch.cuda.empty_cache()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Reloaded on CPU after OOM")
        except Exception as e:
            logger.error(f"Model init error: {e}\n{traceback.format_exc()}")
            self.model = None
            self.tokenizer = None

    @lru_cache(maxsize=512)
    def _analyze_emotion_cached(self, text_normalized):
        if not self.model or not self.tokenizer:
            return "unknown", 0, "Vira", "Model not initialized", "unknown"
        
        inputs = self.tokenizer(text_normalized, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        emotion_idx = scores.argmax().item()
        raw_emotion = self.model.config.id2label[emotion_idx].lower()
        score = scores[0, emotion_idx].item() * 10  # Scale to 0-10
        return raw_emotion, score, "Vira", f"Emotion inferred from: {text_normalized[:50]}...", simplified_emotion_map.get(raw_emotion.capitalize(), "unknown")

    async def analyze_emotion(self, text):
        if not text:
            logger.warning("No text for analysis.")
            return "unknown", 0, "Vira", "No text provided", "unknown"
        
        if self.model is None or self.tokenizer is None:
            logger.error("Model not initialized.")
            return "unknown", 0, "Vira", "Model not initialized", "unknown"
        
        async with asyncio.timeout(2):
            text_normalized = " ".join(text.lower().split())
            logger.debug(f"Analyzing emotion for '{text}'")
            
            # Base analysis
            raw_emotion, score, perspective, context, simplified_emotion = self._analyze_emotion_cached(text_normalized)
            
            # Contextual adjustments
            if vira4_6t.plugin_manager:
                # Temporal rhythm
                temporal_result = await vira4_6t.plugin_manager.execute_specific_plugin(
                    "temporal_plugin", {"command": "get_rhythm"}
                )
                user_rhythm = temporal_result.get("rhythms", {}).get("user", {})
                if user_rhythm.get("dominant_emotion"):
                    rhythm_emotion = user_rhythm["dominant_emotion"].lower()
                    context += f" (Rhythm: {rhythm_emotion})"
                    if rhythm_emotion in emotion_to_polyvagal and rhythm_emotion != raw_emotion:
                        raw_emotion = rhythm_emotion
                        score = max(score, 5)  # Minimum boost from rhythm
                
                # Memory patterns
                hippo_result = await vira4_6t.plugin_manager.execute_specific_plugin(
                    "hippo_plugin", {"command": "patterns", "time_window_hours": 24}
                )
                patterns = hippo_result.get("patterns", [])
                if patterns:
                    top_pattern = max([p for p in patterns if p["type"] == "emotion"], key=lambda p: p["count"], default=None)
                    if top_pattern and top_pattern["significant"]:
                        pattern_emotion = top_pattern["emotion"].lower()
                        context += f" (Pattern: {pattern_emotion}, count: {top_pattern['count']})"
                        if pattern_emotion != raw_emotion and top_pattern["count"] > 3:
                            raw_emotion = pattern_emotion
                            score = max(score, top_pattern["count"] * 2)
                
                # Thala dynamic weights
                thala_result = await vira4_6t.plugin_manager.execute_specific_plugin(
                    "thala_plugin", {"command": "get_dynamic_weights"}
                )
                weights = thala_result.get("weights", {}) if thala_result else {}
                if weights:
                    # Map raw_emotion to simplified_emotion for Thala compatibility if needed
                    thala_emotion_key = simplified_emotion_map.get(raw_emotion.capitalize(), raw_emotion.lower())
                    weight = weights.get(thala_emotion_key, 1.0)  # Default to no adjustment
                    adjusted_score = min(10, max(0, score * weight))  # Bound between 0-10
                    if weight != 1.0:
                        context += f" (Thala weight: {weight:.2f}, adjusted intensity: {adjusted_score:.1f})"
                        score = adjusted_score
                        logger.debug(f"Thala adjusted '{raw_emotion}' intensity from {score:.1f} to {adjusted_score:.1f} with weight {weight:.2f}")
                    # Check if Thala suggests a different dominant emotion
                    max_weighted_emotion = max(weights.items(), key=lambda x: x[1])[0] if weights else None
                    if max_weighted_emotion and max_weighted_emotion != thala_emotion_key and weights[max_weighted_emotion] > 1.5:
                        for mapped_emotion, simplified in simplified_emotion_map.items():
                            if simplified == max_weighted_emotion:
                                raw_emotion = mapped_emotion.lower()
                                score = max(score, 5 * weights[max_weighted_emotion])  # Boost for strong Thala signal
                                context += f" (Thala override to {raw_emotion})"
                                break
            
            # Custom mapping and clustering
            if text in self.custom_emotion_map:
                raw_emotion, score = self.custom_emotion_map[text]
                context += " (Custom mapping)"
            
            word_count = len(text.split())
            if word_count >= 10 and (asyncio.get_event_loop().time() - self.last_cluster_time) > self.cluster_interval:
                async with self.lock:
                    cluster_context = await self._get_cluster_context(text)
                    if cluster_context and cluster_context["count"] >= 3:
                        cluster_emotion = cluster_context["dominant_emotion"]
                        cluster_intensity = cluster_context["average_intensity"]
                        if cluster_emotion != raw_emotion and cluster_intensity > self.cluster_threshold * 10:
                            raw_emotion = cluster_emotion
                            score = max(score, cluster_intensity)
                            context += f" (Cluster: {cluster_emotion}, count: {cluster_context['count']})"
                    self.last_cluster_time = asyncio.get_event_loop().time()
            
            if word_count >= 10:
                nuanced_emotion = await self._detect_nuanced_emotion(text, raw_emotion, None)
                if nuanced_emotion:
                    raw_emotion = nuanced_emotion
                    context += f" (Nuanced: {nuanced_emotion})"
            
            mapped_emotion, perspective, polyvagal_state = emotion_to_polyvagal.get(raw_emotion.lower(), ("Neutral", "Vira", "vagal (connection)"))
            simplified_emotion = simplified_emotion_map.get(mapped_emotion, "unknown")
            
            logger.info(f"Emotion analysis - Text: {text}, Raw: {raw_emotion}, Intensity: {int(score)}, Simplified: {simplified_emotion}")
            return raw_emotion, int(score), perspective, context, simplified_emotion

    async def _get_cluster_context(self, text):
        if not vira4_6t.plugin_manager or "hippo_plugin" not in vira4_6t.plugin_manager.plugins:
            return None
        
        memory_result = await vira4_6t.plugin_manager.execute_specific_plugin(
            "hippo_plugin", {"command": "retrieve", "input_text": text}
        )
        episodic_memories = memory_result.get("episodic", [])
        if not episodic_memories:
            return None
        
        emotions = [memory["emotion_type"] for memory in episodic_memories[:5]]
        intensities = self.xp.array([memory["emotion_intensity"] for memory in episodic_memories[:5]], dtype=self.xp.float32)
        
        unique_emotions, indices = self.xp.unique(self.xp.array([hash(e) for e in emotions]), return_inverse=True)
        emotion_counts = self.xp.bincount(indices)
        
        emotion_dict = {emotions[self.xp.where(indices == i)[0][0]]: int(emotion_counts[i]) for i in range(len(unique_emotions))}
        if not emotion_dict:
            return None
        
        total_intensity = self.xp.sum(intensities).item()
        count = self.xp.sum(emotion_counts).item()
        average_intensity = total_intensity / count if count > 0 else 0
        
        dominant_emotion = max(emotion_dict.items(), key=lambda x: x[1])[0]
        
        return {
            "dominant_emotion": dominant_emotion,
            "average_intensity": average_intensity,
            "count": count
        }

    async def _detect_nuanced_emotion(self, text, raw_emotion, scores):
        async with self.lock:
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: torch.tensor(vira4_6t.embedder.encode([text])[0], device=self.device)
            )
            embedding_xp = self.xp.asarray(embedding) if self.xp is cp else embedding
            
            if CUML_AVAILABLE and self.kmeans and self.xp is cp and self.emotion_clusters:
                centroids = cp.array([centroid for _, (centroid, _) in self.emotion_clusters.items()], dtype=cp.float32)
                if len(centroids) > self.max_clusters:
                    # Prune least frequent clusters
                    sorted_clusters = sorted(self.emotion_clusters.items(), key=lambda x: x[1][1], reverse=True)
                    self.emotion_clusters = dict(sorted_clusters[:self.max_clusters])
                    centroids = cp.array([c for _, (c, _) in self.emotion_clusters.items()], dtype=cp.float32)
                
                self.kmeans.n_clusters = max(1, min(self.max_clusters, len(centroids)))
                self.kmeans.fit(centroids)
                distances = self.kmeans.transform(embedding_xp[None])[0]
                similarities = 1 / (1 + distances)
                max_similarity_idx = cp.argmax(similarities)
                similarity = similarities[max_similarity_idx].item()
                matched_cluster = list(self.emotion_clusters.keys())[max_similarity_idx.item()]
                
                if similarity > self.cluster_threshold:
                    count = self.emotion_clusters[matched_cluster][1]
                    new_centroid = (centroids[max_similarity_idx] * count + embedding_xp) / (count + 1)
                    self.emotion_clusters[matched_cluster] = (cp.asnumpy(new_centroid), count + 1)
                    return matched_cluster
                else:
                    self.emotion_clusters[raw_emotion] = (cp.asnumpy(embedding_xp), 1)
                    return raw_emotion
            else:
                matched_cluster = None
                for cluster_emotion, (centroid, count) in self.emotion_clusters.items():
                    centroid_xp = self.xp.asarray(torch.tensor(centroid, device=self.device)) if self.xp is cp else torch.tensor(centroid, device=self.device)
                    similarity = self.xp.dot(embedding_xp, centroid_xp) / (self.xp.linalg.norm(embedding_xp) * self.xp.linalg.norm(centroid_xp) + 1e-8)
                    if similarity > self.cluster_threshold:
                        matched_cluster = cluster_emotion
                        new_centroid = (centroid_xp * count + embedding_xp) / (count + 1)
                        self.emotion_clusters[cluster_emotion] = (self.xp.asnumpy(new_centroid) if self.xp is cp else new_centroid.cpu().numpy(), count + 1)
                        break
                
                if not matched_cluster and len(self.emotion_clusters) < self.max_clusters:
                    self.emotion_clusters[raw_emotion] = (self.xp.asnumpy(embedding_xp) if self.xp is cp else embedding.cpu().numpy(), 1)
                    matched_cluster = raw_emotion
                elif not matched_cluster:
                    # Replace least frequent cluster
                    least_frequent = min(self.emotion_clusters.items(), key=lambda x: x[1][1])
                    del self.emotion_clusters[least_frequent[0]]
                    self.emotion_clusters[raw_emotion] = (self.xp.asnumpy(embedding_xp) if self.xp is cp else embedding.cpu().numpy(), 1)
                    matched_cluster = raw_emotion
                
                return matched_cluster

    async def process_feedback(self, text, user_emotion, user_intensity=None):
        async with self.lock:
            user_intensity = user_intensity or 5
            self.feedback_buffer.append((text, user_emotion, user_intensity))
            self.custom_emotion_map[text] = (user_emotion, user_intensity)
            logger.info(f"Updated custom emotion map: '{text}' -> {user_emotion}, {user_intensity}")
    
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: torch.tensor(vira4_6t.embedder.encode([text])[0], device=self.device)
            )
            embedding_xp = self.xp.asarray(embedding) if self.xp is cp else embedding
    
            if CUML_AVAILABLE and self.kmeans and self.xp is cp:
                if self.emotion_clusters:
                    centroids = cp.array([centroid for _, (centroid, _) in self.emotion_clusters.items()], dtype=cp.float32)
                    if len(centroids) > self.max_clusters:
                        sorted_clusters = sorted(self.emotion_clusters.items(), key=lambda x: x[1][1], reverse=True)
                        self.emotion_clusters = dict(sorted_clusters[:self.max_clusters])
                        centroids = cp.array([c for _, (c, _) in self.emotion_clusters.items()], dtype=cp.float32)
                    
                    self.kmeans.n_clusters = max(1, min(self.max_clusters, len(centroids)))
                    self.kmeans.fit(centroids)
                    distances = self.kmeans.transform(embedding_xp[None])[0]
                    similarities = 1 / (1 + distances)
                    max_similarity_idx = cp.argmax(similarities)
                    similarity = similarities[max_similarity_idx].item()
                    matched_cluster = list(self.emotion_clusters.keys())[max_similarity_idx.item()]
                    if similarity > self.cluster_threshold:
                        count = self.emotion_clusters[matched_cluster][1]
                        new_centroid = (centroids[max_similarity_idx] * count + embedding_xp) / (count + 1)
                        self.emotion_clusters[matched_cluster] = (new_centroid, count + 1)
                    else:
                        self.emotion_clusters[user_emotion] = (embedding_xp, 1)
                else:
                    self.emotion_clusters[user_emotion] = (embedding_xp, 1)
    
            if vira4_6t.plugin_manager:
                await vira4_6t.plugin_manager.execute_specific_plugin(
                    "hippo_plugin", {
                        "command": "encode",
                        "input_text": text,
                        "perspective": "User",
                        "emotion_data": {
                            "emotion_type": user_emotion,
                            "emotion_intensity": user_intensity,
                            "context": "Feedback processed"
                        }
                    }
                )
    
            if len(self.feedback_buffer) >= 5:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                self.model.train()
                for fb_text, fb_emotion, fb_intensity in list(self.feedback_buffer)[-5:]:
                    inputs = self.tokenizer(fb_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
                    emotion_idx = {v.lower(): k for k, v in self.model.config.id2label.items()}.get(fb_emotion.lower())
                    if emotion_idx is not None:
                        labels = torch.tensor([emotion_idx], dtype=torch.long).to(self.device)
                        outputs = self.model(**inputs, labels=labels)
                        loss = outputs.loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        logger.debug(f"Fine-tuned: '{fb_text}' -> {fb_emotion}, Loss: {loss.item()}")
                self.model.eval()
                # Keep older feedback for future batches
                self.feedback_buffer = deque(list(self.feedback_buffer)[:-5], maxlen=100)

    async def get_emotional_summary(self, time_window_hours=24):
        if not vira4_6t.plugin_manager:
            return {"summary": "No plugin manager", "emotions": [], "plugin_name": "vira_emotion_plugin"}
    
        hippo_result = await vira4_6t.plugin_manager.execute_specific_plugin(
            "hippo_plugin", {"command": "patterns", "time_window_hours": time_window_hours}
        )
        patterns = hippo_result.get("patterns", [])
        unresolved_result = await vira4_6t.plugin_manager.execute_specific_plugin(
            "hippo_plugin", {"command": "unresolved"}
        )
        unresolved = unresolved_result.get("unresolved", [])
        
        emotions = [
            {
                "emotion": p["emotion"],
                "count": p["count"],
                "significant": p["significant"],
                "polyvagal": emotion_to_polyvagal.get(p["emotion"].lower(), ("Neutral", "Vira", "vagal (connection)"))[2]
            }
            for p in patterns if p["type"] == "emotion"
        ]
        
        unresolved_summary = f" ({len(unresolved)} unresolved)" if unresolved else ""
        summary = f"Past {time_window_hours}h{unresolved_summary}: " + "; ".join(f"{e['emotion']} ({e['count']}x, {e['polyvagal']})" for e in emotions) if emotions else "No recent emotional patterns"
        logger.info(summary)
        return {"summary": summary, "emotions": emotions, "plugin_name": "vira_emotion_plugin"}
        
    async def get_clusters(self):
        """Retrieve current emotion clusters for external display or analysis."""
        async with self.lock:
            if not self.emotion_clusters:
                logger.info("No emotion clusters available yet.")
                return {"clusters": {}, "status": "no_clusters", "plugin_name": "vira_emotion_plugin"}
            
            # Convert clusters to a CPU-friendly format for GUI compatibility
            clusters = {}
            for emotion, (centroid, count) in self.emotion_clusters.items():
                centroid_np = self.xp.asnumpy(centroid) if self.xp is cp else centroid.cpu().numpy()
                clusters[emotion] = {"centroid": centroid_np.tolist(), "count": count}
            
            logger.info(f"Retrieved {len(clusters)} emotion clusters.")
            return {
                "clusters": clusters,
                "status": "clusters_retrieved",
                "plugin_name": "vira_emotion_plugin"
            }

    async def run(self, data):
        command = data.get("command", "")
        text = data.get("text", "")
        perspective = data.get("perspective", "Vira")
        feedback = data.get("feedback")
        
        try:
            if command == "analyze":
                emotion_type, emotion_intensity, suggested_perspective, context, simplified_emotion = await self.analyze_emotion(text)
                return {
                    "emotion_type": emotion_type,
                    "emotion_intensity": emotion_intensity,
                    "suggested_perspective": suggested_perspective,
                    "context": context,
                    "simplified_emotion": simplified_emotion,
                    "plugin_name": "vira_emotion_plugin"
                }
            elif command == "summary":
                time_window = data.get("time_window_hours", 24)
                return await self.get_emotional_summary(time_window)
            elif command == "feedback" and feedback:
                user_emotion = feedback.get("emotion")
                user_intensity = feedback.get("intensity")
                if user_emotion:
                    await self.process_feedback(text, user_emotion, user_intensity)
                    return {
                        "emotion_type": user_emotion,
                        "emotion_intensity": user_intensity or 5,
                        "suggested_perspective": perspective,
                        "context": f"Feedback processed at {asyncio.get_event_loop().time()}",
                        "simplified_emotion": simplified_emotion_map.get(user_emotion.capitalize(), "unknown"),
                        "plugin_name": "vira_emotion_plugin"
                    }
            elif command == "clusters":  # New command for GUI compatibility
                return await self.get_clusters()
            
            if not text:
                logger.warning("No text provided to vira_emotion_plugin.")
                return {
                    "emotion_type": "unknown",
                    "emotion_intensity": 0,
                    "suggested_perspective": "Vira",
                    "context": "No text provided",
                    "simplified_emotion": "unknown",
                    "plugin_name": "vira_emotion_plugin"
                }
        
            emotion_type, emotion_intensity, suggested_perspective, context, simplified_emotion = await self.analyze_emotion(text)
            async with self.lock:
                self.context_memory.append({"text": text, "emotion_type": emotion_type, "intensity": emotion_intensity})
            
            result = {
                "emotion_type": emotion_type,
                "emotion_intensity": emotion_intensity,
                "suggested_perspective": suggested_perspective,
                "context": context,
                "context_memory": list(self.context_memory),
                "simplified_emotion": simplified_emotion,
                "plugin_name": "vira_emotion_plugin"
            }
            logger.info(f"Emotion analysis - Text: {text}, Raw: {emotion_type}, Intensity: {emotion_intensity}, Simplified: {simplified_emotion}")
            return result
        except Exception as e:
            logger.error(f"Error running vira_emotion_plugin: {e}\n{traceback.format_exc()}")
            return {
                "emotion_type": "unknown",
                "emotion_intensity": 0,
                "suggested_perspective": "Vira",
                "context": f"Error: {str(e)}",
                "context_memory": [],
                "simplified_emotion": "unknown",
                "plugin_name": "vira_emotion_plugin",
                "error": str(e)
            }

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    config = {
        "model_settings": {"emobert_model": "j-hartmann/emotion-english-distilroberta-base"},
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "emotion_settings": {"learning_rate": 1e-5, "cluster_threshold": 0.7, "max_clusters": 20}
    }
    asyncio.run(vira4_6t.setup_vira())
    plugin = Plugin(config)
    async def test():
        for text in ["I had a great day at work today :)", "I’m so angry about this situation!", "This feels confusing and sad."]:
            result = await plugin.run({"text": text, "command": "analyze"})
            print(f"Input: {text}\nResult: {result}\n")
        result = await plugin.run({"command": "summary"})
        print(f"Summary: {result}")
    asyncio.run(test())