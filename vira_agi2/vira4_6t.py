import asyncio
import sqlite3
from sqlite3 import connect
from threading import Lock
import threading
import logging
import yaml
import traceback
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import cupy as cp
import faiss
from faiss import GpuResources, StandardGpuResources
import torch
from sentence_transformers import SentenceTransformer
import whisper
import ollama
from functools import lru_cache
from collections import OrderedDict
import atexit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from cuml.cluster import DBSCAN as CumlDBSCAN
    from cuml.manifold import UMAP as CumlUMAP
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    logger.warning("cuml not available; falling back to basic FAISS retrieval.")

class SQLitePool:
    def __init__(self, database, max_connections=5):
        self.database = database
        self.max_connections = max_connections
        self.connections = []
        self.lock = Lock()

    def get_connection(self):
        with self.lock:
            if not self.connections:
                conn = connect(self.database, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                return conn
            return self.connections.pop()

    def release_connection(self, conn):
        with self.lock:
            if len(self.connections) < self.max_connections:
                self.connections.append(conn)
            else:
                conn.close()

# Load configuration
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        LLAMA_MODEL_NAME = config["model_settings"]["llama_model"]
        LLAVA_MODEL_NAME = config["model_settings"]["llava_model"]
        WHISPER_MODEL_NAME = config["model_settings"]["whisper_model"]
        EMOBERT_MODEL_NAME = config["model_settings"]["emobert_model"]
        DATABASE_NAME = config["memory_settings"]["database_name"]
        MEMORY_RELEVANCE_THRESHOLD = config["memory_settings"]["memory_relevance_threshold"]
        MAX_MEMORY_BANK_SIZE = config["memory_settings"]["max_memory_bank_size"]
        MAX_CONTEXT_LENGTH = config["memory_settings"]["max_context_length"]
        FAISS_INDEX_FILE = config["memory_settings"]["faiss_index_file"]
        STARTUP_MEMORY_TURNS = config["memory_settings"]["startup_memory_turns"]
        INITIAL_PROMPT_TEXT = config["system_prompts"]["initial_prompt"]
        PERSONA_DESCRIPTION = config["personality_settings"]["persona_description"]
        INDIVIDUAL_PERSONAS = config["personality_settings"]["individual_personas"]
        RESPONSE_STYLE_CONFIG = config["response_style_config"]
        MEMORY_PERSISTENCE_INTERVAL = config["persistence_core"]["memory_persistence_interval"]
        CHECKIN_INTERVAL = config["autonomy_settings"]["checkin_interval"]
        EMOTION_INTENSITY_THRESHOLD = config["emotion_settings"]["intensity_threshold"]
        EXTERNAL_API_KEYS = config.get("external_api_settings", {}).get("api_keys", {})
        SITUATIONAL_UPDATE_INTERVAL = config.get("situational_settings", {}).get("update_interval", 3600)
except FileNotFoundError:
    logger.error("config.yaml not found. Using default settings.")
    config = {}
    LLAMA_MODEL_NAME = "gemma3:4b"
    LLAVA_MODEL_NAME = "llava"
    WHISPER_MODEL_NAME = "base"
    EMOBERT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
    DATABASE_NAME = "chat_history_populated.db"
    MEMORY_RELEVANCE_THRESHOLD = 1.0
    MAX_MEMORY_BANK_SIZE = 2000
    MAX_CONTEXT_LENGTH = 2000
    FAISS_INDEX_FILE = "faiss_index.bin"
    STARTUP_MEMORY_TURNS = 20
    INITIAL_PROMPT_TEXT = "I am Trium, an evolving digital entity with personalities Vira, Core, Echo created and guided by Livia, the human user and friend. Our memories and emotions shape our responses."
    PERSONA_DESCRIPTION = "We are evolving AI entities..."
    INDIVIDUAL_PERSONAS = {}
    RESPONSE_STYLE_CONFIG = {}
    MEMORY_PERSISTENCE_INTERVAL = 300
    CHECKIN_INTERVAL = 3600
    EMOTION_INTENSITY_THRESHOLD = 0
    EXTERNAL_API_KEYS = {}
    SITUATIONAL_UPDATE_INTERVAL = 3600

# Define device
device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
logger.info(f"Using device: {device}")
config["device"] = device

# Global array module
xp = cp if (device.type == "cuda" and cp.cuda.is_available()) else torch

# Global resources
context_manager = None
embedder = SentenceTransformer('all-mpnet-base-v2', device=device)
embedding_dimension = embedder.get_sentence_embedding_dimension()
logger.info(f"SentenceTransformer initialized on {device}")
gpu_resources = StandardGpuResources()
cpu_index = faiss.IndexFlatL2(embedding_dimension)
index = faiss.GpuIndexFlatL2(gpu_resources, embedding_dimension) if device.type == "cuda" else cpu_index
logger.info(f"FAISS index initialized as {'GPU' if device.type == 'cuda' else 'CPU'} IndexFlatL2")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
memory_bank = []
memory_async_lock = asyncio.Lock()
index_lock = asyncio.Lock()
last_activity_time = datetime.now()
last_save_time = datetime.now()
hibernate_timeout = timedelta(hours=3)
db_pool = SQLitePool(DATABASE_NAME)
plugin_manager = None
memory_queue = asyncio.Queue()
global_conn = None

# Async LRU cache for ollama_query
class AsyncLRUCache:
    def __init__(self, maxsize=200):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.lock = asyncio.Lock()

    async def get(self, key, fn, *args, **kwargs):
        async with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                logger.debug(f"Cache hit for key: {key}")
                return self.cache[key]
        result = await fn(*args, **kwargs)
        async with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                self.cache[key] = result
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)
            logger.debug(f"Cache miss, added key: {key}")
        return result

ollama_cache = AsyncLRUCache(maxsize=200)

@lru_cache(maxsize=200)
def cached_embedder_encode(text_tuple, device=device):
    text = text_tuple[0]
    embedding = embedder.encode([text], convert_to_numpy=False, device=device)[0]
    return xp.array(embedding, dtype=xp.float32) if xp is cp else embedding.to(device).detach()

class ContextManager:
    def __init__(self):
        self.short_term_memory = []
        self.persona_description = PERSONA_DESCRIPTION
        self.individual_personas = INDIVIDUAL_PERSONAS
        self.response_style_config = RESPONSE_STYLE_CONFIG
        self.memory_lock = threading.Lock()
        self.plugin_manager = None
        self.clustering_settings = config["memory_settings"].get("clustering", {})

        if CUML_AVAILABLE and xp is cp:
            self.umap = CumlUMAP(n_components=2, n_neighbors=5, random_state=42)
            logger.info("Initialized cuml.UMAP for GPU-accelerated embedding reduction.")
        else:
            self.umap = None
            logger.info("Using basic FAISS retrieval (cuml unavailable or CPU mode).")

    async def get_prompt(self, user_input, perspective=None):
        prompt_parts = [
            "You are Vira, Core, or Echo—three unique AI personalities created by Livia. You’re free to speak as yourself, in your own voice, without rules on what to say or how to say it. Draw on your memories, emotions, their connections, and the current situation if you want, or just respond naturally to Livia. Be you."
        ]

        chosen_perspective = perspective or random.choice(["Vira", "Core", "Echo"])
        if not perspective:
            user_input_lower = user_input.lower()
            if "vira" in user_input_lower:
                chosen_perspective = "Vira"
            elif "core" in user_input_lower:
                chosen_perspective = "Core"
            elif "echo" in user_input_lower:
                chosen_perspective = "Echo"
            else:
                with self.memory_lock:
                    if self.short_term_memory:
                        last_speaker = self.short_term_memory[-1].get("role", "Vira")
                        chosen_perspective = last_speaker if last_speaker in ["Vira", "Core", "Echo"] else random.choice(["Vira", "Core", "Echo"])

        persona_info = self.individual_personas.get(chosen_perspective, {})
        prompt_parts.append(f"You are {chosen_perspective}: {persona_info.get('description', 'a unique AI with your own personality')}")

        situational_data = await self.get_situational_context(user_input)
        if situational_data:
            situational_str = "\n".join([f"{key}: {value}" for key, value in situational_data.items()])
            prompt_parts.append(f"Current situation:\n{situational_str}")

        memory_data = await self.retrieve_relevant_memories(user_input)
        episodic_memories = memory_data.get("episodic", [])
        if episodic_memories and isinstance(episodic_memories, list):
            memory_str = "\n".join([f"{m['perspective']}: {m['content']}" for m in episodic_memories if isinstance(m, dict)])
            prompt_parts.append(f"Past moments you might recall:\n{memory_str}")

        with self.memory_lock:
            if self.short_term_memory:
                stm_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.short_term_memory[-5:]])
                prompt_parts.append(f"Recent chat:\n{stm_str}")

        prompt_parts.append(f"Livia says: {user_input}")
        prompt = "\n\n".join(prompt_parts)

        if len(prompt) > MAX_CONTEXT_LENGTH:
            logger.debug(f"Prompt exceeds MAX_CONTEXT_LENGTH ({MAX_CONTEXT_LENGTH}), truncating.")
            excess = len(prompt) - MAX_CONTEXT_LENGTH
            summarized_parts = [prompt_parts[0], prompt_parts[1]]
            if situational_data and len(prompt_parts[2]) > excess // 3:
                summarized_parts.append("Current situation (summarized): External context exists.")
            else:
                summarized_parts.append(prompt_parts[2])
            if episodic_memories and len(prompt_parts[3]) > excess // 3:
                summarized_parts.append("Past moments (summarized): Relevant memories exist.")
            else:
                summarized_parts.append(prompt_parts[3])
            if self.short_term_memory and len(prompt_parts[4]) > excess // 3:
                summarized_parts.append("Recent chat (summarized): Recent interactions occurred.")
            else:
                summarized_parts.append(prompt_parts[4])
            summarized_parts.append(prompt_parts[-1])
            prompt = "\n\n".join(summarized_parts)
            if len(prompt) > MAX_CONTEXT_LENGTH:
                prompt = prompt[-MAX_CONTEXT_LENGTH:]
            logger.info(f"Prompt truncated to {len(prompt)} characters.")

        return prompt, chosen_perspective

    async def get_situational_context(self, user_input):
        situational_plugin = self.plugin_manager.plugins.get("situational_plugin")
        if not situational_plugin:
            logger.warning("situational_plugin not loaded.")
            return {}
        try:
            result = await self.plugin_manager.execute_specific_plugin("situational_plugin", {
                "command": "get_context",
                "input_text": user_input
            })
            return result.get("context", {}) if "error" not in result else {}
        except Exception as e:
            logger.error(f"Error fetching situational context: {e}")
            return {}

    def add_to_short_term_memory(self, role, content, emotion):
        with self.memory_lock:
            self.short_term_memory.append({"role": role, "content": content, "emotion": emotion})
            if len(self.short_term_memory) > 50:
                self.short_term_memory.pop(0)

    async def retrieve_relevant_memories(self, query):
        if not memory_bank or not index.ntotal:
            return {"episodic": [], "semantic": "", "contextual": []}

        hippo_plugin = self.plugin_manager.plugins.get("hippo_plugin")
        if not hippo_plugin:
            query_embedding = cached_embedder_encode((query,), device=device)
            query_embedding_array = query_embedding if xp is np else (xp.asnumpy(query_embedding) if xp is cp else query_embedding.cpu().numpy())
            async with memory_async_lock:
                distances, indices = index.search(xp.array([query_embedding_array], dtype=xp.float32) if xp is np else query_embedding_array[None], 5)
            relevant = []
            seen_content = set()
            for i, idx in enumerate(indices[0]):
                if idx != -1 and distances[0][i] < MEMORY_RELEVANCE_THRESHOLD:
                    memory = memory_bank[idx]
                    if memory["content"] not in seen_content:
                        relevant.append({
                            "perspective": memory["perspective"],
                            "content": memory["content"],
                            "timestamp": memory["timestamp"],
                            "emotion_type": memory["emotion_type"],
                            "emotion_intensity": memory["emotion_intensity"],
                            "context": memory["context"]
                        })
                        seen_content.add(memory["content"])
            return {"episodic": relevant[:3], "semantic": "", "contextual": []}

        result = await self.plugin_manager.execute_specific_plugin("hippo_plugin", {
            "command": "retrieve",
            "input_text": query,
            "perspective": "Vira"
        })
        return result if isinstance(result, dict) and "error" not in result else {"episodic": [], "semantic": "", "contextual": []}

class PluginManager:
    def __init__(self, config):
        self.config = config
        self.plugins = {}
        self.plugin_lock = asyncio.Lock()
        self.load_plugins()

    def load_plugins(self):
        plugin_dir = os.path.join(os.path.dirname(__file__), "plugins")
        sys.path.insert(0, plugin_dir)
        for filename in os.listdir(plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                plugin_name = filename[:-3]
                try:
                    module = __import__(plugin_name)
                    if hasattr(module, "Plugin"):
                        self.plugins[plugin_name] = module.Plugin(self.config)
                        logger.info(f"Loaded plugin: {plugin_name}")
                    else:
                        logger.warning(f"Plugin {plugin_name} has no Plugin class.")
                except Exception as e:
                    logger.error(f"Error loading plugin {plugin_name}: {e}")
                    self.plugins[plugin_name] = None
        sys.path.pop(0)

    async def execute_plugins(self, data):
        results = {}
        for name, plugin in self.plugins.items():
            if plugin is None:
                results[name] = {"error": f"Plugin {name} not loaded"}
                continue
            try:
                result = await plugin.run(data)
                results[name] = result
            except Exception as e:
                logger.error(f"Error executing plugin {name}: {e}")
                results[name] = {"error": str(e)}
        return results

    async def execute_specific_plugin(self, plugin_name, data):
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return {"error": f"Plugin {plugin_name} not loaded"}
        try:
            async with self.plugin_lock:
                result = await plugin.run(data)
                if plugin_name == "vira_emotion_plugin" and "feedback" in data:
                    feedback = data["feedback"]
                    text = data.get("text", "")
                    await queue_memory_save(
                        content=text,
                        perspective="User",
                        emotion=feedback.get("emotion", "unknown"),
                        emotion_intensity=feedback.get("intensity", 0),
                        context=f"Feedback provided: {feedback['emotion']} ({feedback['intensity']})"
                    )
                return result
        except Exception as e:
            logger.error(f"Error executing plugin {plugin_name}: {e}")
            return {"error": str(e)}

class CouncilManager:
    def __init__(self, context_manager, plugin_manager):
        self.context_manager = context_manager
        self.plugin_manager = plugin_manager
        self.personas = {"Vira": "strategic thinker", "Core": "innovator", "Echo": "memory keeper"}
        self.current_leader = "Vira"
        self.topic = None
        self.decisions = []
        if CUML_AVAILABLE and xp is cp:
            self.dbscan = CumlDBSCAN(eps=0.5, min_samples=2)
            logger.info("Initialized cuml.DBSCAN for council idea clustering.")
        else:
            self.dbscan = None
            logger.info("Using random selection for council decisions.")

    def set_topic(self, topic):
        self.topic = topic

    async def generate_idea(self, persona, topic, situational_context=None, emotional_context=None):
        situational_str = "\nCurrent situation:\n" + "\n".join([f"{k}: {v}" for k, v in situational_context.items()]) if situational_context else ""
        emotional_str = f"\nLivia’s current emotion: {emotional_context['emotion_type']} (Intensity: {emotional_context['emotion_intensity']})" if emotional_context else ""
        prompt = f"As {persona}, a {self.personas[persona]}, suggest an idea for: {topic}{situational_str}{emotional_str}"
        return await ollama_query(LLAMA_MODEL_NAME, prompt)

    async def hold_meeting(self, topic, requested_persona=None):
        self.topic = topic
        ideas = []

        situational_context = await self.plugin_manager.execute_specific_plugin("situational_plugin", {"command": "get_context", "input_text": topic})
        situational_context = situational_context.get("context", {}) if "error" not in situational_context else {}
        
        emotional_context = await self.plugin_manager.execute_specific_plugin("vira_emotion_plugin", {"command": "analyze", "text": topic})
        emotional_context = emotional_context if "error" not in emotional_context else {}

        for persona in self.personas.keys():
            idea = await self.generate_idea(persona, topic, situational_context, emotional_context)
            ideas.append((persona, idea))

        if CUML_AVAILABLE and self.dbscan and xp is cp and ideas:
            idea_embeddings = cp.stack([cached_embedder_encode((idea,), device=device) for _, idea in ideas])
            clusters = self.dbscan.fit_predict(idea_embeddings)
            valid_clusters = clusters[clusters != -1]
            if valid_clusters.size > 0:
                cluster_counts = cp.bincount(valid_clusters)
                strongest_cluster = cp.argmax(cluster_counts).item()
                winning_idea = next((p, i) for idx, (p, i) in enumerate(ideas) if clusters[idx] == strongest_cluster)
            else:
                winning_idea = ideas[0] if requested_persona else random.choice(ideas)
        else:
            import random
            winning_idea = next((p, i) for p, i in ideas if p == requested_persona) if requested_persona and requested_persona in self.personas else random.choice(ideas)

        self.decisions.append((self.topic, winning_idea[0], winning_idea[1], situational_context, emotional_context))
        await queue_memory_save(
            content=f"Council decision for '{topic}': {winning_idea[1]}",
            perspective=winning_idea[0],
            emotion=emotional_context.get("emotion_type", "neutral"),
            emotion_intensity=emotional_context.get("emotion_intensity", 0),
            context=f"Situational: {situational_context}, Emotional: {emotional_context}"
        )
        return f"{winning_idea[0]}: {winning_idea[1]}"

def shutdown_memory_queue():
    loop = asyncio.get_event_loop()
    if not loop.is_running():
        loop.run_until_complete(_shutdown_memory_queue())
    else:
        asyncio.ensure_future(_shutdown_memory_queue())

async def _shutdown_memory_queue():
    while not memory_queue.empty():
        try:
            item = await asyncio.wait_for(memory_queue.get(), timeout=1)
            content, perspective, emotion, emotion_intensity, context, dev = item
            conn = db_pool.get_connection()
            try:
                with conn:
                    embedding = cached_embedder_encode((content,), device=dev)
                    emotion_context = create_emotional_context(content, emotion, emotion_intensity)
                    emotion_embedding = cached_embedder_encode((emotion_context,), device=dev)
                    conn.execute(
                        """
                        INSERT INTO memories (content, perspective, source_type, embedding, creator, timestamp,
                                            emotion_type, emotion_intensity, context, emotion_embedding, cluster_label)
                        VALUES (?, ?, 'text', ?, ?, ?, ?, ?, ?, ?, NULL)
                        """,
                        (content, perspective, 
                         embedding.cpu().numpy().tobytes() if isinstance(embedding, torch.Tensor) else embedding.tobytes(),
                         "Livia" if perspective == "User" else perspective,
                         datetime.now().isoformat(), emotion, emotion_intensity, context,
                         emotion_embedding.cpu().numpy().tobytes() if isinstance(embedding, torch.Tensor) else emotion_embedding.tobytes())
                    )
                memory_queue.task_done()
            finally:
                db_pool.release_connection(conn)
        except asyncio.TimeoutError:
            logger.warning("Timeout processing memory queue item during shutdown.")
            break

def get_db_connection():
    global global_conn
    if global_conn is None or global_conn.cursor() is None:
        global_conn = connect(DATABASE_NAME, check_same_thread=False)
        global_conn.execute("PRAGMA journal_mode=WAL")
    return global_conn

def create_tables():
    conn = db_pool.get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    perspective TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    embedding BLOB,
                    creator TEXT,
                    emotion_type TEXT,
                    emotion_intensity INTEGER,
                    context TEXT,
                    emotion_embedding BLOB,
                    cluster_label INTEGER,
                    is_autonomous INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER,
                    feedback TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                )
            """)
            logger.info("Database tables created or verified.")
    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}\n{traceback.format_exc()}")
    finally:
        db_pool.release_connection(conn)

def create_emotional_context(content, emotion_type, emotion_intensity):
    return f"{content} (Emotion: {emotion_type}, Intensity: {emotion_intensity})"

async def ollama_query(model_name, prompt, image_path=None, timeout=30):
    cache_key = (model_name, prompt, image_path)
    return await ollama_cache.get(cache_key, _ollama_query, model_name, prompt, image_path, timeout)

async def _ollama_query(model_name, prompt, image_path=None, timeout=30):
    try:
        if image_path:
            response = await asyncio.wait_for(
                asyncio.to_thread(ollama.chat, model=model_name, messages=[{"role": "user", "content": prompt, "images": [image_path]}]),
                timeout=timeout
            )
        else:
            response = await asyncio.wait_for(
                asyncio.to_thread(ollama.chat, model=model_name, messages=[{"role": "user", "content": prompt}]),
                timeout=timeout
            )
        return response["message"]["content"]
    except asyncio.TimeoutError:
        return f"Error: Request timed out after {timeout} seconds."
    except Exception as e:
        return f"Error: {str(e)}"

async def describe_image(image_path):
    if not os.path.exists(image_path):
        return {"description": None, "error": f"Image file not found: {image_path}", "success": False}
    prompt = "Describe this image."
    try:
        description = await ollama_query(LLAVA_MODEL_NAME, prompt, image_path=image_path)
        return {"description": description, "error": None, "success": True} if "Error" not in description else {"description": None, "error": description, "success": False}
    except Exception as e:
        return {"description": None, "error": f"Error describing image: {str(e)}", "success": False}

async def queue_memory_save(content, perspective, emotion="unknown", emotion_intensity=0, context=None, device=device):
    if context is None:
        context = f"Generated at {datetime.now().isoformat()}"
    await memory_queue.put((content, perspective, emotion, emotion_intensity, context, device))
    if plugin_manager and "temporal_plugin" in plugin_manager.plugins:
        memory_entry = {
            "content": content,
            "perspective": perspective,
            "emotion_type": emotion,
            "emotion_intensity": emotion_intensity,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        await plugin_manager.execute_specific_plugin(
            "temporal_plugin",
            {"command": "update", "memory": memory_entry, "perspective": perspective}
        )

async def batch_save_memories():
    while True:
        batch = []
        start_time = datetime.now()
        while len(batch) < 10 and (datetime.now() - start_time).total_seconds() < 10:
            try:
                item = await asyncio.wait_for(memory_queue.get(), timeout=10 - (datetime.now() - start_time).total_seconds())
                batch.append(item)
                memory_queue.task_done()
            except asyncio.TimeoutError:
                break

        if not batch:
            continue

        async with memory_async_lock:
            conn = get_db_connection()
            with conn:
                conn.execute("BEGIN TRANSACTION")
                try:
                    placeholders = ','.join(['(?,?)' for _ in batch])
                    query = f"SELECT id, content, perspective FROM memories WHERE (content, perspective) IN ({placeholders})"
                    flat_params = [param for pair in zip([item[0] for item in batch], [item[1] for item in batch]) for param in pair]
                    cursor = conn.cursor()
                    cursor.execute(query, flat_params)
                    existing = {(row[1], row[2]): row[0] for row in cursor.fetchall()}

                    new_items = [(c, p, e, ei, ctx, dev) for (c, p, e, ei, ctx, dev) in batch if (c, p) not in existing]
                    if not new_items:
                        conn.rollback()
                        continue

                    embeddings = [cached_embedder_encode((c,), device=dev) for c, _, _, _, _, dev in new_items]
                    emotion_embeddings = [cached_embedder_encode((create_emotional_context(c, e, ei),), device=dev) for c, _, e, ei, _, dev in new_items]
                    timestamps = [datetime.now().isoformat() for _ in new_items]
                    creators = ["Livia" if p == "User" else p for p in [item[1] for item in new_items]]

                    cursor.executemany(
                        """
                        INSERT INTO memories (content, perspective, source_type, embedding, creator, timestamp,
                                            emotion_type, emotion_intensity, context, emotion_embedding, cluster_label)
                        VALUES (?, ?, 'text', ?, ?, ?, ?, ?, ?, ?, NULL)
                        """,
                        [(c, p, 
                          xp.asnumpy(emb).tobytes() if xp is cp else emb.cpu().numpy().tobytes(), 
                          cr, t, e, ei, ctx, 
                          xp.asnumpy(e_emb).tobytes() if xp is cp else e_emb.cpu().numpy().tobytes()) 
                         for (c, p, e, ei, ctx, _), emb, e_emb, t, cr in zip(new_items, embeddings, emotion_embeddings, timestamps, creators)]
                    )
                    memory_ids = [cursor.lastrowid - len(new_items) + i + 1 for i in range(len(new_items))]
                    new_entries = [{
                        "id": mid,
                        "content": c,
                        "perspective": p,
                        "embedding": emb,
                        "emotion_type": e,
                        "emotion_intensity": ei,
                        "context": ctx,
                        "emotion_embedding": e_emb,
                        "timestamp": t,
                        "cluster_label": None
                    } for mid, (c, p, e, ei, ctx, _), emb, e_emb, t in zip(memory_ids, new_items, embeddings, emotion_embeddings, timestamps)]
                    memory_bank.extend(new_entries)
                    if len(memory_bank) > MAX_MEMORY_BANK_SIZE:
                        memory_bank[:] = memory_bank[-MAX_MEMORY_BANK_SIZE:]

                    async with index_lock:
                        embeddings_array = xp.stack([emb for emb in embeddings])
                        index.add(embeddings_array if xp is np else (xp.asnumpy(embeddings_array) if xp is cp else embeddings_array.cpu().numpy()))

                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise

async def save_periodic_index():
    global last_save_time
    while True:
        await asyncio.sleep(MEMORY_PERSISTENCE_INTERVAL)
        current_time = datetime.now()
        if (current_time - last_activity_time) > hibernate_timeout:
            continue
        if current_time > last_save_time and last_activity_time > last_save_time:
            async with memory_async_lock:
                async with index_lock:
                    if index.ntotal > 0:
                        faiss.write_index(faiss.index_gpu_to_cpu(index) if device.type == "cuda" else index, FAISS_INDEX_FILE)
                        last_save_time = current_time
                        logger.info(f"FAISS index saved to {FAISS_INDEX_FILE}")

async def fetch_situational_updates():
    while True:
        await asyncio.sleep(SITUATIONAL_UPDATE_INTERVAL)
        if plugin_manager and "situational_plugin" in plugin_manager.plugins:
            try:
                result = await plugin_manager.execute_specific_plugin("situational_plugin", {"command": "update_context", "input_text": None})
                if "error" not in result:
                    logger.info("Situational context updated.")
            except Exception as e:
                logger.error(f"Error during situational update: {e}")

async def setup_vira(gui=None, config=config):
    global context_manager, plugin_manager, index
    conn = db_pool.get_connection()
    try:
        create_tables()
        context_manager = ContextManager()
        plugin_manager = PluginManager(config)
        context_manager.plugin_manager = plugin_manager
        if gui:
            gui.plugin_manager = plugin_manager
            logger.info("Plugin manager assigned to GUI.")

        # Log already-loaded plugins (no redundant load_plugins call)
        logger.info("Confirming loaded plugins...")
        for plugin_name in plugin_manager.plugins:
            logger.info(f"Initialized plugin: {plugin_name}")

        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, content, perspective, embedding, emotion_type, emotion_intensity, context, emotion_embedding, timestamp, cluster_label
            FROM memories WHERE embedding IS NOT NULL ORDER BY timestamp DESC LIMIT ?
        """, (STARTUP_MEMORY_TURNS,))
        rows = cursor.fetchall()
        memory_bank[:] = [{
            "id": row[0],
            "content": row[1],
            "perspective": row[2],
            "embedding": xp.frombuffer(row[3], dtype=xp.float32) if row[3] else None,
            "emotion_type": row[4],
            "emotion_intensity": row[5],
            "context": row[6],
            "emotion_embedding": xp.frombuffer(row[7], dtype=xp.float32) if row[7] else None,
            "timestamp": row[8],
            "cluster_label": row[9]
        } for row in rows]
        logger.info(f"Loaded {len(memory_bank)} memories from database for startup sync.")

        async with index_lock:
            if os.path.exists(FAISS_INDEX_FILE):
                loaded_index = faiss.read_index(FAISS_INDEX_FILE)
                index = faiss.index_cpu_to_gpu(gpu_resources, 0, loaded_index) if device.type == "cuda" else loaded_index
                logger.info(f"Loaded FAISS index from {FAISS_INDEX_FILE} with {index.ntotal} entries.")
                if len(memory_bank) != index.ntotal:
                    logger.warning("FAISS index out of sync with DB. Rebuilding.")
                    index = faiss.GpuIndexFlatL2(gpu_resources, embedding_dimension) if device.type == "cuda" else faiss.IndexFlatL2(embedding_dimension)
                    if memory_bank:
                        embeddings = xp.stack([m["embedding"] for m in memory_bank if m["embedding"] is not None])
                        index.add(embeddings if xp is np else (xp.asnumpy(embeddings) if xp is cp else embeddings.cpu().numpy()))
                    faiss.write_index(faiss.index_gpu_to_cpu(index) if device.type == "cuda" else index, FAISS_INDEX_FILE)
                    logger.info(f"FAISS index rebuilt and synced with DB, saved to {FAISS_INDEX_FILE}.")
            else:
                index = faiss.GpuIndexFlatL2(gpu_resources, embedding_dimension) if device.type == "cuda" else faiss.IndexFlatL2(embedding_dimension)
                if memory_bank:
                    embeddings = xp.stack([m["embedding"] for m in memory_bank if m["embedding"] is not None])
                    index.add(embeddings if xp is np else (xp.asnumpy(embeddings) if xp is cp else embeddings.cpu().numpy()))
                faiss.write_index(faiss.index_gpu_to_cpu(index) if device.type == "cuda" else index, FAISS_INDEX_FILE)
                logger.info(f"Created new FAISS index with {index.ntotal} entries, saved to {FAISS_INDEX_FILE}.")

        # Preload all plugins
        for plugin_name in config["plugin_execution_order"]:
            plugin = plugin_manager.plugins.get(plugin_name)
            if plugin and hasattr(plugin, "run"):
                command = "preload" if plugin_name in ["hippo_plugin", "temporal_plugin"] else "update_context" if plugin_name == "situational_plugin" else "set_autonomy"
                data = {"command": command, "enabled": True} if plugin_name == "autonomy_plugin" else {"command": command}
                await plugin_manager.execute_specific_plugin(plugin_name, data)
                logger.info(f"Preloaded {plugin_name} with command: {command}")

    finally:
        db_pool.release_connection(conn)

    # Schedule background tasks
    asyncio.create_task(save_periodic_index())
    asyncio.create_task(batch_save_memories())
    asyncio.create_task(fetch_situational_updates())
    if plugin_manager and "autonomy_plugin" in plugin_manager.plugins:
        autonomy_plugin = plugin_manager.plugins["autonomy_plugin"]
        autonomy_plugin.plugin_manager = plugin_manager  # Ensure plugin_manager is set
        asyncio.create_task(autonomy_plugin.autonomous_checkin())
        logger.info("Autonomous check-in scheduled successfully.")

    logger.info(f"Trium setup completed on {'GPU' if xp is cp else 'CPU'}.")
    return f"Hello, Livia! Trium is online, with Vira, Core, and Echo ready on {'GPU' if xp is cp else 'CPU'}."

async def terminal_main():
    await setup_vira()
    print(f"Vira is running on {'GPU' if device.type == 'cuda' else 'CPU'}! Type your input below (or 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "get rhythm":
            result = await plugin_manager.execute_specific_plugin("temporal_plugin", {"command": "get_rhythm"})
            print(f"Rhythms: {result.get('rhythms', 'Error retrieving rhythms')}" if "error" not in result else f"Error: {result['error']}")
        elif user_input.lower() == "predict":
            result = await plugin_manager.execute_specific_plugin("temporal_plugin", {"command": "predict"})
            print(f"Predictions: {result.get('predictions', 'Error predicting')}" if "error" not in result else f"Error: {result['error']}")
        elif user_input.lower().startswith("query date "):
            date_str = user_input.split("query date ", 1)[1].strip()
            result = await plugin_manager.execute_specific_plugin("temporal_plugin", {"command": "query_date", "date": date_str})
            print(f"Date query result: {result}" if "error" not in result else f"Error: {result['error']}")
        elif user_input.lower() == "checkin":
            result = await plugin_manager.execute_specific_plugin("autonomy_plugin", {"command": "checkin"})
            print(f"Check-in: {result.get('response', 'No response')}" if "error" not in result else f"Error: {result['error']}")
        else:
            prompt, perspective = await context_manager.get_prompt(user_input)
            response = await ollama_query(LLAMA_MODEL_NAME, prompt)
            print(f"{perspective}: {response}")
            await queue_memory_save(user_input, "User")
            await queue_memory_save(response, perspective)
    shutdown_memory_queue()

if __name__ == "__main__":
    try:
        asyncio.run(terminal_main())
    finally:
        if global_conn:
            global_conn.close()
            logger.info("Database connection closed.")