import logging
import platform
import threading
import traceback
from queue import Queue
import pyttsx3
import vira4_6t
import time
import asyncio

logger = logging.getLogger(__name__)

class Plugin:
    def __init__(self, config):
        self.config = config
        self.tts_engine = None
        self.tts_queue = Queue()
        self.is_windows = platform.system() == "Windows"
        self.voice_map = config.get("model_settings", {}).get("tts_settings", {}).get("voices", {
            "Vira": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
            "Core": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0",
            "Echo": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0"
        })
        self.default_voice = None
        self.intensity_threshold = config.get("emotion_settings", {}).get("intensity_threshold", 3)
        self._initialize_tts()
        self.tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
        self.tts_thread.start()
        logger.info("TTS plugin initialized with emotion threshold.")

    def _initialize_tts(self):
        """Initialize the TTS engine with platform-specific handling."""
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            voice_dict = {v.name: v.id for v in voices}
        
            if self.is_windows:
                # Use Windows-specific voices if available
                self.default_voice = self.voice_map.get("Vira", "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0")
                if self.default_voice not in voice_dict.values():
                    logger.warning(f"Default Windows voice {self.default_voice} not found. Selecting first available voice.")
                    self.default_voice = next(iter(voice_dict.values()), None)
                self._set_voice(self.default_voice, emotion_intensity=0)
            else:
                # Dynamically select a voice for non-Windows platforms
                if voices:
                    # Prefer English voices if possible
                    for voice in voices:
                        if "en" in voice.languages or "english" in voice.name.lower():
                            self.default_voice = voice.id
                            break
                    if not self.default_voice:
                        self.default_voice = voices[0].id  # Fallback to first available voice
                    self.voice_map = {
                        "Vira": self.default_voice,
                        "Core": self.default_voice,
                        "Echo": self.default_voice
                    }  # Use same voice for all perspectives on non-Windows
                    self._set_voice(self.default_voice, emotion_intensity=0)
                    logger.info(f"Non-Windows platform: Using voice {self.default_voice} for all perspectives.")
                else:
                    logger.warning("No voices available on this platform.")
                    self.tts_engine = None
                    self.default_voice = None
        
            if not self.default_voice:
                logger.error("No valid voice found; TTS disabled.")
                self.tts_engine = None
        except Exception as e:
            logger.error(f"Error initializing TTS engine: {e}\n{traceback.format_exc()}")
            self.tts_engine = None
            self.default_voice = None

    def _set_voice(self, voice_id, emotion_type="unknown", emotion_intensity=0, cluster_context=None):
        """Set the voice and adjust properties based on emotion and cluster context."""
        try:
            if self.tts_engine:
                self.tts_engine.setProperty('voice', voice_id)
                base_rate = 150
                # Adjust speech rate based on emotion and cluster context
                if emotion_intensity >= self.intensity_threshold:
                    if emotion_type in ["sad", "fear"]:
                        rate = base_rate - 20  # Slower for sadness or fear
                    elif emotion_type in ["angry", "excited"]:
                        rate = base_rate + 20  # Faster for anger or excitement
                    else:
                        rate = base_rate
                    # Apply cluster-based adjustment if available
                    if cluster_context and cluster_context.get("dominant_emotion"):
                        cluster_emotion = cluster_context["dominant_emotion"]
                        cluster_size = cluster_context.get("count", 1)
                        if cluster_emotion in ["sad", "fear"] and emotion_type not in ["sad", "fear"]:
                            rate -= 10 * min(cluster_size / 5, 1)  # Slow further if cluster suggests sadness/fear
                        elif cluster_emotion in ["angry", "excited"] and emotion_type not in ["angry", "excited"]:
                            rate += 10 * min(cluster_size / 5, 1)  # Speed up if cluster suggests anger/excitement
                else:
                    rate = base_rate
                
                self.tts_engine.setProperty('rate', max(100, min(200, rate)))  # Cap between 100-200
                logger.debug(f"Voice set to: {voice_id}, Rate: {rate} (Emotion: {emotion_type}, Intensity: {emotion_intensity}, Cluster: {cluster_context})")
        except Exception as e:
            logger.warning(f"Failed to set voice {voice_id}: {e}")

    def _tts_loop(self):
        """Process TTS requests from the queue."""
        while True:
            try:
                text, perspective, emotion_data = self.tts_queue.get()
                if self.tts_engine:
                    self._speak(text, perspective, emotion_data)
                self.tts_queue.task_done()
            except Exception as e:
                logger.error(f"Error in TTS loop: {e}\n{traceback.format_exc()}")

    def _speak(self, text, perspective, emotion_data=None):
        """Speak the text with the appropriate voice, emotional adjustment, and cluster context."""
        if not text:
            logger.warning("No text provided to speak.")
            return
        
        voice_id = self.voice_map.get(perspective, self.default_voice)
        emotion_type = emotion_data.get("emotion_type", "unknown") if emotion_data else "unknown"
        emotion_intensity = emotion_data.get("emotion_intensity", 0) if emotion_data else 0
        
        if not self.tts_engine or not voice_id:
            logger.warning(f"TTS not available for perspective {perspective} (engine or voice not initialized).")
            return
        
        # Fetch cluster context from hippo_plugin
        cluster_context = None
        if vira4_6t.plugin_manager and "hippo_plugin" in vira4_6t.plugin_manager.plugins:
            try:
                memory_result = vira4_6t.plugin_manager.execute_specific_plugin(
                    "hippo_plugin", {"command": "retrieve", "input_text": text}
                )
                if asyncio.iscoroutine(memory_result):
                    memory_result = asyncio.get_event_loop().run_until_complete(memory_result)
                episodic_memories = memory_result.get("episodic", [])
                if episodic_memories:
                    cluster_groups = {}
                    for memory in episodic_memories:
                        cluster = memory.get("cluster")
                        if cluster is not None and cluster != -1:
                            if cluster not in cluster_groups:
                                cluster_groups[cluster] = []
                            cluster_groups[cluster].append(memory)
                    
                    if cluster_groups:
                        strongest_cluster = max(
                            cluster_groups.items(),
                            key=lambda x: sum(m["emotion_intensity"] for m in x[1]) / len(x[1]) if x[1] else 0,
                            default=(None, [])
                        )[0]
                        if strongest_cluster:
                            cluster_memories = cluster_groups[strongest_cluster]
                            emotion_counts = {}
                            total_intensity = 0
                            for memory in cluster_memories:
                                emotion = memory["emotion_type"]
                                intensity = memory["emotion_intensity"]
                                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                                total_intensity += intensity
                            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
                            cluster_context = {
                                "dominant_emotion": dominant_emotion,
                                "count": len(cluster_memories)
                            }
                            logger.debug(f"Cluster context for '{text}': {cluster_context}")
            except Exception as e:
                logger.error(f"Error fetching cluster context for TTS: {e}\n{traceback.format_exc()}")
        
        try:
            self._set_voice(voice_id, emotion_type, emotion_intensity, cluster_context)
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            logger.info(f"Spoke: '{text}' as {perspective} (Emotion: {emotion_type}, Intensity: {emotion_intensity}, Cluster: {cluster_context})")
        except Exception as e:
            logger.error(f"Error speaking '{text}' as {perspective}: {e}\n{traceback.format_exc()}")

    def run(self, data):
        """Queue text to be spoken with optional emotional context."""
        text = data.get("text", "")
        perspective = data.get("perspective", "Vira")
        emotion_data = data.get("emotion_data")  # From vira_emotion_plugin
        
        if not text:
            logger.warning("No text provided for TTS.")
            return {"status": "no_text", "plugin_name": "tts_plugin"}
        
        if not self.tts_engine:
            logger.warning("TTS engine not initialized or unavailable.")
            return {"status": "tts_unavailable", "plugin_name": "tts_plugin"}
        
        try:
            self.tts_queue.put((text, perspective, emotion_data))
            return {
                "status": "queued",
                "text": text,
                "perspective": perspective,
                "emotion_type": emotion_data.get("emotion_type", "unknown") if emotion_data else "unknown",
                "emotion_intensity": emotion_data.get("emotion_intensity", 0) if emotion_data else 0,
                "plugin_name": "tts_plugin"
            }
        except Exception as e:
            logger.error(f"Error queuing TTS: {e}\n{traceback.format_exc()}")
            return {"status": "error", "error": str(e), "plugin_name": "tts_plugin"}

    def shutdown(self):
        """Cleanly shut down the TTS thread."""
        if self.tts_engine:
            self.tts_engine.stop()
        # Allow thread to exit naturally as daemon

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = {
        "model_settings": {
            "tts_settings": {
                "voices": {
                    "Vira": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
                    "Core": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0",
                    "Echo": "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0"
                }
            }
        },
        "emotion_settings": {
            "intensity_threshold": 3
        }
    }
    
    asyncio.run(vira4_6t.setup_vira())
    plugin = Plugin(config)
    
    test_inputs = [
        {"text": "Hello, Livia! I’m Vira.", "perspective": "Vira", "emotion_data": {"emotion_type": "happy", "emotion_intensity": 6}},
        {"text": "Let’s get to work.", "perspective": "Core", "emotion_data": {"emotion_type": "excited", "emotion_intensity": 8}},
        {"text": "I’m feeling a bit sad today.", "perspective": "Echo", "emotion_data": {"emotion_type": "sad", "emotion_intensity": 7}},
        {"text": "", "perspective": "Vira"}
    ]
    
    for test in test_inputs:
        result = plugin.run(test)
        print(f"Input: {test}\nResult: {result}\n")
        time.sleep(2)  # Allow time for audio to play
    
    plugin.shutdown()