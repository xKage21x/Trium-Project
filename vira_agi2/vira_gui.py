import torch
import sys
import os
import threading
import asyncio
import queue
import tkinter as tk
from tkinter import scrolledtext, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import vira4_6t
import time
import logging
import pyaudio
import wave
import traceback
from datetime import datetime
import importlib
import cupy as cp

device = vira4_6t.device
xp = vira4_6t.xp

importlib.reload(vira4_6t)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ViraGUI(tk.Tk):
    def __init__(self, loop):
        super().__init__()
        logger.debug("Starting ViraGUI __init__")
        self.title("Trium AI Interface")
        self.geometry("1000x700")
        self.style = ttk.Style()
        self.style.theme_use("darkly")

        self.plugin_manager = None
        self.autonomous_enabled = False
        self.is_recording = False
        self.audio_thread = None
        self.audio_queue = queue.Queue()
        self.input_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.last_input = None
        self.last_response = None
        self.last_perspective = None
        self.audio_enabled = False  # Default off per your request
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.loop = loop
        self.config = vira4_6t.config
        self.setup_gui(self.config)

        # Run setup_vira synchronously in a thread to ensure completion
        self.setup_future = asyncio.run_coroutine_threadsafe(self.setup_vira(), self.loop)
        self.after(100, self.check_setup_completion)
        self.after(100, self.check_queues)
        self.loop_thread = threading.Thread(target=self._run_asyncio_loop, daemon=True)
        self.loop_thread.start()
        logger.info(f"ViraGUI initialized on {'GPU' if xp is cp else 'CPU'} - awaiting plugin setup")

    def setup_gui(self, config):
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.pack(fill=BOTH, expand=True)

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=BOTH, expand=True, pady=(0, 10))

        # Trium’s Log Tab (default)
        self.log_frame = ttk.Frame(self.notebook)
        self.log_display = scrolledtext.ScrolledText(self.log_frame, height=20, width=80, wrap=tk.WORD)
        self.log_display.pack(fill=BOTH, expand=True)
        self.log_display.config(font=(config.get("font", "Helvetica"), config.get("font_size", 10)), bg="#1e1e1e", fg="white")
        self.log_display.tag_config("user", foreground="lightblue")
        self.log_display.tag_config("vira", foreground="lightgreen")
        self.log_display.tag_config("core", foreground="#ff8c00")
        self.log_display.tag_config("echo", foreground="#dda0dd")
        self.log_display.tag_config("trium", foreground="#ffffff")
        self.notebook.add(self.log_frame, text="Trium’s Log")

        # Memory Tab
        self.memory_frame = ttk.Frame(self.notebook)
        self.memory_display = scrolledtext.ScrolledText(self.memory_frame, height=20, width=80, wrap=tk.WORD)
        self.memory_display.pack(fill=BOTH, expand=True)
        self.memory_display.config(font=(config.get("font", "Helvetica"), config.get("font_size", 10)), bg="#1e1e1e", fg="white", state=tk.DISABLED)
        self.notebook.add(self.memory_frame, text="Memory")

        # Temporal Analysis Tab
        self.temporal_frame = ttk.Frame(self.notebook)
        self.temporal_display = scrolledtext.ScrolledText(self.temporal_frame, height=20, width=80, wrap=tk.WORD)
        self.temporal_display.pack(fill=BOTH, expand=True)
        self.temporal_display.config(font=(config.get("font", "Helvetica"), config.get("font_size", 10)), bg="#1e1e1e", fg="white", state=tk.DISABLED)
        self.notebook.add(self.temporal_frame, text="Temporal Analysis")

        # Goals Tab
        self.goals_frame = ttk.Frame(self.notebook)
        self.goals_display = scrolledtext.ScrolledText(self.goals_frame, height=20, width=80, wrap=tk.WORD)
        self.goals_display.pack(fill=BOTH, expand=True)
        self.goals_display.config(font=(config.get("font", "Helvetica"), config.get("font_size", 10)), bg="#1e1e1e", fg="white", state=tk.DISABLED)
        self.notebook.add(self.goals_frame, text="Goals")

        # Temporal Buttons
        self.temporal_button_frame = ttk.Frame(self.temporal_frame)
        self.temporal_button_frame.pack(fill=X, pady=5)
        ttk.Button(self.temporal_button_frame, text="Get Rhythm", command=self.get_rhythm, bootstyle=INFO).pack(side=LEFT, padx=5)
        ttk.Button(self.temporal_button_frame, text="Predict", command=self.predict, bootstyle=PRIMARY).pack(side=LEFT, padx=5)
        ttk.Button(self.temporal_button_frame, text="Query Date", command=self.query_date, bootstyle=WARNING).pack(side=LEFT, padx=5)
        ttk.Button(self.temporal_button_frame, text="Get Clusters", command=self.get_clusters, bootstyle=SUCCESS).pack(side=LEFT, padx=5)

        # Input Frame
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.pack(fill=X)
        self.input_text = tk.Text(self.input_frame, height=3, width=60, wrap=tk.WORD)
        self.input_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.input_text.config(font=(config.get("font", "Helvetica"), config.get("font_size", 10)), bg="#2b2b2b", fg="white", insertbackground="white")
        ttk.Button(self.input_frame, text="Talk to Trium", command=self.queue_input, style="Submit.TButton").grid(row=0, column=1, sticky="se", padx=5, pady=5)

        self.button_subframe = ttk.Frame(self.input_frame)
        self.button_subframe.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        self.audio_button = ttk.Button(self.button_subframe, text="🎤 Start Audio", command=self.toggle_audio, bootstyle=INFO)
        self.audio_button.pack(side=LEFT, padx=5)
        ttk.Button(self.button_subframe, text="🖼️ Upload Image", command=self.upload_image, bootstyle=PRIMARY).pack(side=LEFT, padx=5)
        ttk.Button(self.button_subframe, text="💭 Feedback", command=lambda: self.open_feedback_dialog, bootstyle=WARNING).pack(side=LEFT, padx=5)
        ttk.Button(self.button_subframe, text="📝 Export Diary", command=self.export_diary, bootstyle=SUCCESS).pack(side=LEFT, padx=5)

        # Control Frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=X, pady=10)
        self.audio_toggle = ttk.Checkbutton(self.control_frame, text="Audio Output", command=self.toggle_audio_output, bootstyle="success-round-toggle")
        self.audio_toggle.pack(side=LEFT, padx=5)
        self.autonomous_toggle = ttk.Checkbutton(self.control_frame, text="Autonomous Mode", command=self.toggle_autonomous, bootstyle="info-round-toggle")
        self.autonomous_toggle.pack(side=LEFT, padx=5)
        self.autonomous_status = ttk.Label(self.control_frame, text="Autonomous: Off | Pending: 0")
        self.autonomous_status.pack(side=LEFT, padx=5)

        # Command Frame
        self.command_frame = ttk.Frame(self.main_frame)
        self.command_frame.pack(fill=X, pady=5)
        self.command_entry = ttk.Entry(self.command_frame, width=60)
        self.command_entry.pack(fill=X, padx=5, pady=5)
        self.command_entry.insert(0, "Type /command (e.g., /council, /pause_autonomy)")
        self.command_entry.config(state="disabled")
        self.command_entry.bind("<Return>", self.execute_command)
        self.command_entry.bind("<FocusIn>", lambda e: self.command_entry.delete(0, tk.END) if self.command_entry.get().startswith("Type") else None)
        self.bind("<Control-Shift-P>", self.toggle_command_palette)

        # Status Bar
        self.status_bar = ttk.Label(self.main_frame, text="Trium State: Initializing | Last Action: None | Goals Active: 0", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=X, side=tk.BOTTOM, pady=5)

        # Persona Indicators
        self.persona_frame = ttk.Frame(self.main_frame)
        self.persona_frame.pack(fill=X, pady=5)
        self.vira_indicator = ttk.Label(self.persona_frame, text="● Vira", foreground="lightgreen")
        self.vira_indicator.pack(side=LEFT, padx=5)
        self.core_indicator = ttk.Label(self.persona_frame, text="● Core", foreground="#ff8c00")
        self.core_indicator.pack(side=LEFT, padx=5)
        self.echo_indicator = ttk.Label(self.persona_frame, text="● Echo", foreground="#dda0dd")
        self.echo_indicator.pack(side=LEFT, padx=5)

        self.input_frame.grid_rowconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.style.configure("Submit.TButton", font=("Helvetica", 12, "bold"), padding=10)

    async def setup_vira(self):
        try:
            greeting = await vira4_6t.setup_vira(gui=self, config=self.config)
            self.plugin_manager = vira4_6t.plugin_manager
            if not self.plugin_manager or not self.plugin_manager.plugins:
                logger.error("Plugin manager failed to initialize or no plugins loaded.")
                self.display_message("Error: Plugin initialization failed.")
                return
            self.display_message(greeting)
            self.update_status_bar("Active", "Initialized")
            logger.info(f"Trium setup completed in GUI on {'GPU' if xp is cp else 'CPU'}")
            await self.update_goals_display()  # Initial goals display
            # Schedule autonomous check-in explicitly
            asyncio.create_task(self.check_autonomous())
            logger.info("Autonomous check-in task scheduled in GUI.")
        except Exception as e:
            logger.error(f"Error during Trium setup: {e}\n{traceback.format_exc()}")
            self.display_message(f"Error during setup: {str(e)}")
            
    def check_setup_completion(self):
        if self.setup_future.done():
            try:
                self.setup_future.result()  # Raise any exception that occurred
            except Exception as e:
                logger.error(f"Setup failed: {e}")
                self.display_message(f"Setup failed: {str(e)}")
        else:
            self.after(100, self.check_setup_completion)

    def queue_input(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if text:
            self.input_text.delete("1.0", tk.END)
            asyncio.run_coroutine_threadsafe(self.process_input(text), self.loop)

    async def process_input(self, text):
        if not self.plugin_manager:
            self.display_message("Error: Trium is still initializing...")
            return
        try:
            # Pause autonomy
            await self.plugin_manager.execute_specific_plugin("autonomy_plugin", {"command": "set_autonomy", "enabled": False})
            self.update_status_bar("Paused", f"User Input: {text[:20]}...")
            emotion_result = await self.plugin_manager.execute_specific_plugin("vira_emotion_plugin", {"command": "analyze", "text": text})
            emotion_data = emotion_result if "error" not in emotion_result else None
            self.last_emotion_data = emotion_data
            self.display_message(f"Livia: {text}", emotion_data)
            await self.plugin_manager.execute_specific_plugin("autonomy_plugin", {"command": "user_input", "input_text": text})
            self.input_queue.put((0, text))
            # Resume autonomy after 5 minutes if no further input
            asyncio.create_task(self.resume_autonomy_after_delay(300))
        except Exception as e:
            logger.error(f"Error in process_input: {e}")
            self.display_message(f"Error: {e}")

    async def resume_autonomy_after_delay(self, delay):
        await asyncio.sleep(delay)
        if self.autonomous_enabled and (datetime.now() - self.last_input_time).total_seconds() >= delay:
            await self.plugin_manager.execute_specific_plugin("autonomy_plugin", {"command": "set_autonomy", "enabled": True})
            self.update_status_bar("Active", "Resumed Autonomy")

    def check_queues(self):
        try:
            while True:
                item = self.input_queue.get_nowait()
                priority, text = item if isinstance(item, tuple) and len(item) == 2 else (0, item)
                if isinstance(text, dict) and "image_path" in text:
                    asyncio.run_coroutine_threadsafe(self._process_image_input_batch([text["image_path"]]), self.loop)
                else:
                    asyncio.run_coroutine_threadsafe(self._process_text_input(text), self.loop)
                self.input_queue.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error in check_queues: {e}")
            self.display_message(f"Error: {e}")
        self.after(100, self.check_queues)

    def toggle_command_palette(self, event=None):
        if self.command_entry.cget("state") == "disabled":
            self.command_entry.config(state="normal")
            self.command_entry.delete(0, tk.END)
            self.command_entry.focus_set()
        else:
            self.command_entry.config(state="disabled")
            self.command_entry.insert(0, "Type /command (e.g., /council, /pause_autonomy)")
            self.input_text.focus_set()

    def execute_command(self, event=None):
        command = self.command_entry.get().strip()
        if not command.startswith("/"):
            self.display_message("Error: Commands must start with '/'")
            return
        command = command[1:].lower()
        commands = {
            "council": lambda: self.queue_input("council meeting"),
            "toggle_audio": self.toggle_audio,
            "toggle_autonomy": self.toggle_autonomous,
            "toggle_output": self.toggle_audio_output,
            "upload_image": self.upload_image,
            "clear": lambda: self.log_display.delete("1.0", tk.END),
            "pause_autonomy": lambda: asyncio.run_coroutine_threadsafe(self.pause_autonomy(), self.loop),
            "resume_autonomy": lambda: asyncio.run_coroutine_threadsafe(self.resume_autonomy(), self.loop),
            "exit": self.on_closing
        }
        action = commands.get(command)
        if action:
            try:
                action()
                self.display_message(f"Command executed: /{command}")
            except Exception as e:
                self.display_message(f"Error executing /{command}: {str(e)}")
        else:
            self.display_message(f"Unknown command: /{command}")
        self.toggle_command_palette()

    async def pause_autonomy(self):
        await self.plugin_manager.execute_specific_plugin("autonomy_plugin", {"command": "set_autonomy", "enabled": False})
        self.update_status_bar("Paused", "Manual Pause")

    async def resume_autonomy(self):
        if self.autonomous_enabled:
            await self.plugin_manager.execute_specific_plugin("autonomy_plugin", {"command": "set_autonomy", "enabled": True})
            self.update_status_bar("Active", "Manual Resume")

    def export_diary(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "w") as f:
                f.write("Trium’s Diary\n=============\n\n")
                content = self.log_display.get("1.0", tk.END).strip()
                f.write(content.replace("\n", "\n\n"))
            self.display_message(f"Diary exported to {file_path}")

    def toggle_audio(self):
        self.is_recording = not self.is_recording
        self.audio_button.config(text="🎤 Stop Audio" if self.is_recording else "🎤 Start Audio")
        if self.is_recording:
            self.stop_event.clear()
            self.audio_thread = threading.Thread(target=self.record_audio, daemon=True)
            self.audio_thread.start()
        else:
            self.stop_event.set()
            if self.audio_thread:
                self.audio_thread.join(timeout=1)
                self.audio_thread = None
            asyncio.run_coroutine_threadsafe(self._debounce_transcription(), self.loop)

    def toggle_audio_output(self):
        self.audio_enabled = not self.audio_enabled
        logger.info("Audio output %s", "enabled" if self.audio_enabled else "disabled")

    def toggle_autonomous(self):
        self.autonomous_enabled = not self.autonomous_enabled
        asyncio.run_coroutine_threadsafe(
            self.plugin_manager.execute_specific_plugin("autonomy_plugin", {"command": "set_autonomy", "enabled": self.autonomous_enabled}),
            self.loop
        )
        self.autonomous_status.config(text=f"Autonomous: {'On' if self.autonomous_enabled else 'Off'} | Pending: {self.input_queue.qsize()}")
        logger.info("Autonomous mode %s", "enabled" if self.autonomous_enabled else "disabled")

    def record_audio(self):
        p = pyaudio.PyAudio()
        stream = None
        frames = []
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
            logger.info("Recording audio...")
            while self.is_recording and not self.stop_event.is_set():
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            self.audio_queue.put(frames)
        except Exception as e:
            logger.error(f"Error in record_audio: {e}")
            self.display_message("Error: Audio recording failed.")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

    async def _debounce_transcription(self):
        await asyncio.sleep(0.5)
        if not self.audio_queue.empty():
            frames = self.audio_queue.get()
            audio_path = "temp_audio.wav"
            p = pyaudio.PyAudio()
            try:
                wf = wave.open(audio_path, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(frames))
                wf.close()
                transcription = vira4_6t.whisper_model.transcribe(audio_path)["text"].strip()
                if transcription:
                    priority_data = await self.plugin_manager.execute_specific_plugin("thala_plugin", {"command": "prioritize", "input_text": transcription})
                    priority = priority_data.get("priority", 0.3) if "error" not in priority_data else 0.3
                    self.input_queue.put((priority, transcription))
                    self.display_message(f"Livia (audio): {transcription}")
                os.remove(audio_path)
            except Exception as e:
                logger.error(f"Error in transcription: {e}")
                self.display_message("Error: Audio transcription failed.")
            finally:
                p.terminate()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.input_queue.put((0.5, {"image_path": file_path}))
            self.display_message(f"Livia uploaded image: {os.path.basename(file_path)}")

    def get_rhythm(self):
        asyncio.run_coroutine_threadsafe(self._fetch_rhythm(), self.loop)

    async def _fetch_rhythm(self):
        if not self.plugin_manager:
            self.display_temporal_message("Error: Plugin manager not initialized.")
            return
        result = await self.plugin_manager.execute_specific_plugin("temporal_plugin", {"command": "get_rhythm"})
        if "error" not in result:
            self.display_temporal_message("Trium Rhythm:\n" + self._format_rhythm(result["rhythms"]["user"]))
            for persona in ["Vira", "Core", "Echo"]:
                self.display_temporal_message(f"{persona} Rhythm:\n" + self._format_rhythm(result["rhythms"][persona]))
        else:
            self.display_temporal_message(f"Error fetching rhythm: {result['error']}")

    def predict(self):
        asyncio.run_coroutine_threadsafe(self._fetch_prediction(), self.loop)

    async def _fetch_prediction(self):
        if not self.plugin_manager:
            self.display_temporal_message("Error: Plugin manager not initialized.")
            return
        result = await self.plugin_manager.execute_specific_plugin("temporal_plugin", {"command": "predict"})
        if "error" not in result:
            self.display_temporal_message("Trium Prediction:\n" + self._format_prediction(result["predictions"]["user"]))
            for persona in ["Vira", "Core", "Echo"]:
                self.display_temporal_message(f"{persona} Prediction:\n" + self._format_prediction(result["predictions"][persona]))
        else:
            self.display_temporal_message(f"Error fetching prediction: {result['error']}")

    def query_date(self):
        dialog = ttk.Toplevel(self)
        dialog.title("Query Date")
        dialog.geometry("300x150")
        dialog.transient(self)
        dialog.grab_set()
        ttk.Label(dialog, text="Enter date (YYYY-MM-DD):").pack(pady=5)
        date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(dialog, textvariable=date_var).pack(pady=5)
        ttk.Button(dialog, text="Submit", command=lambda: [asyncio.run_coroutine_threadsafe(self._fetch_date(date_var.get().strip()), self.loop), dialog.destroy()], bootstyle=SUCCESS).pack(pady=10)

    async def _fetch_date(self, date_str):
        if not self.plugin_manager:
            self.display_temporal_message("Error: Plugin manager not initialized.")
            return
        result = await self.plugin_manager.execute_specific_plugin("temporal_plugin", {"command": "query_date", "date": date_str})
        if "error" not in result:
            self.display_temporal_message(f"Date Query ({date_str}):\nTrium: {result['user']['description']}")
            for persona, data in result["personas"].items():
                self.display_temporal_message(f"{persona}: {data['description']}")
        else:
            self.display_temporal_message(f"Error querying date: {result['error']}")

    def display_temporal_message(self, message):
        with threading.Lock():
            self.temporal_display.config(state=tk.NORMAL)
            self.temporal_display.insert(tk.END, message + "\n\n")
            self.temporal_display.see(tk.END)
            self.temporal_display.config(state=tk.DISABLED)

    def _format_rhythm(self, rhythm):
        return (f"Mean Activity Gap: {rhythm['mean_activity_gap_hours']:.2f} hours\n"
                f"Main Cycle: {rhythm['main_cycle_hours']:.2f} hours\n"
                f"Dominant Emotion: {rhythm['dominant_emotion']}\n"
                f"Dominant Polyvagal State: {rhythm['dominant_polyvagal']}\n"
                f"Active Hours: {len(rhythm['active_hours'])} logged")

    def _format_prediction(self, prediction):
        return (f"Next Intensity: {prediction['next_intensity']:.2f}\n"
                f"Next Emotion: {prediction['next_emotion']}\n"
                f"Next Polyvagal State: {prediction['next_polyvagal']}\n"
                f"Trend: {prediction['trend']}\n"
                f"Time Context: {prediction['time_context']}")

    def display_message(self, message, emotion=None):
        with threading.Lock():
            self.log_display.config(state=tk.NORMAL)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tag = "vira"
            if message.startswith("Livia"): tag = "user"
            elif message.startswith("Vira:"): tag = "vira"
            elif message.startswith("Core:"): tag = "core"
            elif message.startswith("Echo:"): tag = "echo"
            elif message.startswith("Trium:"): tag = "trium"
            emotion_str = f" [Emotion: {emotion['emotion_type']} ({emotion['emotion_intensity']})]" if emotion and emotion.get("emotion_type") != "unknown" else ""
            formatted_message = f"[{timestamp}] {message}{emotion_str}"
            self.log_display.insert(tk.END, formatted_message + "\n", tag)
            self.log_display.see(tk.END)
            self.log_display.config(state=tk.DISABLED)
            self.update_persona_indicators(message)

    def update_persona_indicators(self, message):
        if "Vira" in message:
            self.vira_indicator.config(text="● Vira (Active)")
        else:
            self.vira_indicator.config(text="● Vira")
        if "Core" in message:
            self.core_indicator.config(text="● Core (Active)")
        else:
            self.core_indicator.config(text="● Core")
        if "Echo" in message:
            self.echo_indicator.config(text="● Echo (Active)")
        else:
            self.echo_indicator.config(text="● Echo")

    def update_status_bar(self, state, last_action):
        goals_active = len([g for g in self.plugin_manager.plugins["autonomy_plugin"].persona_goals.values() if g["progress"] < 1.0]) if self.plugin_manager else 0
        self.autonomous_status.config(text=f"Autonomous: {'On' if self.autonomous_enabled else 'Off'} | Pending: {self.input_queue.qsize()}")
        self.status_bar.config(text=f"Trium State: {state} | Last Action: {last_action} | Goals Active: {goals_active}")

    async def update_goals_display(self):
        if not self.plugin_manager:
            return
        result = await self.plugin_manager.execute_specific_plugin("autonomy_plugin", {"command": "get_goals"})
        self.goals_display.config(state=tk.NORMAL)
        self.goals_display.delete("1.0", tk.END)
        self.goals_display.insert(tk.END, "Trium’s Goals:\n=============\n")
        if "goals" in result:
            for persona, goal_data in result["goals"].items():
                self.goals_display.insert(tk.END, f"{persona}: {goal_data['goal']} (Progress: {goal_data['progress']*100:.0f}%, Priority: {goal_data['priority']:.2f})\n")
        self.goals_display.config(state=tk.DISABLED)

    def get_clusters(self):
        asyncio.run_coroutine_threadsafe(self._fetch_clusters(), self.loop)

    async def _fetch_clusters(self):
        if not self.plugin_manager:
            self.display_temporal_message("Error: Plugin manager not initialized.")
            return
        try:
            result = await self.plugin_manager.execute_specific_plugin("vira_emotion_plugin", {"command": "clusters"})
            if result.get("status") == "clusters_retrieved" and "clusters" in result:
                clusters = result["clusters"]
                if not clusters:
                    self.display_temporal_message("No emotion clusters available yet.")
                else:
                    self.display_temporal_message("Emotion Clusters:")
                    for emotion, data in clusters.items():
                        centroid = self.xp.array(data["centroid"])
                        self.display_temporal_message(f"{emotion}: Count={data['count']}, Centroid Norm={self.xp.linalg.norm(centroid):.2f}")
            else:
                self.display_temporal_message(f"Error fetching clusters: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error fetching clusters: {e}")
            self.display_temporal_message(f"Error: {e}")

    async def display_vira_response(self, text, perspective):
        self.display_message(f"{perspective}: {text}")
        if self.audio_enabled and "tts_plugin" in self.plugin_manager.plugins:
            await self.plugin_manager.execute_specific_plugin("tts_plugin", {"text": text, "perspective": perspective})
        await vira4_6t.queue_memory_save(text, perspective)
        self.update_status_bar("Active", f"{perspective}: {text[:20]}...")
        await self.update_goals_display()

    async def _process_text_input(self, text):
        self.last_input = text
        self.last_input_time = datetime.now()
        try:
            prompt, perspective = await vira4_6t.context_manager.get_prompt(text)
            response = await vira4_6t.ollama_query(vira4_6t.LLAMA_MODEL_NAME, prompt)
            emotion_data = await self.plugin_manager.execute_specific_plugin("vira_emotion_plugin", {"command": "analyze", "text": response})
            await self.display_vira_response(response, perspective)
            vira4_6t.context_manager.add_to_short_term_memory("User", text, emotion_data.get("emotion_type", "unknown") if emotion_data else "unknown")
            await vira4_6t.queue_memory_save(text, "User", emotion_data.get("emotion_type", "unknown") if emotion_data else "unknown", emotion_data.get("emotion_intensity", 0) if emotion_data else 0)
            await self.update_memory_display()
            self.last_response = response
            self.last_perspective = perspective
        except Exception as e:
            logger.error(f"Error in _process_text_input: {e}")
            self.display_message(f"Error: {e}")

    async def _process_image_input_batch(self, image_paths):
        for image_path in image_paths:
            result = await vira4_6t.describe_image(image_path)
            if result["success"]:
                description = result["description"]
                self.last_input = description
                prompt, perspective = await vira4_6t.context_manager.get_prompt(description)
                response = await vira4_6t.ollama_query(vira4_6t.LLAVA_MODEL_NAME, prompt)
                await self.display_vira_response(response, perspective)
                await self.update_memory_display()
            else:
                self.display_message(f"Error: {result['error']}")

    async def check_autonomous(self):
        while True:
            if self.autonomous_enabled and self.plugin_manager:
                try:
                    result = await self.plugin_manager.execute_specific_plugin("autonomy_plugin", {"command": "checkin"})
                    if "response" in result:
                        perspective = result.get("perspective", "Vira")
                        response = result["response"]
                        await self.display_vira_response(response, perspective)
                        await self.update_memory_display()
                except Exception as e:
                    logger.error(f"Error in autonomous check: {e}")
            await asyncio.sleep(self.config["autonomy_settings"]["checkin_interval"])  # Sync with config

    async def update_memory_display(self):
        self.memory_display.config(state=tk.NORMAL)
        self.memory_display.delete("1.0", tk.END)
        self.memory_display.insert(tk.END, "Recent Trium Memories (Last 5):\n")
        with vira4_6t.context_manager.memory_lock:
            for mem in vira4_6t.context_manager.short_term_memory[-5:]:
                emotion_str = f" (Emotion: {mem['emotion']})" if mem['emotion'] != "unknown" else ""
                self.memory_display.insert(tk.END, f"{mem['role']}: {mem['content']}{emotion_str}\n")
        self.memory_display.config(state=tk.DISABLED)

    def on_closing(self):
        self.is_recording = False
        self.stop_event.set()
        if self.audio_thread:
            self.audio_thread.join(timeout=1)
        vira4_6t.shutdown_memory_queue()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.destroy()

    def _run_asyncio_loop(self):
        self.loop.run_forever()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = ViraGUI(loop)
    app.mainloop()