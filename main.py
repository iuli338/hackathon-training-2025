import customtkinter as ctk
import cv2
from PIL import Image
import warnings
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
import psutil
import time
from collections import deque

# Suppress annoying warnings
warnings.filterwarnings("ignore")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ============================================
# SUBPROCESS 3: Performance Monitor Window
# ============================================
def run_performance_monitor(parent_pid, stop_event, shared_fps):
    """Runs the performance monitor in a separate process"""
    import os

    class PerformanceWindow(ctk.CTk):
        def __init__(self, parent_pid, shared_fps):
            super().__init__()

            self.parent_pid = parent_pid
            self.parent_process = psutil.Process(parent_pid)
            self.shared_fps = shared_fps

            # Window setup
            self.title("Performance Analytics")
            self.geometry("450x620")
            self.resizable(False, False)
            self.attributes('-topmost', True)

            # Performance monitoring
            self.cpu_history = deque(maxlen=60)
            self.ram_history = deque(maxlen=60)
            self.max_ram_mb = 0

            # Main container with dark professional background
            container = ctk.CTkFrame(self, fg_color="#0f0f0f")
            container.pack(fill="both", expand=True, padx=0, pady=0)

            # Header section with gradient-like effect
            header = ctk.CTkFrame(container, fg_color="#1a1a1a", corner_radius=0, height=60)
            header.pack(fill="x", padx=0, pady=0)
            header.pack_propagate(False)

            title = ctk.CTkLabel(header, text="SYSTEM PERFORMANCE",
                                font=("Segoe UI", 18, "bold"), text_color="#ffffff")
            title.pack(pady=(12, 0))

            subtitle = ctk.CTkLabel(header, text="Real-time Resource Monitor",
                                   font=("Segoe UI", 10), text_color="#888888")
            subtitle.pack()

            # Content container
            content = ctk.CTkFrame(container, fg_color="transparent")
            content.pack(fill="both", expand=True, padx=20, pady=15)

            # Stats frame with modern card design
            stats_frame = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12,
                                      border_width=1, border_color="#2a2a2a")
            stats_frame.pack(fill="x", pady=(0, 15))

            # Grid layout for stats
            stats_frame.grid_columnconfigure((0, 1, 2), weight=1)

            # CPU stat
            cpu_container = ctk.CTkFrame(stats_frame, fg_color="transparent")
            cpu_container.grid(row=0, column=0, padx=15, pady=15)

            ctk.CTkLabel(cpu_container, text="CPU",
                        font=("Segoe UI", 9, "bold"), text_color="#888888").pack()
            self.cpu_label = ctk.CTkLabel(cpu_container, text="0.0%",
                                         font=("Segoe UI", 24, "bold"), text_color="#4CAF50",
                                         width=90, anchor="center")
            self.cpu_label.pack()

            # RAM stat
            ram_container = ctk.CTkFrame(stats_frame, fg_color="transparent")
            ram_container.grid(row=0, column=1, padx=15, pady=15)

            ctk.CTkLabel(ram_container, text="MEMORY",
                        font=("Segoe UI", 9, "bold"), text_color="#888888").pack()
            self.ram_label = ctk.CTkLabel(ram_container, text="0 MB",
                                         font=("Segoe UI", 24, "bold"), text_color="#2196F3",
                                         width=90, anchor="center")
            self.ram_label.pack()

            # FPS stat
            fps_container = ctk.CTkFrame(stats_frame, fg_color="transparent")
            fps_container.grid(row=0, column=2, padx=15, pady=15)

            ctk.CTkLabel(fps_container, text="FRAMERATE",
                        font=("Segoe UI", 9, "bold"), text_color="#888888").pack()
            self.fps_label = ctk.CTkLabel(fps_container, text="0",
                                         font=("Segoe UI", 24, "bold"), text_color="#FF9800",
                                         width=90, anchor="center")
            self.fps_label.pack()

            # CPU Graph frame with professional card design
            cpu_graph_frame = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12,
                                          border_width=1, border_color="#2a2a2a")
            cpu_graph_frame.pack(fill="both", expand=True, pady=(0, 12))

            cpu_header = ctk.CTkFrame(cpu_graph_frame, fg_color="transparent")
            cpu_header.pack(fill="x", padx=15, pady=(10, 5))

            ctk.CTkLabel(cpu_header, text="CPU UTILIZATION",
                        font=("Segoe UI", 11, "bold"), text_color="#ffffff").pack(side="left")
            ctk.CTkLabel(cpu_header, text="0-100%",
                        font=("Segoe UI", 9), text_color="#666666").pack(side="right")

            # Canvas for CPU graph
            self.cpu_graph_canvas = ctk.CTkCanvas(cpu_graph_frame, bg="#0f0f0f",
                                                  highlightthickness=0, height=110)
            self.cpu_graph_canvas.pack(pady=(0, 10), padx=12, fill="both", expand=True)

            # RAM Graph frame with professional card design
            ram_graph_frame = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12,
                                          border_width=1, border_color="#2a2a2a")
            ram_graph_frame.pack(fill="both", expand=True)

            ram_header = ctk.CTkFrame(ram_graph_frame, fg_color="transparent")
            ram_header.pack(fill="x", padx=15, pady=(10, 5))

            # Get total system RAM for scale
            total_ram_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            ctk.CTkLabel(ram_header, text="MEMORY USAGE",
                        font=("Segoe UI", 11, "bold"), text_color="#ffffff").pack(side="left")

            # Right side container for scale and max
            right_info = ctk.CTkFrame(ram_header, fg_color="transparent")
            right_info.pack(side="right")

            ctk.CTkLabel(right_info, text=f"0-{total_ram_gb:.0f} GB",
                        font=("Segoe UI", 9), text_color="#666666").pack(side="top", anchor="e")
            self.max_ram_label = ctk.CTkLabel(right_info, text="Peak: 0 MB",
                        font=("Segoe UI", 8), text_color="#2196F3")
            self.max_ram_label.pack(side="top", anchor="e")

            # Canvas for RAM graph
            self.ram_graph_canvas = ctk.CTkCanvas(ram_graph_frame, bg="#0f0f0f",
                                                  highlightthickness=0, height=110)
            self.ram_graph_canvas.pack(pady=(0, 10), padx=12, fill="both", expand=True)

            # FPS tracking (no longer tracked here, read from shared value)

            # Start updating
            self.update_metrics()

            # Handle close
            self.protocol("WM_DELETE_WINDOW", self.on_closing)

        def update_metrics(self):
            try:
                # Check if parent process still exists
                if not self.parent_process.is_running():
                    self.on_closing()
                    return

                # Get CPU usage for parent application
                cpu_percent = self.parent_process.cpu_percent(interval=0)

                # Get RAM usage for parent and its children
                app_ram_mb = self.parent_process.memory_info().rss / (1024 * 1024)

                try:
                    children = self.parent_process.children(recursive=True)
                    for child in children:
                        try:
                            app_ram_mb += child.memory_info().rss / (1024 * 1024)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                # Store in history
                self.cpu_history.append(cpu_percent)
                self.ram_history.append(app_ram_mb)

                # Track max RAM usage
                if app_ram_mb > self.max_ram_mb:
                    self.max_ram_mb = app_ram_mb

                # Read FPS from shared value
                current_fps = self.shared_fps.value

                # Update labels with professional formatting
                self.cpu_label.configure(text=f"{cpu_percent:.1f}%")
                self.ram_label.configure(text=f"{app_ram_mb:.0f} MB")
                self.fps_label.configure(text=f"{current_fps:.0f}")
                self.max_ram_label.configure(text=f"Peak: {self.max_ram_mb:.0f} MB")

                # Draw graphs
                self.draw_cpu_graph()
                self.draw_ram_graph()

                # Schedule next update
                self.after(100, self.update_metrics)
            except:
                self.on_closing()

        def draw_cpu_graph(self):
            """Draw professional CPU usage graph"""
            if not self.cpu_graph_canvas:
                return

            self.cpu_graph_canvas.delete("all")

            width = self.cpu_graph_canvas.winfo_width()
            height = self.cpu_graph_canvas.winfo_height()

            if width < 10 or height < 10:
                return

            # Draw subtle gridlines
            grid_color = "#1a1a1a"
            for i in range(1, 5):
                y = (i / 5) * height
                self.cpu_graph_canvas.create_line(0, y, width, y, fill=grid_color, width=1, dash=(2, 4))

            # Draw CPU graph with gradient effect
            if len(self.cpu_history) > 1:
                points = []
                for i, val in enumerate(self.cpu_history):
                    x = (i / max(len(self.cpu_history) - 1, 1)) * width
                    y = height - (val / 100.0) * height
                    points.append((x, y))

                # Draw filled area under curve
                if points:
                    area_points = [(0, height)] + points + [(width, height)]
                    self.cpu_graph_canvas.create_polygon(area_points, fill="#4CAF50",
                                                         outline="", stipple="gray25")

                # Draw main line with shadow effect
                for i in range(len(points) - 1):
                    # Shadow
                    self.cpu_graph_canvas.create_line(points[i][0]+1, points[i][1]+1,
                                                      points[i+1][0]+1, points[i+1][1]+1,
                                                      fill="#000000", width=2, smooth=True)
                    # Main line
                    self.cpu_graph_canvas.create_line(points[i][0], points[i][1],
                                                      points[i+1][0], points[i+1][1],
                                                      fill="#4CAF50", width=3, smooth=True)

        def draw_ram_graph(self):
            """Draw professional RAM usage graph"""
            if not self.ram_graph_canvas:
                return

            self.ram_graph_canvas.delete("all")

            width = self.ram_graph_canvas.winfo_width()
            height = self.ram_graph_canvas.winfo_height()

            if width < 10 or height < 10:
                return

            # Draw subtle gridlines
            grid_color = "#1a1a1a"
            for i in range(1, 5):
                y = (i / 5) * height
                self.ram_graph_canvas.create_line(0, y, width, y, fill=grid_color, width=1, dash=(2, 4))

            # Draw RAM graph with gradient effect
            if len(self.ram_history) > 1:
                total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
                points = []
                for i, val in enumerate(self.ram_history):
                    x = (i / max(len(self.ram_history) - 1, 1)) * width
                    # Ensure the value doesn't exceed total RAM
                    normalized_val = min(val / total_ram_mb, 1.0)
                    y = height - (normalized_val * height)
                    points.append((x, y))

                # Draw filled area under curve
                if points:
                    area_points = [(0, height)] + points + [(width, height)]
                    self.ram_graph_canvas.create_polygon(area_points, fill="#2196F3",
                                                         outline="", stipple="gray25")

                # Draw main line with shadow effect
                for i in range(len(points) - 1):
                    # Shadow
                    self.ram_graph_canvas.create_line(points[i][0]+1, points[i][1]+1,
                                                      points[i+1][0]+1, points[i+1][1]+1,
                                                      fill="#000000", width=2, smooth=True)
                    # Main line
                    self.ram_graph_canvas.create_line(points[i][0], points[i][1],
                                                      points[i+1][0], points[i+1][1],
                                                      fill="#2196F3", width=3, smooth=True)

        def on_closing(self):
            self.destroy()

    # Run the window
    try:
        app = PerformanceWindow(parent_pid, shared_fps)
        app.mainloop()
    except:
        pass

# ============================================
# Helper Function: Create Error Icon
# ============================================
def create_error_frame(width=640, height=480):
    """Creates a gray frame with a crossed webcam icon"""
    # Create gray background
    frame = np.ones((height, width, 3), dtype=np.uint8) * 128

    # Draw webcam icon (simplified rectangle with lens circle)
    center_x, center_y = width // 2, height // 2

    # Webcam body (rectangle)
    cv2.rectangle(frame,
                  (center_x - 80, center_y - 60),
                  (center_x + 80, center_y + 60),
                  (200, 200, 200), 3)

    # Lens (circle)
    cv2.circle(frame, (center_x, center_y), 35, (200, 200, 200), 3)

    # Red X (crossed lines)
    cv2.line(frame,
             (center_x - 60, center_y - 60),
             (center_x + 60, center_y + 60),
             (0, 0, 255), 4)
    cv2.line(frame,
             (center_x + 60, center_y - 60),
             (center_x - 60, center_y + 60),
             (0, 0, 255), 4)

    # Add text
    cv2.putText(frame, "Camera Not Available",
                (center_x - 150, center_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    return frame

# ============================================
# SUBPROCESS 1: Frame Capture from Webcam
# ============================================
def capture_frames(frame_queue, stop_event):
    """Continuously captures frames from webcam and puts them in queue"""
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        frame_queue.put("ERROR")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            # Only put frame if queue is not full (avoid memory buildup)
            if not frame_queue.full():
                frame_queue.put(frame)
        else:
            # Camera disconnected or failed
            frame_queue.put("ERROR")
            break

    cap.release()

# ============================================
# SUBPROCESS 2: Face Detection
# ============================================
def detect_faces(frame_queue, result_queue, stop_event):
    """Takes frames from queue, detects faces, and puts results in result queue"""
    # Load the Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while not stop_event.is_set():
        try:
            # Get frame from queue (timeout to allow checking stop_event)
            if not frame_queue.empty():
                frame = frame_queue.get(timeout=0.1)

                # Check if it's an error signal
                if isinstance(frame, str) and frame == "ERROR":
                    result_queue.put("ERROR")
                    break

                # Convert to grayscale for detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Draw rectangles around faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Put processed frame in result queue
                if not result_queue.full():
                    result_queue.put(frame)
        except:
            continue

class FaceIDApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Face Shield Pro - Live Feed")
        self.geometry("900x600")
        
        # Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo = ctk.CTkLabel(self.sidebar, text="FaceID System", font=("Helvetica", 20, "bold"))
        self.logo.grid(row=0, column=0, padx=20, pady=20)

        # Buttons
        self.btn_start = ctk.CTkButton(self.sidebar, text="START CAMERA", fg_color="#2ecc71", hover_color="#27ae60", command=self.start_camera)
        self.btn_start.grid(row=1, column=0, padx=20, pady=10)

        self.btn_stop = ctk.CTkButton(self.sidebar, text="STOP CAMERA", fg_color="#e74c3c", hover_color="#c0392b", command=self.stop_camera)
        self.btn_stop.grid(row=2, column=0, padx=20, pady=10)

        # --- Main Camera Area ---
        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.status_label = ctk.CTkLabel(self.main_view, text="Status: Ready", font=("Arial", 14))
        self.status_label.pack(anchor="w", pady=(0, 10))

        # This Frame holds the camera image
        self.camera_frame = ctk.CTkFrame(self.main_view, fg_color="#1a1a1a", corner_radius=15)
        self.camera_frame.pack(expand=True, fill="both")

        # The actual Label where the image will be placed
        self.cam_display = ctk.CTkLabel(self.camera_frame, text="")
        self.cam_display.pack(expand=True, fill="both", padx=10, pady=10)

        # --- Variables ---
        self.is_running = False

        # Multiprocessing setup
        self.frame_queue = None
        self.result_queue = None
        self.stop_event = None
        self.capture_process = None
        self.detection_process = None

        # Performance monitoring
        self.debug_mode = False
        self.monitor_process = None
        self.monitor_stop_event = None
        self.shared_fps = None

        # FPS tracking for camera feed
        self.fps_counter = 0
        self.fps_start_time = None

        # Bind Tab key to toggle debug mode
        self.bind("<Tab>", self.toggle_debug_mode)

    def start_camera(self):
        if not self.is_running:
            # Create queues for inter-process communication
            self.frame_queue = Queue(maxsize=2)  # Small buffer to avoid lag
            self.result_queue = Queue(maxsize=2)
            self.stop_event = mp.Event()

            # Create shared FPS counter
            if self.shared_fps is None:
                self.shared_fps = mp.Value('d', 0.0)

            # Reset FPS tracking
            self.fps_counter = 0
            self.fps_start_time = time.time()

            # Start the two subprocesses
            self.capture_process = Process(target=capture_frames, args=(self.frame_queue, self.stop_event))
            self.detection_process = Process(target=detect_faces, args=(self.frame_queue, self.result_queue, self.stop_event))

            self.capture_process.start()
            self.detection_process.start()

            self.is_running = True
            self.status_label.configure(text="Status: Live Feed Active (Multiprocessing)", text_color="#2ecc71")
            self.update_feed()

    def stop_camera(self):
        if self.is_running:
            self.is_running = False

            # Signal processes to stop
            if self.stop_event:
                self.stop_event.set()

            # Immediately terminate processes without waiting
            if self.capture_process and self.capture_process.is_alive():
                self.capture_process.terminate()

            if self.detection_process and self.detection_process.is_alive():
                self.detection_process.terminate()

            self.status_label.configure(text="Status: Camera Stopped", text_color="#e74c3c")
            # Clear the image
            self.cam_display.configure(image=None)

    def update_feed(self):
        if self.is_running:
            # Check if there's a processed frame available
            if not self.result_queue.empty():
                frame = self.result_queue.get()

                # Check if it's an error signal
                if isinstance(frame, str) and frame == "ERROR":
                    # Display error frame with crossed webcam icon
                    error_frame = create_error_frame()
                    frame_rgb = cv2.cvtColor(error_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    ctk_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
                    self.cam_display.configure(image=ctk_img)
                    self.status_label.configure(text="Status: Camera Error", text_color="#e74c3c")
                    # Stop the camera after showing error
                    self.after(2000, self.stop_camera)
                    return

                # 1. Convert Color: OpenCV uses BGR, we need RGB for UI
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 2. Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)

                # 3. Convert to CustomTkinter Image
                ctk_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))

                # 4. Update the Label
                self.cam_display.configure(image=ctk_img)

                # 5. Track FPS
                self.fps_counter += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    current_fps = self.fps_counter / elapsed
                    if self.shared_fps is not None:
                        self.shared_fps.value = current_fps
                    self.fps_counter = 0
                    self.fps_start_time = time.time()

            # 6. Repeat this function after 10ms
            self.after(10, self.update_feed)

    def toggle_debug_mode(self, event=None):
        """Toggle debug mode on/off with Tab key"""
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            self.start_performance_monitor()
        else:
            self.stop_performance_monitor()
        return "break"  # Prevent default Tab behavior

    def start_performance_monitor(self):
        """Start the performance monitor in a separate process"""
        if self.monitor_process is None or not self.monitor_process.is_alive():
            import os
            # Create shared FPS if it doesn't exist
            if self.shared_fps is None:
                self.shared_fps = mp.Value('d', 0.0)

            self.monitor_stop_event = mp.Event()
            parent_pid = os.getpid()
            self.monitor_process = Process(target=run_performance_monitor,
                                          args=(parent_pid, self.monitor_stop_event, self.shared_fps))
            self.monitor_process.start()

    def stop_performance_monitor(self):
        """Stop the performance monitor process"""
        if self.monitor_process and self.monitor_process.is_alive():
            if self.monitor_stop_event:
                self.monitor_stop_event.set()
            self.monitor_process.terminate()
            self.monitor_process = None

    def on_closing(self):
        # Clean up properly when closing the app
        self.stop_camera()
        # Close performance monitor if open
        self.stop_performance_monitor()
        self.destroy()

if __name__ == "__main__":
    app = FaceIDApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()