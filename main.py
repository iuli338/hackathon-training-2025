import customtkinter as ctk
import cv2
from PIL import Image
import warnings
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
import time
from performance_monitor import run_performance_monitor

# Suppress annoying warnings
warnings.filterwarnings("ignore")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

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