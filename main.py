import customtkinter as ctk
import cv2
from PIL import Image
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

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
        self.cap = None
        self.is_running = False

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0) # 0 is usually the default webcam
            self.is_running = True
            self.status_label.configure(text="Status: Live Feed Active", text_color="#2ecc71")
            self.update_feed()

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.status_label.configure(text="Status: Camera Stopped", text_color="#e74c3c")
            # Clear the image
            self.cam_display.configure(image=None)

    def update_feed(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # 1. Convert Color: OpenCV uses BGR, we need RGB for UI
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 2. Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # 3. Convert to CustomTkinter Image
                # We define the size to ensure it fits the window nicely
                ctk_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
                
                # 4. Update the Label
                self.cam_display.configure(image=ctk_img)
                
                # 5. Repeat this function after 10ms
                self.after(10, self.update_feed)
            else:
                self.stop_camera() # Stop if camera disconnects

    def on_closing(self):
        # Clean up properly when closing the app
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = FaceIDApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()