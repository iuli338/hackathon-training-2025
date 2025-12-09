import customtkinter as ctk
import psutil
from collections import deque


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
