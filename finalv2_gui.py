import os
import json
import tkinter as tk
from tkinter import Label, Button, Entry, messagebox
from tkinter import ttk  # Used for progress bar
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import subprocess

# Paths setup
RUN_NAME = "DDPM_Unconditional"
RESULTS_DIR = os.path.join("results", RUN_NAME)
LOG_FILE = os.path.join(RESULTS_DIR, "training_log.json")

class TrainingViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("DDPM Training Results")

        # Image display (for current epoch output image)
        self.image_label = Label(root)
        self.image_label.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Metrics Frame (for Epoch, MSE, FID, LPIPS) with minimal spacing.
        self.metrics_frame = tk.Frame(root)
        self.metrics_frame.grid(row=1, column=0, columnspan=4, padx=2, pady=2)
        
        # Reorder the metrics so that "Epoch" is on the left.
        self.epoch_label = Label(self.metrics_frame, text="Epoch: ", font=("Arial", 14))
        self.epoch_label.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.mse_label = Label(self.metrics_frame, text="MSE: ", font=("Arial", 14))
        self.mse_label.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.fid_label = Label(self.metrics_frame, text="FID: ", font=("Arial", 14))
        self.fid_label.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.lpips_label = Label(self.metrics_frame, text="LPIPS: ", font=("Arial", 14))
        self.lpips_label.pack(side=tk.LEFT, padx=2, pady=2)

        # Navigation Frame for controls (Row 2)
        self.nav_frame = tk.Frame(root)
        self.nav_frame.grid(row=2, column=0, columnspan=4, padx=2, pady=2)

        self.prev_button = Button(self.nav_frame, text="Previous", font=("Arial", 14), width=10, command=self.prev_epoch)
        self.prev_button.pack(side=tk.LEFT, padx=1, pady=1)

        self.search_entry = Entry(self.nav_frame, font=("Arial", 14), width=8)
        self.search_entry.pack(side=tk.LEFT, padx=1, pady=1)

        self.next_button = Button(self.nav_frame, text="Next", font=("Arial", 14), width=10, command=self.next_epoch)
        self.next_button.pack(side=tk.LEFT, padx=1, pady=1)

        self.search_button = Button(self.nav_frame, text="Go", font=("Arial", 14), width=6, command=self.search_epoch)
        self.search_button.pack(side=tk.LEFT, padx=1, pady=1)

        # Graph Figures arranged horizontally in row 3:
        # MSE plot (left)
        self.fig_mse, self.ax_mse = plt.subplots(figsize=(4, 3))
        self.canvas_mse = FigureCanvasTkAgg(self.fig_mse, master=root)
        self.canvas_mse.get_tk_widget().grid(row=3, column=0, padx=10, pady=10)

        # FID plot (center)
        self.fig_fid, self.ax_fid = plt.subplots(figsize=(4, 3))
        self.canvas_fid = FigureCanvasTkAgg(self.fig_fid, master=root)
        self.canvas_fid.get_tk_widget().grid(row=3, column=1, padx=10, pady=10)

        # LPIPS plot (right)
        self.fig_lpips, self.ax_lpips = plt.subplots(figsize=(4, 3))
        self.canvas_lpips = FigureCanvasTkAgg(self.fig_lpips, master=root)
        self.canvas_lpips.get_tk_widget().grid(row=3, column=2, padx=10, pady=10)

        # Training control buttons (Row 4)
        self.run_button = Button(root, text="Start Training", font=("Arial", 14), width=15, command=self.run_training)
        self.run_button.grid(row=4, column=0, padx=10, pady=10, sticky="w")

        self.resume_button = Button(root, text="Resume Training", font=("Arial", 14), width=15, command=self.resume_training)
        self.resume_button.grid(row=4, column=1, padx=10, pady=10, sticky="w")

        # Refresh Viewer Button (Row 5)
        self.refresh_button = Button(root, text="Refresh Viewer", font=("Arial", 14), width=20, command=self.refresh_viewer)
        self.refresh_button.grid(row=5, column=0, columnspan=4, padx=10, pady=10)

        # Load training logs from JSON training_log file.
        self.epochs = self.load_epochs()
        self.current_epoch_idx = len(self.epochs) - 1 if self.epochs else -1

        # Initial display update and plotting.
        self.update_display()
        self.plot_loss_curves()

    def load_epochs(self):
        """Load training epochs from log file."""
        if not os.path.exists(LOG_FILE):
            return []
        epochs = []
        with open(LOG_FILE, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    epochs.append(data)
        return epochs

    def update_display(self):
        """Update the UI with the metrics and image for the current epoch."""
        if not self.epochs or self.current_epoch_idx < 0:
            self.mse_label.config(text="No training logs found!")
            self.epoch_label.config(text="")
            self.fid_label.config(text="FID: N/A")
            self.lpips_label.config(text="LPIPS: N/A")
            self.image_label.config(image="")
            return

        epoch_data = self.epochs[self.current_epoch_idx]
        img_path = os.path.join(RESULTS_DIR, epoch_data["image"])

        self.epoch_label.config(text=f"Epoch: {epoch_data['epoch']}")
        self.mse_label.config(text=f"MSE: {epoch_data['mse']:.6f}")
        self.fid_label.config(text=f"FID: {epoch_data['fid']:.2f}" if "fid" in epoch_data else "FID: N/A")
        self.lpips_label.config(text=f"LPIPS: {epoch_data['lpips']:.3f}" if "lpips" in epoch_data else "LPIPS: N/A")

        # Load and display image, if available.
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).resize((512, 64))
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk  # Prevent garbage collection
            except Exception as e:
                self.image_label.config(text="Error loading image")
        else:
            self.image_label.config(text="Image not found")

    def next_epoch(self):
        if self.current_epoch_idx < len(self.epochs) - 1:
            self.current_epoch_idx += 1
            self.update_display()

    def prev_epoch(self):
        if self.current_epoch_idx > 0:
            self.current_epoch_idx -= 1
            self.update_display()

    def search_epoch(self):
        """Jump to a specific epoch using the search bar."""
        try:
            target_epoch = int(self.search_entry.get())
            epoch_indices = {data["epoch"]: idx for idx, data in enumerate(self.epochs)}
            if target_epoch in epoch_indices:
                self.current_epoch_idx = epoch_indices[target_epoch]
                self.update_display()
            else:
                self.epoch_label.config(text="Epoch not found!")
        except ValueError:
            self.epoch_label.config(text="Invalid epoch number!")

    def plot_loss_curves(self):
        """Plot MSE, FID, and LPIPS curves in horizontal subplots."""
        if not self.epochs:
            return
        
        # Get lists for each metric.
        epochs_list = [data["epoch"] for data in self.epochs]
        mse_values = [data["mse"] for data in self.epochs]
        fid_values = [data.get("fid", None) for data in self.epochs]
        lpips_values = [data.get("lpips", None) for data in self.epochs]

        # Plot MSE Loss Curve
        self.ax_mse.clear()
        self.ax_mse.plot(epochs_list, mse_values, marker="o", linestyle="-", color="b", label="MSE Loss")
        self.ax_mse.set_xlabel("Epoch")
        self.ax_mse.set_ylabel("MSE Loss")
        self.ax_mse.set_title("MSE Loss Curve")
        self.ax_mse.legend()
        self.fig_mse.tight_layout()
        self.canvas_mse.draw()

        # Plot FID Curve
        self.ax_fid.clear()
        self.ax_fid.plot(epochs_list, fid_values, marker="o", linestyle="-", color="r", label="FID")
        self.ax_fid.set_xlabel("Epoch")
        self.ax_fid.set_ylabel("FID Score")
        self.ax_fid.set_title("FID Curve")
        self.ax_fid.legend()
        self.fig_fid.tight_layout()
        self.canvas_fid.draw()

        # Plot LPIPS Curve
        self.ax_lpips.clear()
        self.ax_lpips.plot(epochs_list, lpips_values, marker="o", linestyle="-", color="g", label="LPIPS")
        self.ax_lpips.set_xlabel("Epoch")
        self.ax_lpips.set_ylabel("LPIPS")
        self.ax_lpips.set_title("LPIPS Curve")
        self.ax_lpips.legend()
        self.fig_lpips.tight_layout()
        self.canvas_lpips.draw()

    def run_training(self):
        """Starts training from scratch. (You can hook in your training script here.)"""
        if not messagebox.askyesno("Confirm Training", "Do you want to start training from scratch?"):
            return
        progress_window, progress_bar = self.show_progress_window("Training from Scratch")
        threading.Thread(target=self._run_training_thread, args=(False, progress_window, progress_bar), daemon=True).start()

    def resume_training(self):
        """Resumes training from a checkpoint."""
        if not messagebox.askyesno("Confirm Training", "Do you want to resume training from the last checkpoint?"):
            return
        progress_window, progress_bar = self.show_progress_window("Resuming Training")
        threading.Thread(target=self._run_training_thread, args=(True, progress_window, progress_bar), daemon=True).start()

    def show_progress_window(self, title):
        """Creates and returns a progress window with an indeterminate progress bar."""
        progress_window = tk.Toplevel(self.root)
        progress_window.title(title)
        progress_window.geometry("350x100")
        label = tk.Label(progress_window, text="Training in progress, please wait...", font=("Arial", 14))
        label.pack(padx=10, pady=10)
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", mode="indeterminate", length=300)
        progress_bar.pack(padx=10, pady=10)
        progress_bar.start(10)
        return progress_window, progress_bar

    def _run_training_thread(self, resume: bool, progress_window, progress_bar):
        """Thread for running the training process."""
        try:
            import sys
            command = [sys.executable, "ddpm.py"]
            if resume:
                command.append("--resume")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            for line in process.stdout:
                print(line.strip())
            for err_line in process.stderr:
                print("ERROR:", err_line.strip())
            process.stdout.close()
            process.stderr.close()
            process.wait()
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
        finally:
            self.root.after(0, lambda: progress_bar.stop())
            self.root.after(0, lambda: progress_window.destroy())
            self.root.after(0, self.refresh_viewer)

    def refresh_viewer(self):
        """Refresh the training log and update displayed metrics and plots."""
        self.epochs = self.load_epochs()
        self.current_epoch_idx = len(self.epochs) - 1 if self.epochs else -1
        self.update_display()
        self.plot_loss_curves()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingViewer(root)
    root.mainloop()
