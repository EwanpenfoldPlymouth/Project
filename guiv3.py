import os
import json
import tkinter as tk
from tkinter import Label, Button, Entry, messagebox
from tkinter import ttk  # Used for the progress bar
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import subprocess

# Paths
RUN_NAME = "DDPM_Unconditional"
RESULTS_DIR = os.path.join("results", RUN_NAME)
LOG_FILE = os.path.join(RESULTS_DIR, "training_log.json")

class TrainingViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("DDPM Training Results")

        # UI Components for viewing training logs and images
        self.image_label = Label(root)
        self.image_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.mse_label = Label(root, text="MSE: ", font=("Arial", 14))
        self.mse_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        # New FID Label
        self.fid_label = Label(root, text="FID: ", font=("Arial", 14))
        self.fid_label.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.epoch_label = Label(root, text="Epoch: ", font=("Arial", 14))
        self.epoch_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.prev_button = Button(root, text="Previous", command=self.prev_epoch)
        self.prev_button.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        self.next_button = Button(root, text="Next", command=self.next_epoch)
        self.next_button.grid(row=3, column=1, padx=10, pady=10, sticky="e")

        # Search Bar to jump to a specific epoch
        self.search_entry = Entry(root)
        self.search_entry.grid(row=4, column=0, padx=10, pady=10)
        self.search_button = Button(root, text="Go", command=self.search_epoch)
        self.search_button.grid(row=4, column=1, padx=10, pady=10)

        # Loss Curve Figure
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        # Training Control Buttons
        self.run_button = Button(root, text="Start Training", command=self.run_training)
        self.run_button.grid(row=6, column=0, padx=10, pady=10, sticky="w")

        self.resume_button = Button(root, text="Resume Training", command=self.resume_training)
        self.resume_button.grid(row=6, column=1, padx=10, pady=10, sticky="e")

        # Refresh Viewer Button
        self.refresh_button = Button(root, text="Refresh Viewer", command=self.refresh_viewer)
        self.refresh_button.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

        # Load training logs
        self.epochs = self.load_epochs()
        self.current_epoch_idx = len(self.epochs) - 1 if self.epochs else -1  # Start at latest epoch if available

        # Initial Display
        self.update_display()
        self.plot_loss_curve()

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
        """Update the UI with the current epoch's details."""
        if not self.epochs or self.current_epoch_idx < 0:
            self.mse_label.config(text="No training logs found!")
            self.epoch_label.config(text="")
            self.fid_label.config(text="FID: N/A")
            self.image_label.config(image="")
            return

        epoch_data = self.epochs[self.current_epoch_idx]
        img_path = os.path.join(RESULTS_DIR, epoch_data["image"])

        self.mse_label.config(text=f"MSE: {epoch_data['mse']:.6f}")
        self.epoch_label.config(text=f"Epoch: {epoch_data['epoch']}")

        # Update FID label if available
        if "fid" in epoch_data:
            self.fid_label.config(text=f"FID: {epoch_data['fid']:.2f}")
        else:
            self.fid_label.config(text="FID: N/A")

        # Show single image
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
            target_epoch = int(self.search_entry.get())  # Get user input
            epoch_indices = {data["epoch"]: idx for idx, data in enumerate(self.epochs)}

            if target_epoch in epoch_indices:
                self.current_epoch_idx = epoch_indices[target_epoch]
                self.update_display()
            else:
                self.epoch_label.config(text="Epoch not found!")
        except ValueError:
            self.epoch_label.config(text="Invalid epoch number!")

    def plot_loss_curve(self):
        """Plot MSE loss over epochs."""
        if not self.epochs:
            return
        epochs_list = [data["epoch"] for data in self.epochs]
        mse_values = [data["mse"] for data in self.epochs]

        self.ax.clear()
        self.ax.plot(epochs_list, mse_values, marker="o", linestyle="-", color="b", label="MSE Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("MSE Loss")
        self.ax.set_title("Training Loss Curve")
        self.ax.legend()
        self.canvas.draw()

    def show_progress_window(self, title):
        """Creates and returns a progress window with an indeterminate progress bar."""
        progress_window = tk.Toplevel(self.root)
        progress_window.title(title)
        progress_window.geometry("350x100")
        label = tk.Label(progress_window, text="Training in progress, please wait...")
        label.pack(padx=10, pady=10)
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", mode="indeterminate", length=300)
        progress_bar.pack(padx=10, pady=10)
        progress_bar.start(10)  # Adjust the speed as desired
        return progress_window, progress_bar

    def run_training(self):
        """Initiate training from scratch after confirmation."""
        if not messagebox.askyesno("Confirm Training", "Do you want to start training from scratch?"):
            return
        progress_window, progress_bar = self.show_progress_window("Training from Scratch")
        threading.Thread(target=self._run_training_thread, args=(False, progress_window, progress_bar), daemon=True).start()

    def resume_training(self):
        """Resume training from checkpoint after confirmation."""
        if not messagebox.askyesno("Confirm Training", "Do you want to resume training from the last checkpoint?"):
            return
        progress_window, progress_bar = self.show_progress_window("Resuming Training")
        threading.Thread(target=self._run_training_thread, args=(True, progress_window, progress_bar), daemon=True).start()

    def _run_training_thread(self, resume: bool, progress_window, progress_bar):
        """Thread target for running or resuming training."""
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

            # Print process outputs (optionally, you may update a log area in the GUI)
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
        """Refresh the training log and plots."""
        self.epochs = self.load_epochs()
        self.current_epoch_idx = len(self.epochs) - 1 if self.epochs else -1
        self.update_display()
        self.plot_loss_curve()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingViewer(root)
    root.mainloop()
