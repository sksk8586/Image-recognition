"""
CIFAR-10 Image Classifier Desktop Application
A user-friendly desktop interface for image classification
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import os
import threading

class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.class_names = ["Airplane", "Car", "Bird", "Cat", "Deer",
                           "Dog", "Frog", "Horse", "Ship", "Truck"]
        self.class_emojis = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐕", "🐸", "🐎", "🚢", "🚛"]

        self.setup_gui()
        self.load_model_async()

    def setup_gui(self):
        """Initialize the GUI components"""
        self.root.title("CIFAR-10 Image Classifier")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        # Main title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=20)

        title_label = tk.Label(
            title_frame,
            text="🖼️ CIFAR-10 Image Classifier",
            font=('Arial', 24, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack()

        subtitle_label = tk.Label(
            title_frame,
            text="Upload an image to classify it into one of 10 categories",
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        subtitle_label.pack(pady=(5, 0))

        # Upload section
        upload_frame = tk.Frame(self.root, bg='#f0f0f0')
        upload_frame.pack(pady=20)

        self.upload_btn = tk.Button(
            upload_frame,
            text="📁 Select Image",
            font=('Arial', 14, 'bold'),
            bg='#3498db',
            fg='white',
            padx=30,
            pady=15,
            cursor='hand2',
            command=self.upload_image
        )
        self.upload_btn.pack()

        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready to classify images",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#27ae60'
        )
        self.status_label.pack(pady=5)

        # Image display frame
        self.image_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.image_frame.pack(pady=20)

        self.image_label = tk.Label(
            self.image_frame,
            text="No image selected",
            font=('Arial', 12),
            bg='#ecf0f1',
            fg='#7f8c8d',
            width=40,
            height=15,
            relief='sunken',
            bd=2
        )
        self.image_label.pack()

        # Results frame
        self.results_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.results_frame.pack(pady=20, fill='x', padx=50)

        # Prediction result
        self.prediction_frame = tk.Frame(self.results_frame, bg='#2c3e50', relief='raised', bd=2)

        self.prediction_title = tk.Label(
            self.prediction_frame,
            text="Prediction Result",
            font=('Arial', 14, 'bold'),
            bg='#2c3e50',
            fg='white'
        )

        self.prediction_result = tk.Label(
            self.prediction_frame,
            text="--",
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='#ecf0f1'
        )

        self.confidence_label = tk.Label(
            self.prediction_frame,
            text="Confidence: --%",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#bdc3c7'
        )

        # Progress bar for confidence
        self.confidence_var = tk.DoubleVar()
        self.confidence_bar = ttk.Progressbar(
            self.prediction_frame,
            variable=self.confidence_var,
            maximum=100,
            length=300,
            mode='determinate'
        )

        # All predictions frame
        self.all_predictions_frame = tk.Frame(self.results_frame, bg='#f0f0f0')

        # Initially hide results (only try to hide elements that exist)
        self.results_visible = False

        # Progress bar for loading
        self.progress_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.progress_label = tk.Label(
            self.progress_frame,
            text="Processing...",
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#e67e22'
        )

        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate',
            length=300
        )

        # Reset button (create but don't pack initially)
        self.reset_btn = tk.Button(
            self.root,
            text="🔄 Upload Another Image",
            font=('Arial', 12),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            command=self.reset_interface
        )

    def load_model_async(self):
        """Load the model in a separate thread"""
        def load_model():
            try:
                model_path = '../models/imageClassifier.keras'
                if os.path.exists(model_path):
                    self.model = tf.keras.models.load_model(model_path)
                    self.root.after(0, lambda: self.status_label.config(
                        text="Model loaded successfully ✓", fg='#27ae60'))
                else:
                    self.root.after(0, lambda: self.status_label.config(
                        text="⚠️ Model file not found. Please train the model first.", fg='#e74c3c'))
                    self.root.after(0, lambda: self.upload_btn.config(state='disabled'))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Error loading model: {str(e)}", fg='#e74c3c'))
                self.root.after(0, lambda: self.upload_btn.config(state='disabled'))

        threading.Thread(target=load_model, daemon=True).start()

    def upload_image(self):
        """Handle image upload"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded yet. Please wait...")
            return

        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        """Process and classify the uploaded image"""
        try:
            # Show loading
            self.show_loading()

            # Process in separate thread to avoid freezing GUI
            def classify():
                try:
                    # Load and preprocess image
                    img = cv2.imread(file_path)
                    if img is None:
                        raise ValueError("Could not load image")

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (32, 32))

                    # Display image
                    self.root.after(0, lambda: self.display_image(img_rgb))

                    # Make prediction
                    img_array = np.array([img_resized]) / 255.0
                    predictions = self.model.predict(img_array, verbose=0)

                    # Update GUI with results
                    self.root.after(0, lambda: self.display_results(predictions[0]))

                except Exception as e:
                    self.root.after(0, lambda: self.show_error(f"Error processing image: {str(e)}"))
                finally:
                    self.root.after(0, self.hide_loading)

            threading.Thread(target=classify, daemon=True).start()

        except Exception as e:
            self.show_error(f"Error: {str(e)}")
            self.hide_loading()

    def display_image(self, img_array):
        """Display the uploaded image"""
        # Resize image for display (maintain aspect ratio)
        height, width = img_array.shape[:2]
        max_size = 250

        if height > width:
            new_height = max_size
            new_width = int(width * max_size / height)
        else:
            new_width = max_size
            new_height = int(height * max_size / width)

        img_resized = cv2.resize(img_array, (new_width, new_height))

        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(img_resized)
        photo = ImageTk.PhotoImage(pil_image)

        # Update label
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep a reference

    def display_results(self, predictions):
        """Display classification results"""
        # Get top prediction
        top_index = np.argmax(predictions)
        top_confidence = predictions[top_index] * 100

        # Update main prediction display
        self.prediction_result.config(
            text=f"{self.class_emojis[top_index]} {self.class_names[top_index]}"
        )
        self.confidence_label.config(text=f"Confidence: {top_confidence:.1f}%")

        # Animate confidence bar
        self.animate_confidence_bar(top_confidence)

        # Display all predictions
        self.display_all_predictions(predictions)

        # Show results
        self.show_results()

    def animate_confidence_bar(self, target_value):
        """Animate the confidence progress bar"""
        def animate():
            current = self.confidence_var.get()
            if current < target_value:
                current = min(current + 2, target_value)
                self.confidence_var.set(current)
                self.root.after(50, animate)

        self.confidence_var.set(0)
        animate()

    def display_all_predictions(self, predictions):
        """Display all class predictions"""
        # Clear previous results
        for widget in self.all_predictions_frame.winfo_children():
            widget.destroy()

        # Title
        title = tk.Label(
            self.all_predictions_frame,
            text="All Class Probabilities",
            font=('Arial', 14, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title.pack(pady=(20, 10))

        # Create grid for all predictions
        grid_frame = tk.Frame(self.all_predictions_frame, bg='#f0f0f0')
        grid_frame.pack()

        # Sort predictions by confidence
        sorted_indices = np.argsort(predictions)[::-1]

        for i, idx in enumerate(sorted_indices):
            row = i // 2
            col = i % 2

            class_frame = tk.Frame(grid_frame, bg='#ecf0f1', relief='raised', bd=1)
            class_frame.grid(row=row, column=col, padx=5, pady=3, sticky='ew')

            class_text = f"{self.class_emojis[idx]} {self.class_names[idx]}"
            confidence = predictions[idx] * 100

            class_label = tk.Label(
                class_frame,
                text=class_text,
                font=('Arial', 10, 'bold'),
                bg='#ecf0f1',
                fg='#2c3e50'
            )
            class_label.pack(pady=2)

            conf_label = tk.Label(
                class_frame,
                text=f"{confidence:.1f}%",
                font=('Arial', 9),
                bg='#ecf0f1',
                fg='#7f8c8d'
            )
            conf_label.pack(pady=(0, 5))

        # Configure grid weights
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

    def show_loading(self):
        """Show loading animation"""
        self.progress_frame.pack(pady=20)
        self.progress_label.pack()
        self.progress_bar.pack(pady=10)
        self.progress_bar.start(10)

        self.upload_btn.config(state='disabled')
        self.status_label.config(text="Processing image...", fg='#e67e22')

    def hide_loading(self):
        """Hide loading animation"""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.upload_btn.config(state='normal')

    def show_results(self):
        """Show the results section"""
        self.prediction_frame.pack(fill='x', pady=10)
        self.prediction_title.pack(pady=10)
        self.prediction_result.pack(pady=5)
        self.confidence_label.pack(pady=5)
        self.confidence_bar.pack(pady=10)

        self.all_predictions_frame.pack(fill='x', pady=10)
        self.reset_btn.pack(pady=20)
        self.results_visible = True

    def hide_results(self):
        """Hide the results section"""
        if hasattr(self, 'prediction_frame'):
            self.prediction_frame.pack_forget()
        if hasattr(self, 'all_predictions_frame'):
            self.all_predictions_frame.pack_forget()
        if hasattr(self, 'reset_btn'):
            self.reset_btn.pack_forget()
        self.results_visible = False

    def reset_interface(self):
        """Reset the interface for a new image"""
        if self.results_visible:
            self.hide_results()

        self.image_label.config(
            image='',
            text="No image selected"
        )
        self.image_label.image = None

        if hasattr(self, 'confidence_var'):
            self.confidence_var.set(0)
        self.status_label.config(text="Ready to classify images", fg='#27ae60')

    def show_error(self, message):
        """Show error message"""
        messagebox.showerror("Error", message)
        self.status_label.config(text="Error occurred", fg='#e74c3c')

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = ImageClassifierGUI(root)

    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    root.mainloop()

if __name__ == "__main__":
    main()


