import os
import sys
import json
import torch
import timm
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from pathlib import Path


class AnimalClassifierApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Animal Species Classifier")
        self.root.geometry("1200x700")

        # Load model and class mapping
        self.load_model(model_path)

        # Create UI elements
        self.create_widgets()

        # Current prediction values
        self.current_image_path = None
        self.current_predictions = None

    def load_model(self, model_path):
        # Load class mapping
        model_dir = os.path.dirname(model_path)
        with open(os.path.join(model_dir, '..', 'class_mapping.json'), 'r') as f:
            self.idx_to_class = json.load(f)
            # Convert string keys to integers
            self.idx_to_class = {int(k): v for k, v in self.idx_to_class.items()}

        # Get number of classes
        num_classes = len(self.idx_to_class)

        # Initialize model
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Model loaded successfully with {num_classes} classes")

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame for image and buttons
        left_frame = ttk.Frame(main_frame, padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right frame for prediction results
        right_frame = ttk.Frame(main_frame, padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(left_frame, text="MultiZoo Animal Classifier", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Frame for image
        self.image_frame = ttk.LabelFrame(left_frame, text="Image", padding="10")
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Image display area
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Buttons frame
        buttons_frame = ttk.Frame(left_frame, padding="10")
        buttons_frame.pack(fill=tk.X, pady=10)

        # Load image button
        self.load_button = ttk.Button(buttons_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Classify button
        self.classify_button = ttk.Button(buttons_frame, text="Classify", command=self.classify_image,
                                          state=tk.DISABLED)
        self.classify_button.pack(side=tk.LEFT, padx=5)

        # Batch classify button
        self.batch_button = ttk.Button(buttons_frame, text="Batch Classify", command=self.batch_classify)
        self.batch_button.pack(side=tk.LEFT, padx=5)

        # Results title
        results_title = ttk.Label(right_frame, text="Classification Results", font=("Arial", 14, "bold"))
        results_title.pack(pady=10)

        # Top prediction
        self.top_pred_frame = ttk.LabelFrame(right_frame, text="Top Prediction", padding="10")
        self.top_pred_frame.pack(fill=tk.X, pady=10)

        self.top_pred_label = ttk.Label(self.top_pred_frame, text="No prediction yet", font=("Arial", 12))
        self.top_pred_label.pack(pady=5)

        self.confidence_label = ttk.Label(self.top_pred_frame, text="")
        self.confidence_label.pack(pady=5)

        # Confidence bar
        self.confidence_bar = ttk.Progressbar(self.top_pred_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.confidence_bar.pack(pady=5)

        # Create matplotlib figure for top 5 predictions
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Add plot to canvas
        self.canvas_frame = ttk.LabelFrame(right_frame, text="Top 5 Predictions", padding="10")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        """Load an image from file system."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]

        filepath = filedialog.askopenfilename(
            title="Select an image",
            filetypes=filetypes
        )

        if filepath:
            try:
                self.current_image_path = filepath
                self.display_image(filepath)
                self.classify_button.config(state=tk.NORMAL)
                self.status_var.set(f"Loaded image: {os.path.basename(filepath)}")
            except Exception as e:
                self.status_var.set(f"Error loading image: {e}")

    def display_image(self, filepath):
        """Display the image in the UI."""
        # Open and resize image for display
        image = Image.open(filepath)

        # Calculate new dimensions while maintaining aspect ratio
        width, height = image.size
        max_size = 400

        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        # Resize image for display
        display_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Convert to PhotoImage
        self.tk_image = ImageTk.PhotoImage(display_image)

        # Display in the label
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image  # Keep a reference to prevent garbage collection

    def classify_image(self):
        """Classify the current image with the model."""
        if not self.current_image_path:
            self.status_var.set("No image loaded")
            return

        try:
            # Open and preprocess image
            image = Image.open(self.current_image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

            # Forward pass
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

            # Convert to numpy for processing
            probs = probabilities.numpy()

            # Get top 5 predictions
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5_probs = probs[top5_idx]
            top5_classes = [self.idx_to_class.get(str(idx), f"Class {idx}") for idx in top5_idx]

            # Store current predictions
            self.current_predictions = {
                'classes': top5_classes,
                'probabilities': top5_probs,
                'top_class': top5_classes[0],
                'top_prob': top5_probs[0]
            }

            # Update UI with results
            self.update_results_display()

            self.status_var.set(f"Classified as {top5_classes[0]} with {top5_probs[0]:.2%} confidence")

        except Exception as e:
            self.status_var.set(f"Error during classification: {e}")
            import traceback
            traceback.print_exc()

    def update_results_display(self):
        """Update the UI with current prediction results."""
        if not self.current_predictions:
            return

        # Update top prediction display
        top_class = self.current_predictions['top_class']
        top_prob = self.current_predictions['top_prob']

        # Format class name for display (capitalize words, replace underscores with spaces)
        display_class = ' '.join(word.capitalize() for word in top_class.replace('_', ' ').split())

        self.top_pred_label.config(text=f"{display_class}")
        self.confidence_label.config(text=f"Confidence: {top_prob:.2%}")
        self.confidence_bar['value'] = top_prob * 100

        # Update plot
        self.ax.clear()

        # Get classes and probabilities
        classes = [' '.join(word.capitalize() for word in cls.replace('_', ' ').split()) for cls in
                   self.current_predictions['classes']]
        probs = self.current_predictions['probabilities']

        # Create horizontal bar chart
        y_pos = np.arange(len(classes))
        self.ax.barh(y_pos, probs, align='center')
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(classes)
        self.ax.invert_yaxis()  # Labels read top-to-bottom
        self.ax.set_xlabel('Probability')
        self.ax.set_title('Top 5 Predictions')

        # Add percentage labels to bars
        for i, v in enumerate(probs):
            self.ax.text(v + 0.01, i, f'{v:.2%}', va='center')

        self.fig.tight_layout()
        self.canvas.draw()

    def batch_classify(self):
        """Classify a batch of images from a directory."""
        directory = filedialog.askdirectory(title="Select directory with images to classify")

        if not directory:
            return

        # Get all image files in the directory
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

        if not image_files:
            self.status_var.set("No image files found in the selected directory")
            return

        # Create a progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Classification Progress")
        progress_window.geometry("400x150")

        # Add progress information
        ttk.Label(progress_window, text="Processing images...").pack(pady=10)

        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=len(image_files))
        progress_bar.pack(fill=tk.X, padx=20, pady=10)

        status_label = ttk.Label(progress_window, text="")
        status_label.pack(pady=10)

        # Create results directory
        results_dir = os.path.join(directory, "classification_results")
        os.makedirs(results_dir, exist_ok=True)

        # Open results CSV file
        csv_path = os.path.join(results_dir, "batch_results.csv")
        with open(csv_path, 'w') as f:
            f.write("filename,predicted_class,confidence\n")

        # Function to classify a single image
        def classify_single(idx):
            if idx >= len(image_files):
                # All done
                progress_window.destroy()
                self.status_var.set(f"Batch classification complete. Results saved to {csv_path}")
                return

            filename = image_files[idx]
            filepath = os.path.join(directory, filename)

            try:
                # Open and preprocess image
                image = Image.open(filepath).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

                # Get top prediction
                top_idx = torch.argmax(probabilities).item()
                top_prob = probabilities[top_idx].item()
                top_class = self.idx_to_class.get(str(top_idx), f"Class {top_idx}")

                # Append to CSV
                with open(csv_path, 'a') as f:
                    f.write(f"{filename},{top_class},{top_prob:.6f}\n")

                # Update progress
                progress_var.set(idx + 1)
                status_label.config(text=f"Processed {idx + 1}/{len(image_files)}: {filename}")

                # Continue with next image
                self.root.after(10, lambda: classify_single(idx + 1))

            except Exception as e:
                status_label.config(text=f"Error processing {filename}: {e}")
                # Continue with next image
                self.root.after(10, lambda: classify_single(idx + 1))

        # Start processing
        self.root.after(100, lambda: classify_single(0))


def main():
    """Main entry point of the application."""
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Find most recent model directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dirs = [d for d in os.listdir(base_dir) if d.startswith('model_outputs_')]

        if not model_dirs:
            print("No model directories found. Please train a model first or provide a model path.")
            sys.exit(1)

        # Get most recent directory
        most_recent_dir = max(model_dirs)
        model_path = os.path.join(base_dir, most_recent_dir, 'best_model.pth')

        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            sys.exit(1)

    # Create the Tkinter root window
    root = tk.Tk()

    # Set theme
    try:
        # Try to use a modern theme if available
        style = ttk.Style()
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'vista' in available_themes:
            style.theme_use('vista')
    except Exception:
        pass  # Use default theme if others are not available

    # Create and run the application
    app = AnimalClassifierApp(root, model_path)

    # Center the window on screen
    window_width = 1200
    window_height = 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    center_x = int((screen_width - window_width) / 2)
    center_y = int((screen_height - window_height) / 2)

    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    # Run the application
    root.mainloop()


if __name__ == "__main__":
    main()