import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import torch
from torchvision import transforms
import torch.nn.functional as F

class Gui:
    def __init__(self, model, device, class_labels):
        self.img_paths = []
        self.current_index = 0
        self.results = []

        self.model = model
        self.device = device
        self.class_labels = class_labels  # For mapping classes after prediction

        self._setup_gui()

    def _setup_gui(self):  # Setting up GUI foundation
        self.root = tk.Tk()
        self.root.title("Object Recognition with CIFAR-10")
        self.root.geometry("500x700")
        self.root.configure(bg="#f0f0f0")
        self._setup_widgets()

    def _create_button(self, parent, text, command, bg, activebg):  # Function to create a button
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg="white",
            font=("Arial", 12, "bold"),
            activebackground=activebg,
            activeforeground="white",
            relief="raised",
            bd=2,
        )

    def _setup_widgets(self):  # Sets up Select Folder and Navigation buttons
        self.select_button = self._create_button(self.root, "Select Folder", self.select_folder, "#007BFF", "#0056b3")
        self.select_button.pack(pady=10)

        self.img_label = tk.Label(self.root, bg="#f0f0f0", relief="groove", bd=2)
        self.img_label.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 14), bg="#f0f0f0", fg="#333", wraplength=450)
        self.result_label.pack(pady=10)

        # Navigation buttons
        self.prev_button = self._create_button(self.root, "← Previous", self.previous_image, "#6c757d", "#5a6268")
        self.prev_button.pack(side=tk.LEFT, padx=20, pady=20)

        self.next_button = self._create_button(self.root, "Next →", self.next_image, "#6c757d", "#5a6268")
        self.next_button.pack(side=tk.RIGHT, padx=20, pady=20)

    def select_folder(self):  # Function to select a folder and process images
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.img_paths = [
                os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
            ]
            if not self.img_paths:
                self.result_label.config(text="No valid images found in the selected folder.", fg="red")
                return

            self.current_index = 0
            self.results = []
            self.classify_images()
            self.display_current_image()

    def classify_images(self):  # Classify all images in the folder
        self.model.to(self.device)
        self.model.eval()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])

        for img_path in self.img_paths:
            try:
                image = Image.open(img_path)
                image = transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(image)
                    probabilities = F.softmax(output, dim=1)
                    top_probs, top_indices = probabilities.topk(3)  # Get top 3 predictions
                    top_predictions = [
                        (self.class_labels[idx.item()], prob.item() * 100) 
                        for idx, prob in zip(top_indices[0], top_probs[0])
                    ]
                self.results.append(top_predictions)
            except Exception as e:
                self.results.append([("Error", 0)])

    def display_current_image(self):  # Display the current image and its result
        if not self.img_paths:
            self.result_label.config(text="No images to display.", fg="red")
            return

        img_path = self.img_paths[self.current_index]
        img = Image.open(img_path).resize((400, 400))  # Resize for display
        photo = ImageTk.PhotoImage(img)

        self.img_label.config(image=photo)
        self.img_label.image = photo

        result_text = f"{os.path.basename(img_path)}:\n"
        result_text += "\n".join([f"{label}: {prob:.2f}%" for label, prob in self.results[self.current_index]])
        self.result_label.config(text=result_text, fg="#333")

    def previous_image(self):  # Navigate to the previous image
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()

    def next_image(self):  # Navigate to the next image
        if self.current_index < len(self.img_paths) - 1:
            self.current_index += 1
            self.display_current_image()

    def open(self):  # Open the GUI
        self.root.mainloop()
