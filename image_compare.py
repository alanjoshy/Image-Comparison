import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

class ImageComparisonApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Image Comparison Tool")
        self.window.geometry("600x400")
        
        self.image1_path = None
        self.image2_path = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Image 1 selection
        self.img1_btn = tk.Button(self.window, text="Select First Image", command=lambda: self.select_image(1))
        self.img1_btn.pack(pady=10)
        self.img1_label = tk.Label(self.window, text="No image selected")
        self.img1_label.pack()
        
        # Image 2 selection
        self.img2_btn = tk.Button(self.window, text="Select Second Image", command=lambda: self.select_image(2))
        self.img2_btn.pack(pady=10)
        self.img2_label = tk.Label(self.window, text="No image selected")
        self.img2_label.pack()
        
        # Compare button
        self.compare_btn = tk.Button(self.window, text="Compare Images", command=self.compare_images)
        self.compare_btn.pack(pady=20)
        
        # Result label
        self.result_label = tk.Label(self.window, text="")
        self.result_label.pack(pady=10)
        
    def select_image(self, img_num):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            if img_num == 1:
                self.image1_path = file_path
                self.img1_label.config(text=f"Image 1: {Path(file_path).name}")
            else:
                self.image2_path = file_path
                self.img2_label.config(text=f"Image 2: {Path(file_path).name}")
    
    def compare_images(self):
        if not self.image1_path or not self.image2_path:
            messagebox.showerror("Error", "Please select both images first!")
            return
        
        try:
            # Read images
            img1 = cv2.imread(self.image1_path)
            img2 = cv2.imread(self.image2_path)
            
            # Convert to same size
            img1 = cv2.resize(img1, (500, 500))
            img2 = cv2.resize(img2, (500, 500))
            
            # Convert images to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Calculate structural similarity index
            score = cv2.compareHist(
                cv2.calcHist([gray1], [0], None, [256], [0, 256]),
                cv2.calcHist([gray2], [0], None, [256], [0, 256]),
                cv2.HISTCMP_CORREL
            )
            
            # Convert score to percentage
            similarity = (score + 1) * 50  # Convert from [-1,1] to [0,100]
            similarity = max(0, min(100, similarity))  # Clamp between 0 and 100
            
            self.result_label.config(
                text=f"Similarity: {similarity:.2f}%",
                font=("Arial", 14, "bold")
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ImageComparisonApp()
    app.run()
