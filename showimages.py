import cv2
import tkinter as tk
from PIL import Image, ImageTk


def show_images():
    # Create a frame to hold the labels for the images
    frame = tk.Frame(root)
    frame.pack()

    # Load and display each image
    for i, image_path in enumerate(image_paths):
        # Load the image with OpenCV
        img = cv2.imread(image_path)
        # Convert the image from OpenCV format to PIL format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        # Resize the image to fit the window size (optional)
        img_pil = img_pil.resize((128, 128))  # Adjust the size as needed

        # Create a PhotoImage object from the PIL image
        img_tk = ImageTk.PhotoImage(img_pil)

        # Create a label widget to display the image
        label = tk.Label(frame, image=img_tk)
        label.grid(row=i // 5, column=i % 5)  # Display 5 images per row
        label.image = img_tk  # Keep a reference to avoid garbage collection

    # Create a scrollbar for the frame
    scrollbar = tk.Scrollbar(root, orient="vertical", command=frame.yview)
    scrollbar.pack(side="right", fill="y")
    frame.configure(yscrollcommand=scrollbar.set)


# List of image paths (replace with your image paths)
image_paths = [
    "blackcat.jpeg",
    "blackcat.jpeg",
    "blackcat.jpeg",
    "blackcat.jpeg",
    "blackcat.jpeg",
]

# Create the main application window
root = tk.Tk()
root.title("Display Images")

# Button to display the images
show_button = tk.Button(root, text="Show Images", command=show_images)
show_button.pack()

# Run the Tkinter event loop
root.mainloop()
