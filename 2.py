import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageTk


def browse_file():
    filename = filedialog.askopenfilename()
    if filename:
        path_var.set(filename)


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()


def calculate_similarity(img_features1, img_features2):
    return cosine_similarity([img_features1], [img_features2])[0][0]


def search():
    user_input_path = path_var.get()
    user_input_features = extract_features(user_input_path, model)

    pet_data_list = []
    for pet_name in os.listdir(data_dir):
        if pet_name == ".DS_Store":
            continue
        pet_name_dir = os.path.join(data_dir, pet_name)
        max_similarity = -1
        max_img_path = ""
        for img_file in os.listdir(pet_name_dir):
            img_path = os.path.join(pet_name_dir, img_file)
            features = extract_features(img_path, model)
            similarity = calculate_similarity(features, user_input_features)
            if similarity > max_similarity:
                max_similarity = similarity
                max_img_path = img_path
        pet_data_list.append(
            {
                "image_path": max_img_path,
                "pet_name": pet_name,
                "similarity": max_similarity,
            }
        )

    pet_data = pd.DataFrame(pet_data_list)
    top_5_recommended_pets = pet_data.nlargest(5, "similarity")
    print("Top 5 recommended pets:")
    print(top_5_recommended_pets[["image_path", "pet_name", "similarity"]])
    show_images(top_5_recommended_pets)


def show_images(top_5_recommended_pets):
    # Create a frame to hold the labels for the images
    frame = tk.Frame(root)
    frame.pack()

    # Load and display each image
    for i, (_, row) in enumerate(top_5_recommended_pets.iterrows()):
        # Load the image with OpenCV
        img = cv2.imread(row["image_path"])
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

        # Create a label widget to display pet name and similarity percentage
        pet_label = tk.Label(
            frame,
            text=f"{row['pet_name']} ({row['similarity']:.2%} match)",
            font=("Arial", 10),
        )
        pet_label.grid(row=(i // 5) + 1, column=i % 5)  # Display below the image

    # Create a scrollbar for the frame
    scrollbar = tk.Scrollbar(root, orient="vertical", command=frame.yview)
    scrollbar.pack(side="right", fill="y")
    frame.configure(yscrollcommand=scrollbar.set)


# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Path to the directory containing pet images
data_dir = "dataset"

# Create the main application window
root = tk.Tk()
root.title("ค้นหาสัตว์เลี้ยง")
root.geometry("1280x900")  # Set the window size to 1280x720

# Create a frame to hold other widgets
frame = tk.Frame(root)
frame.pack(padx=200, pady=200)

# Create some widgets
label = tk.Label(frame, text="เลือกรูปสัตว์เลี้ยงที่ค้องการ", font=("Arial", 24))
label.pack(pady=10)

# File picker button and text field
file_button = tk.Button(frame, text="Browse", command=browse_file)
file_button.pack(pady=10)

path_var = tk.StringVar()
path_entry = tk.Entry(frame, textvariable=path_var, width=50)
path_entry.pack(pady=10)

searchbtn = tk.Button(frame, text="ค้นหา", command=search)
searchbtn.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
