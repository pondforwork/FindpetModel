import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()


# Function to calculate similarity between two images based on cosine similarity
def calculate_similarity(img_features1, img_features2):
    return cosine_similarity([img_features1], [img_features2])[0][0]

# Path to the directory containing pet images
data_dir = "dataset"

# Load pet images and their paths into a DataFrame
pet_data_list = []
for pet_type in os.listdir(data_dir):
    # Ignore .DS_Store files
    if pet_type == ".DS_Store":
        continue
    pet_type_dir = os.path.join(data_dir, pet_type)
    for img_file in os.listdir(pet_type_dir):
        img_path = os.path.join(pet_type_dir, img_file)
        features = extract_features(img_path, model)
        pet_data_list.append(
            {"image_path": img_path, "pet_type": pet_type, "features": features}
        )


# Create DataFrame from the list of dictionaries
pet_data = pd.DataFrame(pet_data_list)

# Example user input: path to the desired pet image
user_input_path = "newwhitecat.jpg"

# Extract features for the user input image
user_input_features = extract_features(user_input_path, model)

# Calculate similarity between user input image and database images
pet_data["similarity"] = pet_data["features"].apply(
    lambda x: calculate_similarity(x, user_input_features)
)

# Sort database images by similarity and recommend top matches
recommended_pets = pet_data.sort_values(by="similarity", ascending=False).head(5)

print("Top 5 recommended pets:")
print(recommended_pets[["image_path", "pet_type", "similarity"]])
