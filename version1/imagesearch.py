import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

# Load pre-trained ResNet model
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


# Function to extract features from an image using a pre-trained ResNet model
def extract_features(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image '{image_path}'")
            return None

        # Continue with feature extraction
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        features = model.predict(image)
        return features.flatten()
    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")
        return None


# Function to search for images containing cats based on text query
def search_images_with_text(query_text, folder_path):
    query_features = extract_features(
        query_text
    )  # Extract features from the query text
    if query_features is None:
        return

    image_files = os.listdir(folder_path)
    matched_images = []

    for file in image_files:
        image_path = os.path.join(folder_path, file)
        features = extract_features(image_path)
        if features is not None:
            similarity = cosine_similarity([query_features], [features])[0][0]
            matched_images.append((file, similarity))

    matched_images.sort(key=lambda x: x[1], reverse=True)
    return matched_images


# Example usage
if __name__ == "__main__":
    query_text = "cat"  # Text query to search for images containing cats
    folder_path = "allcat"  # Path to the folder containing images

    matched_images = search_images_with_text(query_text, folder_path)
    if matched_images:
        print("Top matched images:")
        for image_name, _ in matched_images[:5]:  # Display top 5 matched images
            print(image_name)
    else:
        print("No matching images found.")
