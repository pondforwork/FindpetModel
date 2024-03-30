import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import cv2


# Function to extract features using a pre-trained CNN model
def extract_features(image_paths, model):
    features = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(
            image, (224, 224)
        )  # Resize image to fit the model's input size
        image = tf.keras.applications.resnet.preprocess_input(image)  # Preprocess image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        feature = model.predict(image).flatten()  # Extract features
        features.append(feature)
    return np.array(features)


# Function to load dataset
def load_dataset(dataset_dir):
    image_paths = []
    labels = []
    for person_dir in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, person_dir)):
            for image_file in os.listdir(os.path.join(dataset_dir, person_dir)):
                image_paths.append(os.path.join(dataset_dir, person_dir, image_file))
                labels.append(
                    person_dir
                )  # Assuming folder names are the labels (pet types)
    return image_paths, labels


# Function to build index for fast similarity search
def build_index(image_paths, model):
    features = extract_features(image_paths, model)
    neighbors = NearestNeighbors(n_neighbors=5, algorithm="auto").fit(features)
    return neighbors


# Function to recommend similar pets based on user input
def recommend_similar_pets(user_input_image_path, neighbors, image_paths, labels):
    # Extract features from user input image
    user_input_features = extract_features([user_input_image_path], model)

    # Find similar images in the dataset
    _, indices = neighbors.kneighbors(user_input_features)

    # Display recommended pets
    print("Recommended pets:")
    for i in range(len(indices[0])):
        print("Similar Pet", i + 1)
        print("Image Path:", image_paths[indices[0][i]])
        print("Pet Type:", labels[indices[0][i]])
        print()


# Example usage
def main(dataset_dir, user_input_image_path):
    image_paths, labels = load_dataset(dataset_dir)
    model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, pooling="avg"
    )
    neighbors = build_index(image_paths, model)
    recommend_similar_pets(user_input_image_path, neighbors, image_paths, labels)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <dataset_dir> <user_input_image_path>")
        sys.exit(1)
    dataset_dir = sys.argv[1]
    user_input_image_path = sys.argv[2]
    main(dataset_dir, user_input_image_path)
