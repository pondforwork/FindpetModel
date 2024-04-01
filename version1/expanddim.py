import numpy as np

# Suppose we have grayscale image data with shape (height, width)
# Here, let's create a sample image with dimensions 3x3
img_data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

# Before expansion, the shape of img_data is (3, 3)
print("Before expansion:")
print("Shape of img_data:", img_data.shape)

# Now, let's expand the dimensions to make it suitable for a batch
expanded_img_data = np.expand_dims(img_data, axis=0)

# After expansion, the shape of img_data is (1, 3, 3)
print("\nAfter expansion:")
print("Shape of expanded_img_data:", expanded_img_data.shape)
print("Expanded img_data:")
print(expanded_img_data)
