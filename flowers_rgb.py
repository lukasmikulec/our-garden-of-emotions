# For working with folders
import os

# For working with arrays
import numpy as np

# For randomization
import random

# For visualization
import matplotlib.pyplot as plt

# Import K neighbors method from scikit learn
from sklearn.neighbors import KNeighborsRegressor

# For working with images
from PIL import Image

# Define which flowers you want to generate ("sunflowers"/"hyacinths"/"roses")
flower_type = "sunflowers"

# Define pathways to train and test files
flower_data_train = os.listdir(f"flowers_k_neighbors/{flower_type}/train")
flower_data_test = os.listdir(f"flowers_k_neighbors/{flower_type}/test")

# Create a list into which training data will be put
training_data = []

# For all train data of a particular flower
for file in flower_data_train:
    # Specify the file path
    img_path = os.path.join(f"flowers_k_neighbors/{flower_type}/train", file)
    # Load the image, resize it, and ensure it has 3 RGB channels
    img = Image.open(img_path)
    img = img.resize((64, 64))
    img = img.convert('RGB')
    # Turn the image into array (2D)
    img = np.asarray(img)
    # Make the 2D array 1D array
    img = img.flatten()
    # Append the array to the training data list
    training_data.append(img)

# Reshuffle the training data list to avoid bias when training the data
random.shuffle(training_data)

# Turn the data into an array
data = np.asarray(training_data)

# Define how many pixels each image has (64x64 = 4096)
n_pixels = 4096

# Get the upper half of the flowers
X_train = data[:, : (n_pixels + 1) // 2]
# Get the lower half of the flowers
y_train = data[:, n_pixels // 2:]

# Create an estimator object
estimator = KNeighborsRegressor(n_neighbors=2)

# Train the estimator so that it learns a model of the data
estimator.fit(X_train, y_train)

# Create a list into which testing data will be put
testing_data = []

# For all testing data of a particular flower
for file in flower_data_test:
    # Specify the file path
    img_path = os.path.join(f"flowers_k_neighbors/{flower_type}/test", file)
    # Load the image, resize it, and ensure it has 3 RGB channels
    img = Image.open(img_path)
    img = img.resize((64, 64))
    img = img.convert('RGB')
    # Turn the image into array (2D)
    img = np.asarray(img)
    # Make the 2D array 1D array
    img = img.flatten()
    # Append the array to the data list
    testing_data.append(img)

# Reshuffle the data list
random.shuffle(testing_data)

# Turn the data into an array
input_data = np.asarray(testing_data)

# Get the upper part of the flowers
X_test = input_data[:, : (n_pixels + 1) // 2]
# Get the lower part of the flowers
y_test = input_data[:, n_pixels // 2:]

# Predict the lower part of the flowers based on the upper part
y_test_predict = estimator.predict(X_test)

# How many flowers to display
n_flowers = 5

# Plot the completed flowers
image_shape = (64, 64, 3)

n_cols = 2

plt.figure(figsize=(2.0 * n_cols, 2.26 * n_flowers))

for i in range(n_flowers):
    true_flower = np.hstack((X_test[i], y_test[i]))

    # Only show the title at the top of the column, not above each image
    if i:
        sub = plt.subplot(n_flowers, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_flowers, n_cols, i * n_cols + 1, title=f"Original {flower_type}")

    sub.axis("off")
    sub.imshow(
        true_flower.reshape(image_shape), interpolation="nearest"
    )

    completed_flower = np.hstack((X_test[i], y_test_predict[i]))
    # Only show the title at the top of the column, not above each image
    if i:
        sub = plt.subplot(n_flowers, n_cols, i * n_cols + 2)
    else:
        sub = plt.subplot(n_flowers, n_cols, i * n_cols + 2,  title=f"Generated {flower_type}")

    sub.axis("off")
    sub.imshow(
        (completed_flower.reshape(image_shape)/255),
        interpolation="nearest",
    )
# Remove redundant space in the figure
plt.tight_layout()

# Show the figure with original and generated flowers
plt.show()
