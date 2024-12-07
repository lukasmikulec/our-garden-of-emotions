# For the gui capabilities
import customtkinter as ctk

# For random operations
import random

# For neural networks
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Import PIL to work with images
from PIL import Image, ImageOps, ImageTk


# Define the path to the dataset
image_path = "dataset"


# Create object of the convolutional neural network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define first convolutional layer for input of 1 channel (grayscale)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # Define first non-linear function
        self.relu1 = nn.ReLU()
        # Reduce the dimensionality
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Define second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Define second non-linear function
        self.relu2 = nn.ReLU()
        # Reduce dimensionality
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Apply linear transformation to input data using weights and biases
        self.fc1 = nn.Linear(32 * 12 * 12, 64)
        # Define third non-linear function
        self.relu3 = nn.ReLU()
        # Apply linear transformation to input data using weights and biases
        self.fc2 = nn.Linear(64, 3)

    # Define forward propagation
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 12 * 12)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# Define transformational procedure to be applied to all dataset images before processing
transform = transforms.Compose([
    # Ensure every image is grayscale (1 dimension)
    transforms.Grayscale(),
    # Transform it to tensor
    transforms.ToTensor(),
    # At the end, images are grayscale (1 dimension) and 48x48 px
    transforms.Normalize(mean=[0.485], std=[0.229]),])

# Define dataset object using the pre-made ImageFolder function
dataset = ImageFolder(root=image_path, transform=transform)
print(dataset.classes)
print(dataset.class_to_idx)

# Use 80 percent of data for training
train_size = int(0.8 * len(dataset))
# Use 20 percent of data for training
val_size = len(dataset) - train_size
# Do the splitting of dataset based on defined proportions
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define batch size and load the data accordingly
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize the convolutional neural network
model = SimpleCNN()

# Define which function will compute difference between prediction and actual result
criterion = nn.CrossEntropyLoss()
# Define optimizer which will optimize weights and biases (stochastic gradient descent method)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define number of learning epochs
num_epochs = 5
# Find out if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Assign the process to available device
model.to(device)

# For each epoch
for epoch in range(num_epochs):
    # Switch the model to training mode
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        image, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    # Switch the model to evaluation mode
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = val_corrects.double() / len(val_loader.dataset)

    # Print the accuracy of each epoch
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), "custom_model.pth")
# Define the labels
class_labels = ["angry", "happy", "sad"]


# Define a function which will take in a file and output prediction for it using the trained model
def predict_for_image(file):
    image = Image.open(file)
    image = ImageOps.grayscale(image)
    image = image.resize((48, 48))
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    prediction = class_labels[predicted.item()]

    return prediction


# Define lists for both people
one_person = []
second_person = []

# Predict for 8 images of one person
one_person.append(predict_for_image("angry_1.jpg"))
one_person.append(predict_for_image("happy_1.jpg"))
one_person.append(predict_for_image("happy_2.jpg"))
one_person.append(predict_for_image("happy_3.jpg"))
one_person.append(predict_for_image("angry_3.jpg"))
one_person.append(predict_for_image("sad_3.jpg"))
one_person.append(predict_for_image("happy_5.jpg"))
one_person.append(predict_for_image("happy_6.jpg"))

# Predict for 8 images of second person
second_person.append(predict_for_image("angry_2.jpg"))
second_person.append(predict_for_image("sad_1.jpg"))
second_person.append(predict_for_image("sad_2.jpg"))
second_person.append(predict_for_image("happy_4.jpg"))
second_person.append(predict_for_image("angry_4.jpg"))
second_person.append(predict_for_image("sad_4.jpg"))
second_person.append(predict_for_image("sad_5.jpg"))
second_person.append(predict_for_image("sad_6.jpg"))

print(one_person)
print(second_person)

# Create the GUI window
window = ctk.CTk()

# Name the GUI window
window.title("Our garden of emotions")

# Define the width and height of the interface
width, height = 1920, 1080

# Set the interface window size
window.geometry(f"{width}x{height}")

# Create a canvas on which all elements will be put
canvas = ctk.CTkCanvas(window)
# Put it to the gui window
canvas.pack(fill="both", expand=True)

# Load the image of the grass
grass_file = Image.open("elements/grass.jpg")
# Resize the image of the grass
grass_resized = grass_file.resize((4000, 2000))
# Save it in format for the canvas
grass_image = ImageTk.PhotoImage(grass_resized)
# Put the image of the grass to canvas
canvas.create_image(0, 0, image=grass_image)

# Create variable which will store the size of the lake
lake_size = 0.5

# Determine the size of the lake as the central element
for i in range(7):
    if one_person[i] == "sad" and second_person[i] == "sad":
        lake_size += 0.2

# Load the image of the lake
lake_file = Image.open("elements/lake.png")
# Resize the image of the lake
lake_resized = lake_file.resize((int(lake_size*426), int(lake_size*260)))
# Save it in format for the canvas
lake_image = ImageTk.PhotoImage(lake_resized)
# Put the image of the lake to canvas
canvas.create_image(960, 450, image=lake_image)


# Define a function which will count the number of flowers for combination of person and emotion
def count_number_of_flowers(emotion, person_list):
    # Start with 0 for both flower and emotion count
    flower_count = 0
    emotion_count = 0

    # Check if the first item in list is the emotion
    if person_list[0] == emotion:
        # If yes, add 1 to the emotion's count and add one flower
        emotion_count = 1
        flower_count = 1

    # For the rest of the items (apart from the first)
    for i in range(len(person_list)-1):
        # If the emotion's streak starts
        if person_list[i+1] == emotion and person_list[i] != emotion:
            # Add one to emotion's count
            emotion_count += 1
            # Add so many flowers like the current streak
            flower_count += emotion_count
        # If the emotion's streak continues
        if person_list[i+1] == emotion and person_list[i] == emotion:
            # Add one to emotion's count
            emotion_count += 1
            # Add so many flowers like the current streak
            flower_count += emotion_count

        # If the emotion's streak ends
        if person_list[i+1] != emotion and person_list[i] == emotion:
            # Resent the emotion's counter to zero
            emotion_count = 0

    # Return flower count
    return flower_count


# Get the count for each flower type

# For being happy for both people
sunflower_count = count_number_of_flowers("happy", one_person)
sunflower_count += count_number_of_flowers("happy", second_person)

# For being sad for both people
hyacinth_count = count_number_of_flowers("sad", one_person)
hyacinth_count += count_number_of_flowers("sad", second_person)

# For being angry for both people
rose_count = count_number_of_flowers("angry", one_person)
rose_count += count_number_of_flowers("angry", second_person)

# Define 5 variations of sunflower images and ensure standardized size
sunflower1_file = Image.open("elements/sunflowers/sunflower1.png")
sunflower1_resized = sunflower1_file.resize((70, 70))
sunflower1_image = ImageTk.PhotoImage(sunflower1_resized)
sunflower2_file = Image.open("elements/sunflowers/sunflower2.png")
sunflower2_resized = sunflower2_file.resize((70, 70))
sunflower2_image = ImageTk.PhotoImage(sunflower2_resized)
sunflower3_file = Image.open("elements/sunflowers/sunflower3.png")
sunflower3_resized = sunflower3_file.resize((70, 70))
sunflower3_image = ImageTk.PhotoImage(sunflower3_resized)
sunflower4_file = Image.open("elements/sunflowers/sunflower4.png")
sunflower4_resized = sunflower4_file.resize((70, 70))
sunflower4_image = ImageTk.PhotoImage(sunflower4_resized)
sunflower5_file = Image.open("elements/sunflowers/sunflower5.png")
sunflower5_resized = sunflower5_file.resize((70, 70))
sunflower5_image = ImageTk.PhotoImage(sunflower5_resized)

# Define 5 variations of hyacinth images and ensure standardized size
hyacinth1_file = Image.open("elements/hyacinths/hyacinth1.png")
hyacinth1_resized = hyacinth1_file.resize((70, 70))
hyacinth1_image = ImageTk.PhotoImage(hyacinth1_resized)
hyacinth2_file = Image.open("elements/hyacinths/hyacinth2.png")
hyacinth2_resized = hyacinth2_file.resize((70, 70))
hyacinth2_image = ImageTk.PhotoImage(hyacinth2_resized)
hyacinth3_file = Image.open("elements/hyacinths/hyacinth3.png")
hyacinth3_resized = hyacinth3_file.resize((70, 70))
hyacinth3_image = ImageTk.PhotoImage(hyacinth3_resized)
hyacinth4_file = Image.open("elements/hyacinths/hyacinth4.png")
hyacinth4_resized = hyacinth4_file.resize((70, 70))
hyacinth4_image = ImageTk.PhotoImage(hyacinth4_resized)
hyacinth5_file = Image.open("elements/hyacinths/hyacinth5.png")
hyacinth5_resized = hyacinth5_file.resize((70, 70))
hyacinth5_image = ImageTk.PhotoImage(hyacinth5_resized)

# Define 5 variations of rose images and ensure standardized size
rose1_file = Image.open("elements/roses/rose1.png")
rose1_resized = rose1_file.resize((70, 70))
rose1_image = ImageTk.PhotoImage(rose1_resized)
rose2_file = Image.open("elements/roses/rose2.png")
rose2_resized = rose2_file.resize((70, 70))
rose2_image = ImageTk.PhotoImage(rose2_resized)
rose3_file = Image.open("elements/roses/rose3.png")
rose3_resized = rose3_file.resize((70, 70))
rose3_image = ImageTk.PhotoImage(rose3_resized)
rose4_file = Image.open("elements/roses/rose4.png")
rose4_resized = rose4_file.resize((70, 70))
rose4_image = ImageTk.PhotoImage(rose4_resized)
rose5_file = Image.open("elements/roses/rose5.png")
rose5_resized = rose5_file.resize((70, 70))
rose5_image = ImageTk.PhotoImage(rose5_resized)


# Define function to place flowers based on flower type and its count
def place_flowers(flower_type, flower_count):
    # Define list which will store the flower coordinates
    flower_coordinates = []
    # For as many flowers as there are
    for i in range(flower_count):
        # Get a random x coordinate
        x_coordinate = random.randint(20, 1850)
        # Get random y coordinate based on the x coordinate (avoids locations where lake is placed)
        if x_coordinate < 380 or x_coordinate > 1400:
            y_coordinate = random.randint(50, 950)
        else:
            # Define a list to store to possible options for y variable
            y_coordinate_options = []
            # Get numbers from two possible ranges of y (which do not clash with lake location)
            y_coordinate_options.append(random.randint(50, 290))
            y_coordinate_options.append(random.randint(720, 950))
            # Randomly choose one of two possibilities by reshuffling and choosing the first position in list
            random.shuffle(y_coordinate_options)
            y_coordinate = y_coordinate_options[0]
        # Save random coordinates for the nth flower
        flower_coordinates.append((x_coordinate, y_coordinate))

    # For as many flowers as there are to be placed
    for i in range(flower_count):
        # Decide on which variation of the flower to use (1-5)
        flower_number = random.randrange(1, 5)
        # Place the corresponding flower variation based on the passed flower type onto the canvas
        if flower_type == "sunflowers":
            if flower_number == 1:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=sunflower1_image)
            if flower_number == 2:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=sunflower2_image)
            if flower_number == 3:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=sunflower3_image)
            if flower_number == 4:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=sunflower4_image)
            if flower_number == 5:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=sunflower5_image)
        if flower_type == "hyacinths":
            if flower_number == 1:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=hyacinth1_image)
            if flower_number == 2:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=hyacinth2_image)
            if flower_number == 3:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=hyacinth3_image)
            if flower_number == 4:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=hyacinth4_image)
            if flower_number == 5:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=hyacinth5_image)
        if flower_type == "roses":
            if flower_number == 1:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=rose1_image)
            if flower_number == 2:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=rose2_image)
            if flower_number == 3:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=rose3_image)
            if flower_number == 4:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=rose4_image)
            if flower_number == 5:
                # Put the image of the flower to canvas
                canvas.create_image(flower_coordinates[i][0], flower_coordinates[i][1], image=rose5_image)


# Place all the flowers based on their calculated count using the defined function
place_flowers("sunflowers", sunflower_count)
place_flowers("hyacinths", hyacinth_count)
place_flowers("roses", rose_count)

# Create variable which will store the number of benches to be stored
number_of_benches = 0

# Determine the number of benches to be placed
for i in range(7):
    if one_person[i] == "angry" and second_person[i] == "angry":
        number_of_benches += 1

# Load the image of bench
bench_file = Image.open("elements/bench.png")
# Resize the image of the bench
bench_resized = bench_file.resize((100, 100))
# Save it in format for the canvas
bench_image = ImageTk.PhotoImage(bench_resized)

# Define the list for benches coordinates (around lake)
locations_of_benches = [(780, 270), (1195, 262), (1530, 475), (1518, 645), (1163, 765), (727, 769), (373, 650), (370, 435)]

for i in range(number_of_benches):
    # Put the image of the bench to canvas
    canvas.create_image(locations_of_benches[i][0], locations_of_benches[i][1], image=bench_image)

# Create variable which will store the number of butterflies to be stored
number_of_butterflies = 0

# Determine the number of butterflies to be placed
for i in range(7):
    if one_person[i] == "happy" and second_person[i] == "happy":
        number_of_butterflies += 1

# Load the image of butterfly
butterfly_file = Image.open("elements/butterfly.png")
# Resize the image of the butterfly
butterfly_resized = butterfly_file.resize((32, 32))
# Save it in format for the canvas
butterfly_image = ImageTk.PhotoImage(butterfly_resized)

# Define the list for butterfly coordinates
locations_of_butterflies = []

# For as many butterflies as should be placed
for i in range(number_of_butterflies):
    # Generate random coordinates which are not too close to the border of the window
    coordinate_x = random.randrange(20, 1500)
    coordinate_y = random.randrange(20, 1000)
    # Save the coordinates in the format for image placing function to canvas
    coordinates = (coordinate_x, coordinate_y)
    # Save the coordinates in a list
    locations_of_butterflies.append(coordinates)

#  For as many butterflies as should be placed
for i in range(number_of_butterflies):
    # Put the image of the butterfly to canvas
    canvas.create_image(locations_of_butterflies[i][0], locations_of_butterflies[i][1], image=butterfly_image)

# Run the interface
window.mainloop()