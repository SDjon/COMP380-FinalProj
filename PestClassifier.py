import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from PIL import Image
import csv
import matplotlib.pyplot as plt

# Parameters
IMG_SIZE = (500, 500)
TRAIN_DATA_DIR = 'Training_data'
TEST_DATA_DIR = 'Testing_data'
CSV_OUTPUT = 'predictions.csv'

userConfidenceChoice = float(input("Enter a confidence level to spray a pest(0.0 - 1.0): "))

def load_images(data_dir, class_names=None):
    images = []
    labels = []
    image_paths = []
    if class_names is None:
        class_names = sorted(os.listdir(data_dir))
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    for class_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = img_array * 2 - 1  # bipolar rep
                images.append(img_array)
                labels.append(class_to_index.get(class_name, -1))
                image_paths.append(img_path)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    images = np.expand_dims(images, axis=-1)
    return np.array(images), np.array(labels), class_names, image_paths

# Load training data
X_train, y_train, class_names, _ = load_images(TRAIN_DATA_DIR)

# Split train/validation
from sklearn.model_selection import train_test_split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Build model
model = models.Sequential([
    layers.Input(shape=(500, 500, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train_split, y_train_split, epochs=10, batch_size=16, validation_data=(X_val_split, y_val_split))

# Load test data
X_test, y_test, _, image_paths = load_images(TEST_DATA_DIR, class_names=class_names)

# Predict
predictions = model.predict(X_test)

# Open CSV file
with open(CSV_OUTPUT, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Index', 'Image Path', 'Predicted Label', 'Confidence', 'Decision'])

    for idx, pred in enumerate(predictions):
        confidence = np.max(pred)
        predicted_class = np.argmax(pred)
        predicted_label = class_names[predicted_class]

        # Spray logic
        if confidence >= userConfidenceChoice:
            decision = f"Spray → Detected: {predicted_label}"
        else:
            decision = f"No Spray (unknown or low confidence) → Guess: {predicted_label}"

        # Write to CSV
        writer.writerow([idx, image_paths[idx], predicted_label, f"{confidence:.2f}", decision])

        # Display image with matplotlib
        img_to_show = (X_test[idx].squeeze() + 1) / 2  # un-bipolar for display
        plt.imshow(img_to_show, cmap='gray')
        plt.title(f"Image {idx}: {decision}\nConfidence: {confidence:.2f}")
        plt.axis('off')
        plt.show()
