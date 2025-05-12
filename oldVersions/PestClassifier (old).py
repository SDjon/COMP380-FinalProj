import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import os
from PIL import Image
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Global constants
IMG_SIZE = (500, 500)
TRAIN_DATA_DIR = 'Training_data'
TEST_DATA_DIR = 'small_test'
CSV_OUTPUT = 'predictions.csv'
MODEL_FILE = 'mymodel.h5'


def load_images(data_dir, class_names=None):
    """Load grayscale images from directory and preprocess them."""
    images, labels, image_paths = [], [], []
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
                img = Image.open(img_path).convert('L')  # grayscale
                img = img.resize(IMG_SIZE)
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = img_array * 2 - 1  # bipolar [-1, 1]
                images.append(img_array)
                labels.append(class_to_index.get(class_name, -1))
                image_paths.append(img_path)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    images = np.expand_dims(images, axis=-1)  # add channel dimension
    return np.array(images), np.array(labels), class_names, image_paths


def build_model(num_classes):
    """Create and compile a CNN model with regularization and augmentation."""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.1),
    ])

    l2_reg = regularizers.l2(0.001)

    model = models.Sequential([
        layers.Input(shape=(500, 500, 1)),  # grayscale, 1 channel
        data_augmentation,
        layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2_reg),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2_reg),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=l2_reg),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model():
    """Train the model and save it to file."""
    X_train, y_train, class_names, _ = load_images(TRAIN_DATA_DIR)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    model = build_model(len(class_names))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train_split, y_train_split,
                        epochs=20, batch_size=4,  # reduced batch size
                        validation_data=(X_val_split, y_val_split),
                        callbacks=[early_stopping],
                        verbose=1)
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    final_val_acc = history.history['val_accuracy'][-1]
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")


def run_model(user_confidence_choice):
    """Load the trained model, make predictions, and save/display results."""
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Please train the model first.")
        return

    model = tf.keras.models.load_model(MODEL_FILE)
    _, _, class_names, _ = load_images(TRAIN_DATA_DIR)
    X_test, _, _, image_paths = load_images(TEST_DATA_DIR, class_names=class_names)

    predictions = model.predict(X_test)

    with open(CSV_OUTPUT, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Index', 'Image Path', 'Predicted Label', 'Confidence', 'Decision'])

        for idx, pred in enumerate(predictions):
            confidence = np.max(pred)
            predicted_class = np.argmax(pred)
            predicted_label = class_names[predicted_class]

            decision = (f"Spray → Detected: {predicted_label}"
                        if confidence >= user_confidence_choice
                        else f"No Spray (unknown or low confidence) → Guess: {predicted_label}")

            writer.writerow([idx, image_paths[idx], predicted_label, f"{confidence:.2f}", decision])

            img_to_show = (X_test[idx].squeeze() + 1) / 2  # un-bipolar for display
            plt.imshow(img_to_show, cmap='gray')
            plt.title(f"Image {idx}: {decision}\nConfidence: {confidence:.2f}")
            plt.axis('off')
            plt.show()
    print(f"Results saved to {CSV_OUTPUT}")


def main():
    """Main entry point for user interaction."""
    print("Select an option:")
    print("1 - Train Model")
    print("2 - Run Model")

    choice = input("Enter 1 or 2: ").strip()
    if choice == '1':
        train_model()
    elif choice == '2':
        try:
            confidence_input = float(input("Enter a confidence level to spray a pest (%): "))
            user_confidence_choice = confidence_input / 100
            run_model(user_confidence_choice)
        except ValueError:
            print("Invalid confidence value. Please enter a number.")
    else:
        print("Invalid option. Please enter 1 or 2.")


if __name__ == '__main__':
    main()
