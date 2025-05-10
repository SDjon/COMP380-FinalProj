import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from PIL import Image
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

# Global constants
IMG_SIZE = (500, 500)
TRAIN_DATA_DIR = 'Training_data'
TEST_DATA_DIR = 'Testing_data'
CSV_OUTPUT = 'predictions.csv'
MODEL_FILE = 'mymodel.h5'


def load_images(data_dir, class_names=None):
    """Load images from directory and preprocess them."""
    images, labels, image_paths = [], [], []
    if class_names is None:
        class_names = sorted(os.listdir(data_dir))
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    # Always loop over actual folders in data_dir
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
                img_array = img_array * 2 - 1  # bipolar representation
                images.append(img_array)
                labels.append(class_to_index.get(class_name, -1))
                image_paths.append(img_path)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    images = np.expand_dims(images, axis=-1)
    return np.array(images), np.array(labels), class_names, image_paths


def build_model(num_classes):
    """Create and compile the CNN model."""
    model = models.Sequential([
        layers.Input(shape=(500, 500, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model_kfold(k=5, epochs=50, batch_size=16, patience=5):
    """Train the model using K-fold cross-validation with early stopping."""
    X, y, class_names, _ = load_images(TRAIN_DATA_DIR)
    num_classes = len(class_names)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    best_val_accuracy = 0
    best_model = None

    fold_num = 1
    for train_index, val_index in kf.split(X):
        print(f"\n--- Fold {fold_num}/{k} ---")
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model = build_model(num_classes)

        # Add EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',     # You can change to 'val_accuracy' if preferred
            patience=patience,      # Number of epochs to wait for improvement
            restore_best_weights=True
        )

        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping],
            verbose=1
        )

        val_acc = max(history.history['val_accuracy'])
        print(f"Fold {fold_num} Best Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model = model

        # Clear memory between folds
        K.clear_session()
        fold_num += 1

    if best_model:
        best_model.save(MODEL_FILE)
        print(f"\nBest model saved to {MODEL_FILE} with validation accuracy: {best_val_accuracy:.4f}")
    else:
        print("No best model found.")


def run_model(user_confidence_choice):
    """Load the trained model, make predictions, and save results to CSV."""
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Please train the model first.")
        return

    model = tf.keras.models.load_model(MODEL_FILE)
    # Reload class_names from training set to ensure correct label map
    _, _, class_names, _ = load_images(TRAIN_DATA_DIR)
    X_test, y_test, _, image_paths = load_images(TEST_DATA_DIR, class_names=class_names)

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
                        else f"No Spray (unknown or low confidence)")

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
