import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Input

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = 'train'
CSV_FILE = 'train.csv'
TEST_CSV = 'test.csv'
TEST_IMAGE_DIR = 'Testing_data'
MODEL_FILE = 'best_model.h5'

def run_testing_gui():
    import sys
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QLabel, QPushButton,
        QVBoxLayout, QFileDialog, QTextEdit
    )
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtCore import Qt
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    class LeafLensApp(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("LeafLens - Image Tester")
            self.setGeometry(100, 100, 600, 700)

            self.model = load_model(MODEL_FILE)
            self.class_indices = load_class_indices()
            self.index_to_class = {v: k for k, v in self.class_indices.items()}

            self.init_ui()

        def init_ui(self):
            layout = QVBoxLayout()

            self.image_label = QLabel("Image Preview")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setFixedHeight(300)
            layout.addWidget(self.image_label)

            self.result_text = QTextEdit("Prediction results will appear here.")
            self.result_text.setReadOnly(True)
            layout.addWidget(self.result_text)

            self.button = QPushButton("Select Image")
            self.button.clicked.connect(self.select_image)
            layout.addWidget(self.button)

            self.setLayout(layout)

        def select_image(self):
            file_path, _ = QFileDialog.getOpenFileName(self, "Select an image", "", "Images (*.png *.jpg *.jpeg)")
            if not file_path:
                return

            # Show image
            pixmap = QPixmap(file_path).scaled(300, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

            # Predict
            raw_img = load_img(file_path, target_size=IMG_SIZE)
            img_array = img_to_array(raw_img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = self.model.predict(img_array, verbose=0)[0]
            top_indices = prediction.argsort()[-3:][::-1]

            result_lines = ["Top 3 Predictions:\n"]
            for idx in top_indices:
                label = self.index_to_class.get(idx, f"[Unknown class {idx}]")
                conf = prediction[idx] * 100
                result_lines.append(f"{label}: {conf:.2f}%")

            self.result_text.setPlainText("\n".join(result_lines))

    app = QApplication(sys.argv)
    window = LeafLensApp()
    window.show()
    sys.exit(app.exec_())

def train_model():
    print("Starting training...")

    df = pd.read_csv(CSV_FILE)
    df['filename'] = df['filename'].apply(lambda x: x.split('/agricultural-pests-image-dataset/')[-1])
    df['filename'] = df['filename'].apply(lambda x: os.path.join(TRAIN_DIR, x))

    if not os.path.exists(df['filename'].iloc[0]):
        raise FileNotFoundError("Check if 'train/' folder and image paths in train.csv are correct.")

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filename',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filename',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(len(train_generator.class_indices), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )

    print(f"\n Training complete. Best model saved as '{MODEL_FILE}'")
    save_class_indices(train_generator.class_indices)

def save_class_indices(class_indices):
    import json
    with open("class_indices.json", "w") as f:
        json.dump(class_indices, f)

def load_class_indices():
    import json
    with open("class_indices.json", "r") as f:
        return json.load(f)

def test_with_csv():
    print("Starting CSV-based test...")

    model = load_model(MODEL_FILE)
    class_indices = load_class_indices()
    index_to_class = {v: k for k, v in class_indices.items()}

    df = pd.read_csv(TEST_CSV)
    df['filename'] = df['filename'].apply(lambda x: x.split('/agricultural-pests-image-dataset/')[-1])
    df['filename'] = df['filename'].apply(lambda x: os.path.join(TEST_IMAGE_DIR, x))

    correct = 0
    total = 0

    for _, row in df.iterrows():
        img_path = row['filename']
        true_label = row['label']

        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)
        pred_index = np.argmax(preds)
        pred_label = index_to_class[pred_index]
        confidence = np.max(preds)

        plt.imshow(img)
        plt.axis('off')
        if confidence < 0.5:
            plt.title("(NO SPRAY) Uncertain Prediction")
            if pred_label != true_label:
                correct += 1
        else:
            plt.title(f"(SPRAY) Predicted: {pred_label} ({confidence*100:.2f}% confidence)")
            if pred_label == true_label:
                correct += 1
        total += 1
        #Optional: plt.show() to see the image that the model saw during the test

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n Final Test Accuracy: {accuracy:.2f}% on {total} images.")

if __name__ == "__main__":
    print("Select an option:")
    print("1 - Train Model")
    print("2 - Test Images using 'test.csv'")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        train_model()
    elif choice == "2":
        print("Use GUI for testing? (y/n): ", end="")
        gui_choice = input().strip().lower()
        if gui_choice == 'y':
            run_testing_gui()
        else:
            test_with_csv()
    else:
        print("Invalid option.")
