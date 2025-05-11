import os
import sys
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
MODEL_FILE = 'best_model.h5'
TEST_DIR = 'small_test/pics'

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

def test_images():
    print("Starting image testing...")
    model = load_model(MODEL_FILE)
    class_indices = load_class_indices()
    index_to_class = {v: k for k, v in class_indices.items()}

    image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for img_name in image_files:
        img_path = os.path.join(TEST_DIR, img_name)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        class_id = np.argmax(preds)
        confidence = np.max(preds)
        label = index_to_class[class_id]

        # Display with matplotlib
        plt.imshow(img)
        if confidence < .50:
            plt.title("(NO SPRAY) Image unknown or confidence too low")
        else:
            plt.title(f"(SPRAY) Prediction: {label} ({confidence*100:.2f}% confidence)")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    print("Select an option:")
    print("1 - Train Model")
    print("2 - Test Images in 'small_test/pics'")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        train_model()
    elif choice == "2":
        test_images()
    else:
        print("Invalid option.")
