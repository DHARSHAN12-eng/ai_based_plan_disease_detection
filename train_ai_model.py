import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
import json

# ==========================================
# 🌿 PlantGuard AI - Model Training Script
# ==========================================
# INSTRUCTIONS:
# 1. Download a plant disease dataset (e.g., PlantVillage from Kaggle).
# 2. Extract it. It should have folders for each disease (e.g., "Tomato___Early_blight", "Healthy").
# 3. Change the DATASET_DIR variable below to point to that extracted folder.
# 4. Run this script: python train_ai_model.py
# 5. It will save 'plant_disease_model.h5' and 'class_indices.json'. 
#    Copy these into your 'detection' folder for the Django app to use!

DATASET_DIR = "dummy_dataset"  # CHANGE THIS TO YOUR REAL DATASET LATER!
MODEL_SAVE_PATH = "detection/plant_disease_model.h5"
CLASSES_SAVE_PATH = "detection/class_indices.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 1

def train_model():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Directory '{DATASET_DIR}' not found!")
        print("Please set DATASET_DIR to the folder containing your plant disease images.")
        return

    print("🚀 Starting Data Processing...")
    
    # 1. Data Augmentation (Helps model generalize better)
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 # Use 20% for validation testing
    )

    # Training Data Generator
    train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Validation Data Generator
    validation_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    num_classes = train_generator.num_classes
    print(f"✅ Found {num_classes} disease classes!")

    # Save the class mapping for the Django app
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    with open(CLASSES_SAVE_PATH, 'w') as f:
        json.dump(class_indices, f)
    print(f"💾 Saved class indices to {CLASSES_SAVE_PATH}")

    print("🏗️ Building the MobileNetV2 Neural Network...")
    
    # 2. Build Model using Transfer Learning (MobileNetV2 is fast and accurate)
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model layers (don't retrain the basic shapes/colors)
    base_model.trainable = False

    # Add custom layers for our specific plant diseases
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x) # Prevent overfitting
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("🧠 Starting AI Training (This will take a while)...")
    
    # 4. Train!
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    print("✅ Training Complete!")
    
    # 5. Save the final model
    model.save(MODEL_SAVE_PATH)
    print(f"🎉 Model saved successfully as '{MODEL_SAVE_PATH}'")
    print(f"➡️ Next Step: Move '{MODEL_SAVE_PATH}' and '{CLASSES_SAVE_PATH}' into your 'detection' folder!")

if __name__ == "__main__":
    train_model()
