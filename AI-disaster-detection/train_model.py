import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# ==============================
# DATASET PATH
# ==============================

DATASET_PATH = r"C:\Users\BHAVA\Downloads\AIDER_Split"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "validation")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# ==============================
# SETTINGS
# ==============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 5

# ==============================
# LOAD DATASET
# ==============================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names

print("\nClasses:")
print(class_names)

# ==============================
# PREPROCESSING
# ==============================

def preprocess(images, labels):
    images = preprocess_input(images)
    return images, labels


train_dataset = train_dataset.map(preprocess)
validation_dataset = validation_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# ==============================
# RESNET50 BASE MODEL
# ==============================

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False

# ==============================
# BUILD MODEL
# ==============================

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# TRAIN
# ==============================

EPOCHS = 10

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# ==============================
# TEST
# ==============================

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\n==============================")
print("TEST RESULTS")
print("==============================")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ==============================
# SAVE MODEL
# ==============================

os.makedirs("model", exist_ok=True)

model.save("model/disaster_resnet50.keras")

print("\nModel saved successfully!")
print("Location: model/disaster_resnet50.keras")

# ==============================
# SAVE CLASS NAMES
# ==============================

with open("model/class_names.txt", "w") as file:
    for class_name in class_names:
        file.write(class_name + "\n")

print("Class names saved successfully!")