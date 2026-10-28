
import os
import tensorflow as tf
from tensorflow.keras import layers, models

os.makedirs("models", exist_ok=True)

# Load dataset
train = tf.keras.utils.image_dataset_from_directory(
    "data/chest_xray/train",
    image_size=(128, 128),
    batch_size=32,
    label_mode="binary",
    class_names=["NORMAL", "PNEUMONIA"]
)

val = tf.keras.utils.image_dataset_from_directory(
    "data/chest_xray/val",
    image_size=(128, 128),
    batch_size=32,
    label_mode="binary",
    class_names=["NORMAL", "PNEUMONIA"]
)

# Simple CNN model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train,
    validation_data=val,
    epochs=3
)

model.save("models/simple_model.keras")

print("Training completed!")
