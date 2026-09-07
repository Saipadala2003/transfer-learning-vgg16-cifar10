import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU available:", bool(gpus))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# ---------------- Baseline CNN ----------------
x_train_cnn = x_train.astype("float32") / 255.0
x_test_cnn = x_test.astype("float32") / 255.0
y_train_cnn = tf.keras.utils.to_categorical(y_train, 10)
y_test_cnn = tf.keras.utils.to_categorical(y_test, 10)

baseline_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])
baseline_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

start_time = time.time()
baseline_history = baseline_model.fit(
    x_train_cnn, y_train_cnn, epochs=5, batch_size=64,
    validation_split=0.1, verbose=1
)
baseline_time = time.time() - start_time
baseline_loss, baseline_accuracy = baseline_model.evaluate(x_test_cnn, y_test_cnn, verbose=1)
print("Baseline accuracy:", round(baseline_accuracy * 100, 2), "%")
print("Baseline loss:", round(baseline_loss, 4))
print("Baseline training time:", round(baseline_time, 2), "seconds")

# ---------------- VGG16 Transfer Learning ----------------
IMG_SIZE = 96
BATCH_SIZE = 64

def preprocess_vgg(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    return image, label

train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10000)
            .map(preprocess_vgg, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
           .map(preprocess_vgg, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(96, 96, 3))
vgg_model = models.Sequential([
    vgg_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])
vgg_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

start_time = time.time()
vgg_history = vgg_model.fit(train_ds, epochs=5, validation_data=test_ds, verbose=1)
vgg_time = time.time() - start_time
vgg_loss, vgg_accuracy = vgg_model.evaluate(test_ds, verbose=1)
print("VGG16 transfer accuracy:", round(vgg_accuracy * 100, 2), "%")
print("VGG16 transfer loss:", round(vgg_loss, 4))
print("VGG16 transfer time:", round(vgg_time, 2), "seconds")

# ---------------- Fine-Tuning ----------------
for layer in vgg_base.layers:
    layer.trainable = False
for layer in vgg_base.layers[-4:]:
    layer.trainable = True

vgg_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

start_time = time.time()
fine_tune_history = vgg_model.fit(
    train_ds, epochs=3, validation_data=test_ds, verbose=1
)
fine_tune_time = time.time() - start_time
final_loss, final_accuracy = vgg_model.evaluate(test_ds, verbose=1)

total_vgg_time = vgg_time + fine_tune_time
accuracy_improvement = (final_accuracy - baseline_accuracy) * 100
print("Final VGG16 accuracy:", round(final_accuracy * 100, 2), "%")
print("Final VGG16 loss:", round(final_loss, 4))
print("Fine-tuning time:", round(fine_tune_time, 2), "seconds")
print("Total VGG16 time:", round(total_vgg_time, 2), "seconds")
print("Accuracy improvement:", round(accuracy_improvement, 2), "percentage points")

# ---------------- Evaluation ----------------
y_true, y_pred = [], []
for images, labels in test_ds:
    predictions = vgg_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
cm = confusion_matrix(y_true, y_pred)
print(cm)
print(classification_report(y_true, y_pred, target_names=class_names))

# ---------------- Plots ----------------
plt.figure(figsize=(8, 5))
plt.plot(baseline_history.history["accuracy"], label="Training Accuracy")
plt.plot(baseline_history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Baseline CNN Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(baseline_history.history["loss"], label="Training Loss")
plt.plot(baseline_history.history["val_loss"], label="Validation Loss")
plt.title("Baseline CNN Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(vgg_history.history["accuracy"], label="Training Accuracy")
plt.plot(vgg_history.history["val_accuracy"], label="Validation Accuracy")
plt.title("VGG16 Transfer Learning Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(vgg_history.history["loss"], label="Training Loss")
plt.plot(vgg_history.history["val_loss"], label="Validation Loss")
plt.title("VGG16 Transfer Learning Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(); plt.show()

plt.figure(figsize=(8, 5))
plt.bar(["Baseline CNN", "VGG16 Transfer Learning"], [baseline_accuracy * 100, final_accuracy * 100])
plt.title("Model Accuracy Comparison"); plt.xlabel("Model"); plt.ylabel("Test Accuracy (%)"); plt.ylim(0, 100); plt.show()

plt.figure(figsize=(8, 5))
plt.bar(["Baseline CNN", "VGG16 Transfer Learning"], [baseline_time, total_vgg_time])
plt.title("Training Time Comparison"); plt.xlabel("Model"); plt.ylabel("Training Time (seconds)"); plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Class"); plt.ylabel("Actual Class"); plt.title("VGG16 Confusion Matrix"); plt.show()

baseline_model.save("CIFAR10_Baseline_CNN.keras")
vgg_model.save("CIFAR10_VGG16_Transfer_Learning.keras")
print("Both models saved successfully.")
