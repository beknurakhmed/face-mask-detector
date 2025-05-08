import os
import glob
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Define dataset paths
DATASET_BASE = "observations/experiments/dest_folder"
TRAIN_DIR = os.path.join(DATASET_BASE, "train")
VAL_DIR = os.path.join(DATASET_BASE, "val")
TEST_DIR = os.path.join(DATASET_BASE, "test")

# Print dataset base path for debugging
print(f"Dataset base path: {os.path.abspath(DATASET_BASE)}")

# Check if directories exist and print dataset statistics
def check_and_print_stats(directory, name):
    with_mask_dir = os.path.join(directory, "with_mask")
    without_mask_dir = os.path.join(directory, "without_mask")
    
    if not os.path.exists(with_mask_dir):
        print(f"Error: Directory not found: {with_mask_dir}")
        return False
    if not os.path.exists(without_mask_dir):
        print(f"Error: Directory not found: {without_mask_dir}")
        return False
    
    print(f"{name} with mask: {len(os.listdir(with_mask_dir))}")
    print(f"{name} without mask: {len(os.listdir(without_mask_dir))}")
    return True

# Verify all dataset directories
print("Checking dataset directories...")
if not all([
    check_and_print_stats(TRAIN_DIR, "Train"),
    check_and_print_stats(VAL_DIR, "Validation"),
    check_and_print_stats(TEST_DIR, "Test")
]):
    print("One or more dataset directories are missing. Please verify the dataset structure.")
    exit()

IMG_HEIGHT = 150
IMG_WIDTH = 150

# Check for existing model
model_files = glob.glob("models/model-*.keras")
if model_files:
    # Load the most recent model based on epoch number
    latest_model = max(model_files, key=lambda x: int(x.split('-')[1].split('.')[0]))
    print(f"Loading existing model: {latest_model}")
    try:
        model = tf.keras.models.load_model(latest_model)
    except Exception as e:
        print(f"Error loading model {latest_model}: {e}")
        print("Falling back to training a new model...")
        model_files = []  # Trigger training
else:
    print("No existing model found. Training a new model...")

# Define and compile the CNN model if no model was loaded
if not model_files:
    model = tf.keras.models.Sequential([
        Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

acc = []
val_acc = []
loss = []
val_loss = []
epochs = 30

def train_model():
    """Train the model using data augmentation for training and validation."""
    global acc, val_acc, loss, val_loss
    
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=60,  # Increased for better generalization
        width_shift_range=0.3,  # Increased
        height_shift_range=0.3,  # Increased
        shear_range=0.3,  # Increased
        zoom_range=0.3,  # Increased
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]  # Added to handle lighting variations
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        batch_size=10,
        class_mode='binary',
        target_size=(IMG_WIDTH, IMG_HEIGHT)
    )

    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_generator = validation_datagen.flow_from_directory(
        VAL_DIR,
        batch_size=10,
        class_mode='binary',
        target_size=(IMG_WIDTH, IMG_HEIGHT)
    )

    checkpoint = ModelCheckpoint(
        'models/model-{epoch:03d}.keras',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto'
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint]
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

def plot():
    """Plot training and validation accuracy and loss."""
    epochs_range = range(epochs)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # save plt 
    plt.savefig('plot.png')
    plt.show()
    
    

# Train the model if no model was loaded, then plot results
if not model_files:
    train_model()
    plot()

# Evaluate on test set
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    batch_size=10,
    class_mode='binary',
    target_size=(IMG_WIDTH, IMG_HEIGHT)
)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Live face mask detection using webcam
face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels_dict = {0: 'without mask', 1: 'with mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
size = 4
threshold = 0.7  # Adjust threshold for "with mask" prediction

webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    rval, im = webcam.read()
    if not rval:
        print("Error: Failed to capture image from webcam.")
        break

    im = cv2.flip(im, 1, 1)  # Flip to act as a mirror
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
    faces = face_clsfr.detectMultiScale(mini, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = im[y*size:(y+h)*size, x*size:(x+w)*size]
        resized = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, IMG_WIDTH, IMG_HEIGHT, 3))
        result = model.predict(reshaped, verbose=0)
        
        score = result[0, 0]
        print(f"Prediction score: {score:.4f}")  # Debug: Print raw prediction score
        label = 1 if score >= threshold else 0  # Use custom threshold
        cv2.rectangle(im, (x*size, y*size), ((x+w)*size, (y+h)*size), color_dict[label], 2)
        cv2.rectangle(im, (x*size, (y*size)-40), ((x+w)*size, y*size), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x*size + 10, (y*size)-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Mask Detection', im)
    key = cv2.waitKey(10)
    if key == 27:  # ESC key to exit
        break

webcam.release()
cv2.destroyAllWindows()