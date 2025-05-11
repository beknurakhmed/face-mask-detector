import os
import glob
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

# Define dataset paths
DATASET_BASE = "observations/experiments/dest_folder"
TRAIN_DIR = os.path.join(DATASET_BASE, "train")
VAL_DIR = os.path.join(DATASET_BASE, "val")
TEST_DIR = os.path.join(DATASET_BASE, "test")

# Create output directory for saving processed images/videos
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Print dataset base path for debugging
print(f"Dataset base path: {os.path.abspath(DATASET_BASE)}")

def check_and_print_stats(directory, name):
    """Check if directory exists and print image counts for each class."""
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
        rotation_range=60,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
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

# Face detection setup
face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels_dict = {0: 'without mask', 1: 'with mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
size = 4
threshold = 0.7

def process_frame(frame, save_path=None):
    """Process a single frame for face mask detection, optionally saving the output."""
    im = cv2.flip(frame, 1, 1)  # Flip to act as a mirror
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
    faces = face_clsfr.detectMultiScale(mini, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = im[y*size:(y+h)*size, x*size:(x+w)*size]
        resized = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, IMG_WIDTH, IMG_HEIGHT, 3))
        result = model.predict(reshaped, verbose=0)
        
        score = result[0, 0]
        print(f"Prediction score: {score:.4f}")
        label = 1 if score >= threshold else 0
        cv2.rectangle(im, (x*size, y*size), ((x+w)*size, (y+h)*size), color_dict[label], 2)
        cv2.rectangle(im, (x*size, (y*size)-40), ((x+w)*size, y*size), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x*size + 10, (y*size)-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, im)
    
    return im

def process_webcam(save_output=False):
    """Process live webcam feed, optionally saving a snapshot."""
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    save_path = None
    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(OUTPUT_DIR, f"webcam_snapshot_{timestamp}.jpg")

    while True:
        rval, im = webcam.read()
        if not rval:
            print("Error: Failed to capture image from webcam.")
            break

        im = process_frame(im, save_path if save_output else None)
        save_output = False  # Save only the first frame
        cv2.imshow('Mask Detection', im)
        key = cv2.waitKey(10)
        if key == 27:  # ESC key to exit
            break

    webcam.release()
    cv2.destroyAllWindows()

def process_image(image_path, save_output=False):
    """Process a single image file, optionally saving the output."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    im = cv2.imread(image_path)
    if im is None:
        print(f"Error: Failed to load image at {image_path}")
        return

    save_path = None
    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_image_{timestamp}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)

    im = process_frame(im, save_path)
    cv2.imshow('Mask Detection', im)
    cv2.waitKey(0)  # Wait for any key press to close
    cv2.destroyAllWindows()

def process_video(video_path, save_output=False):
    """Process a video file, optionally saving the output."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Failed to open video at {video_path}")
        return

    save_path = None
    video_writer = None
    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(OUTPUT_DIR, f"processed_video_{timestamp}.mp4")
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        rval, im = video.read()
        if not rval:
            break

        im = process_frame(im)
        if video_writer:
            video_writer.write(im)
        cv2.imshow('Mask Detection', im)
        key = cv2.waitKey(30)  # 30ms delay for ~33fps
        if key == 27:  # ESC key to exit
            break

    video.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

def set_threshold():
    """Prompt user to set a new prediction threshold."""
    global threshold
    try:
        new_threshold = float(input("Enter new prediction threshold (0.0 to 1.0): "))
        if 0.0 <= new_threshold <= 1.0:
            threshold = new_threshold
            print(f"Threshold updated to {threshold}")
        else:
            print("Error: Threshold must be between 0.0 and 1.0")
    except ValueError:
        print("Error: Invalid input. Please enter a number.")

def menu():
    """Display menu and handle user input for face mask detection."""
    while True:
        print("\n=== Face Mask Detection Menu ===")
        print("1. Webcam Feed")
        print("2. Image File")
        print("3. Video File")
        print("4. Set Prediction Threshold")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        save_output = False
        if choice in ['1', '2', '3']:
            save_choice = input("Save output? (y/n): ").lower()
            save_output = save_choice == 'y'

        if choice == '1':
            process_webcam(save_output)
        elif choice == '2':
            image_path = input("Enter the path to the image file (e.g., image.jpg): ")
            process_image(image_path, save_output)
        elif choice == '3':
            video_path = input("Enter the path to the video file (e.g., video.mp4): ")
            process_video(video_path, save_output)
        elif choice == '4':
            set_threshold()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

# Run the menu
if __name__ == "__main__":
    menu()