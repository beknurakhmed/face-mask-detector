# Face Mask Detector

This project implements a face mask detection system using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The system can:

- Train a CNN model to classify images as "with mask" or "without mask."
- Evaluate the model on a test set.
- Perform real-time face mask detection using a webcam, drawing bounding boxes and labels on detected faces.

The model is trained on a dataset of images with and without masks, achieving a test accuracy of approximately 97.94%. The project includes data augmentation to improve generalization and a custom threshold for predictions to reduce misclassifications.

## Project Structure

```
face-mask-detector/
│
├── .venv/                    # Virtual environment
├── requirements.txt          # Dependencies
├── face_mask_detector.py     # Main script for training and webcam detection
├── observations/             # Dataset directory
│   └── experiments/
│       └── dest_folder/
│           ├── train/        # Training images (with_mask/, without_mask/)
│           ├── val/          # Validation images (with_mask/, without_mask/)
│           ├── test/         # Test images (with_mask/, without_mask/)
│           ├── train.csv     # Training metadata (not used in script)
│           └── test.csv      # Test metadata (not used in script)
├── models/                   # Saved models (e.g., model-001.keras)
└── plot.png                  # Training/validation accuracy and loss plot
```

## Prerequisites

- **Python 3.8+**: Ensure Python is installed on your system.
- **Webcam**: Required for real-time face mask detection.
- **Git**: To download the dataset (optional, if cloning the repository).

## Setup Instructions

### 1. Clone or Create the Project Directory

Create a project directory and place the `face_mask_detector.py` script and `requirements.txt` file inside it.

```bash
mkdir face-mask-detector
cd face-mask-detector
```

### 2. Download the Dataset

The dataset is sourced from [prajnasb/observations](https://github.com/prajnasb/observations). Follow these steps to download and set it up:

#### Option 1: Download via Git
```bash
git clone https://github.com/prajnasb/observations.git
```

#### Option 2: Manual Download
1. Visit [https://github.com/prajnasb/observations](https://github.com/prajnasb/observations).
2. Download the repository as a ZIP file and extract it.
3. Move the `observations` folder into the `face-mask-detector` directory.

#### Verify Dataset Structure
Ensure the dataset is located at `face-mask-detector/observations/experiments/dest_folder/`. Inside `dest_folder`, you should see:
- `train/with_mask/` and `train/without_mask/`
- `val/with_mask/` and `val/without_mask/`
- `test/with_mask/` and `test/without_mask/`
- `train.csv` and `test.csv`

The script expects 658/657 images in `train`, 71/71 in `val`, and 97/97 in `test` for `with_mask`/`without_mask`.

### 3. Set Up the Virtual Environment

Create and activate a virtual environment to manage dependencies.

#### On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

Install the required packages listed in `requirements.txt`. This file includes all necessary dependencies, including `tensorflow==2.19.0`, `opencv-python==4.11.0.86`, `numpy==2.1.3`, and `matplotlib==3.10.1`, along with their transitive dependencies.

```bash
pip install -r requirements.txt
```

### 5. Run the Script

Run the `face_mask_detector.py` script to train the model (if no pre-trained model exists) and start webcam detection.

```bash
python face_mask_detector.py
```

## Usage

### What the Script Does
1. **Check for Pre-trained Model**:
   - Looks for existing models in the `models/` directory (e.g., `models/model-001.keras`).
   - If found, loads the most recent model based on the epoch number.
   - If not found, trains a new model.

2. **Training (if no model exists)**:
   - Trains a CNN for 30 epochs using the `train` and `val` datasets.
   - Uses data augmentation (rotation, shifts, shear, zoom, brightness) for better generalization.
   - Saves the best model (based on validation loss) to `models/model-XXX.keras`.
   - Plots training/validation accuracy and loss, saving the plot as `plot.png`.

3. **Evaluation**:
   - Evaluates the model on the `test` dataset and prints test loss and accuracy.

4. **Webcam Detection**:
   - Uses OpenCV to capture webcam feed.
   - Detects faces using Haar Cascade (`haarcascade_frontalface_default.xml`).
   - Predicts "with mask" or "without mask" for each face.
   - Draws a bounding box (red for "without mask," green for "with mask") and label.
   - Prints the raw prediction score (0 to 1) for debugging.
   - Press `ESC` to exit the webcam feed.

### Expected Output
- **Dataset Statistics**:
  ```
  Dataset base path: C:\Users\Victus\Desktop\face-mask-detector\observations\experiments\dest_folder
  Checking dataset directories...
  Train with mask: 658
  Train without mask: 657
  Validation with mask: 71
  Validation without mask: 71
  Test with mask: 97
  Test without mask: 97
  ```
- **Model Loading or Training**:
  - If a model exists: `Loading existing model: models/model-XXX.keras`
  - If training: `No existing model found. Training a new model...` followed by training logs.
- **Test Results**:
  ```
  Found 194 images belonging to 2 classes.
  20/20 [==============================] - 2s 87ms/step - accuracy: 0.9794 - loss: 0.0987
  Test Loss: 0.0987, Test Accuracy: 0.9794
  ```
- **Webcam Detection**:
  - Opens a window showing the webcam feed with bounding boxes and labels.
  - Prints prediction scores: `Prediction score: 0.XXXX`

## Troubleshooting

### 1. Dataset Not Found
- **Error**: `Error: Directory not found: observations/experiments/dest_folder/train/with_mask`
- **Solution**:
  - Ensure the dataset is correctly placed at `face-mask-detector/observations/experiments/dest_folder/`.
  - Verify the folder structure using:
    ```bash
    dir observations\experiments\dest_folder /s  # Windows
    ls -R observations/experiments/dest_folder   # macOS/Linux
    ```
  - If missing, re-download from [https://github.com/prajnasb/observations](https://github.com/prajnasb/observations).

### 2. Webcam Not Working
- **Error**: `Error: Could not open webcam.`
- **Solution**:
  - Ensure your webcam is connected and not in use by another application.
  - Test the webcam independently:
    ```python
    import cv2
    cap = cv2.VideoCapture(0)
    print(cap.isOpened())
    cap.release()
    ```
  - If `False`, check your webcam drivers or try a different camera index (e.g., `cv2.VideoCapture(1)`).

### 3. Poor Detection Accuracy
- **Issue**: The model misclassifies "with mask" or "without mask."
- **Solution**:
  - Check the prediction score printed in the terminal (e.g., `Prediction score: 0.XXXX`).
    - If the score is close to 0.7 (e.g., 0.65), adjust the `threshold` in the script (e.g., to 0.8).
  - Improve lighting and ensure a plain background.
  - Retrain the model with more diverse data:
    - Add more "without_mask" images that match your webcam conditions.
    - Increase data augmentation further (e.g., `rotation_range=90`).
  - Simplify the model to reduce overfitting (e.g., reduce filters to 16/32/64).

### 4. Face Detection Issues
- **Issue**: No faces detected or bounding box misaligned.
- **Solution**:
  - Adjust `detectMultiScale` parameters in the script:
    - Decrease `scaleFactor` (e.g., to 1.05).
    - Decrease `minNeighbors` (e.g., to 3).
  - Test face detection independently:
    ```python
    import cv2
    cap = cv2.VideoCapture(0)
    face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_clsfr.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(10) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

### 5. Overfitting
- **Issue**: High training accuracy but fluctuating validation accuracy.
- **Solution**:
  - The validation set is small (142 images). Merge `val` into `train` and split 10% for validation.
  - Increase `Dropout` (e.g., to 0.5).
  - Add L2 regularization: `kernel_regularizer=tf.keras.regularizers.l2(0.01)` in `Conv2D` or `Dense` layers.
  - Reduce model complexity (e.g., use 16/32/64 filters).

## Additional Notes

- **Model Architecture**:
  - The CNN uses three convolutional layers (32/64/128 filters), max pooling, dropout (0.3), and a dense layer (256 neurons) with a sigmoid output for binary classification.
  - Achieves ~97.94% test accuracy on the provided dataset.

- **Prediction Threshold**:
  - The script uses a threshold of 0.7 for "with mask" predictions. Adjust this value in the script if misclassifications occur.

- **Dataset**:
  - Contains 1315 training images, 142 validation images, and 194 test images, balanced between "with_mask" and "without_mask."
  - `train.csv` and `test.csv` are not used but could be incorporated for metadata or custom splits.

- **Future Improvements**:
  - Use `train.csv` and `test.csv` for more precise data loading.
  - Add more diverse training data to improve generalization.
  - Implement real-time model retraining based on user feedback.

- **Dependencies**:
  - The project uses `tensorflow==2.19.0`. Ensure your system meets TensorFlow's requirements (e.g., compatible CPU/GPU support).
  - If you encounter compatibility issues, consider downgrading to `tensorflow==2.17.0` and adjusting related dependencies.