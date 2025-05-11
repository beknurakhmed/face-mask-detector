# Face Mask Detection

This project implements a face mask detection system using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The system can:

- Train a CNN to classify images as "with mask" or "without mask."
- Evaluate the model on a test dataset.
- Detect face masks in real-time via webcam, single images, or video files, with a user-friendly menu interface.
- Save processed outputs (images or videos) with detection annotations.
- Adjust the prediction threshold dynamically.

The model achieves approximately 97.94% test accuracy on the provided dataset, using data augmentation for robust generalization and a customizable threshold to optimize predictions.

## Project Structure

```
face-mask-detector/
│
├── .venv/                    # Virtual environment
├── requirements.txt          # Dependencies
├── face_mask_detector.py     # Main script for training and detection
├── observations/             # Dataset directory
│   └── experiments/
│       └── dest_folder/
│           ├── train/        # Training images (with_mask/, without_mask/)
│           ├── val/          # Validation images (with_mask/, without_mask/)
│           ├── test/         # Test images (with_mask/, without_mask/)
│           ├── train.csv     # Training metadata (not used in script)
│           └── test.csv      # Test metadata (not used in script)
├── models/                   # Saved models (e.g., model-001.keras)
├── outputs/                  # Processed images/videos (e.g., processed_image_*.jpg, processed_video_*.mp4)
└── plot.png                  # Training/validation accuracy and loss plot
```

## Prerequisites

- **Python 3.8+**: Ensure Python is installed.
- **Webcam**: Required for real-time detection (optional if using image/video inputs).
- **Git**: To clone the dataset repository (optional).

## Setup Instructions

### 1. Create the Project Directory

```bash
mkdir face-mask-detector
cd face-mask-detector
```

Place `face_mask_detector.py` and `requirements.txt` in this directory.

### 2. Download the Dataset

The dataset is sourced from [prajnasb/observations](https://github.com/prajnasb/observations).

#### Option 1: Clone via Git
```bash
git clone https://github.com/prajnasb/observations.git
```

#### Option 2: Manual Download
1. Visit [https://github.com/prajnasb/observations](https://github.com/prajnasb/observations).
2. Download the ZIP file and extract it.
3. Move the `observations` folder into `face-mask-detector/`.

#### Verify Dataset Structure
Ensure the dataset is at `face-mask-detector/observations/experiments/dest_folder/` with:
- `train/with_mask/` and `train/without_mask/` (~658/657 images)
- `val/with_mask/` and `val/without_mask/` (~71/71 images)
- `test/with_mask/` and `test/without_mask/` (~97/97 images)
- `train.csv` and `test.csv` (not used)

### 3. Set Up the Virtual Environment

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

Install packages listed in `requirements.txt` (e.g., `tensorflow==2.19.0`, `opencv-python==4.11.0.86`, `numpy==2.1.3`, `matplotlib==3.10.1`).

```bash
pip install -r requirements.txt
```

### 5. Run the Script

```bash
python face_mask_detector.py
```

## Usage

### Script Workflow
1. **Model Check**:
   - Loads the latest model from `models/` (e.g., `model-030.keras`) if available.
   - Otherwise, trains a new model for 30 epochs using `train` and `val` datasets, saving the best model.

2. **Training** (if no model):
   - Applies data augmentation (rotation, shifts, zoom, brightness).
   - Saves models to `models/model-XXX.keras` based on validation loss.
   - Plots training/validation accuracy and loss as `plot.png`.

3. **Evaluation**:
   - Tests the model on the `test` dataset, reporting loss and accuracy.

4. **Menu Interface**:
   - Presents a text-based menu for selecting input sources and settings.
   - Supports webcam feed, image files, video files, threshold adjustment, and exit.

### Menu Interface
After training or loading the model, the script displays:

```
=== Face Mask Detection Menu ===
1. Webcam Feed
2. Image File
3. Video File
4. Set Prediction Threshold
5. Exit
Enter your choice (1-5):
```

For options 1–3, you’re prompted to save the output (`y/n`). Saved files are stored in `outputs/` with timestamps (e.g., `processed_image_20250511_123456.jpg`).

- **1. Webcam Feed**:
  - Streams webcam video, detecting faces and masks in real-time.
  - Draws red (without mask) or green (with mask) bounding boxes and labels.
  - If saving, captures one snapshot.
  - Press `ESC` to return to the menu.

- **2. Image File**:
  - Prompts for an image path (e.g., `image.jpg`).
  - Processes the image, displaying detections.
  - If saving, saves the annotated image.
  - Press any key to return to the menu.

- **3. Video File**:
  - Prompts for a video path (e.g., `video.mp4`).
  - Plays the video with detections (~33fps).
  - If saving, saves the annotated video as MP4.
  - Press `ESC` to return to the menu.

- **4. Set Prediction Threshold**:
  - Prompts for a new threshold (0.0 to 1.0, default 0.7).
  - Updates the threshold for "with mask" predictions.

- **5. Exit**:
  - Exits the program.

### Example Interaction

```
Dataset base path: /path/to/observations/experiments/dest_folder
Checking dataset directories...
Train with mask: 658
Train without mask: 657
Validation with mask: 71
Validation without mask: 71
Test with mask: 97
Test without mask: 97
Loading existing model: models/model-030.keras
Found 194 images belonging to 2 classes.
20/20 [==============================] - 2s 87ms/step - loss: 0.0987 - accuracy: 0.9794
Test Loss: 0.0987, Test Accuracy: 0.9794

=== Face Mask Detection Menu ===
1. Webcam Feed
2. Image File
3. Video File
4. Set Prediction Threshold
5. Exit
Enter your choice (1-5): 2
Save output? (y/n): y
Enter the path to the image file (e.g., image.jpg): test.jpg
Prediction score: 0.8500
[Image displayed with "with mask" label, saved as outputs/processed_image_20250511_123456.jpg]

=== Face Mask Detection Menu ===
...
Enter your choice (1-5): 4
Enter new prediction threshold (0.0 to 1.0): 0.8
Threshold updated to 0.8
```

## Troubleshooting

### 1. Dataset Not Found
- **Error**: `Error: Directory not found: observations/experiments/dest_folder/...`
- **Solution**:
  - Verify the dataset path and structure.
  - Re-download from [https://github.com/prajnasb/observations](https://github.com/prajnasb/observations).

### 2. Webcam Issues
- **Error**: `Error: Could not open webcam.`
- **Solution**:
  - Ensure the webcam is connected and not in use.
  - Test with:
    ```python
    import cv2
    cap = cv2.VideoCapture(0)
    print(cap.isOpened())
    cap.release()
    ```
  - Try a different camera index (e.g., `cv2.VideoCapture(1)`).

### 3. File Input Errors
- **Error**: `Error: Image file not found at ...`
- **Solution**:
  - Check the file path and ensure it exists.
  - Use absolute paths (e.g., `/path/to/image.jpg`) or relative paths from the script’s directory.
  - Ensure the file format is supported (e.g., `.jpg`, `.png` for images; `.mp4`, `.avi` for videos).

### 4. Poor Detection Accuracy
- **Issue**: Misclassifications (e.g., "with mask" detected as "without mask").
- **Solution**:
  - Check prediction scores (`Prediction score: X.XXXX`).
  - Adjust the threshold via menu option 4 (e.g., increase to 0.8 for stricter "with mask" detection).
  - Retrain with more diverse data or adjust augmentation parameters.

### 5. Face Detection Issues
- **Issue**: Faces not detected or bounding boxes misaligned.
- **Solution**:
  - Modify `detectMultiScale` parameters in `process_frame`:
    - `scaleFactor=1.05`
    - `minNeighbors=3`
  - Test face detection separately (see script comments).

### 6. Overfitting
- **Issue**: High training accuracy, low validation accuracy.
- **Solution**:
  - Increase `Dropout` (e.g., to 0.5).
  - Add L2 regularization to `Conv2D`/`Dense` layers.
  - Merge `val` into `train` and use a 10% validation split.

## Additional Notes

- **Model Architecture**:
  - The CNN uses three convolutional layers (32/64/128 filters), max pooling, dropout (0.3), and a dense layer (256 neurons) with a sigmoid output for binary classification.
  - Achieves ~97.94% test accuracy on the provided dataset.

- **Output Storage**: Processed images/videos are saved in `outputs/` with timestamps.

- **Prediction Threshold**:
  - The script uses a threshold of 0.7 for "with mask" predictions. Adjust this value in the script if misclassifications occur.

- **Dataset**:
  - Contains 1315 training images, 142 validation images, and 194 test images, balanced between "with_mask" and "without_mask."
  - `train.csv` and `test.csv` are not used but could be incorporated for metadata or custom splits.

- **Dependencies**:
  - The project uses `tensorflow==2.19.0`. Ensure your system meets TensorFlow's requirements (e.g., compatible CPU/GPU support).
  - If you encounter compatibility issues, consider downgrading to `tensorflow==2.17.0` and adjusting related dependencies.

- **Future Improvements**:
  - Add batch image processing.
  - Implement a GUI menu with `tkinter`.
  - Use `train.csv`/`test.csv` for metadata-driven data loading.