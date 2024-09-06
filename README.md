# Fingerprint Classification Model

**Overview**

This project involves a Convolutional Neural Network (CNN) for classifying fingerprint images. The model is built using TensorFlow and Keras and is trained on grayscale images of fingerprints. It utilizes edge detection with the Canny algorithm to preprocess images before classification. The project includes code for loading data, preprocessing images, training the model, and making predictions on test images.

## Features

- **Image Preprocessing**: Uses the Canny edge detection algorithm to preprocess fingerprint images.
- **Model Training**: Trains a CNN to classify fingerprint images into different categories.
- **Model Evaluation**: Evaluates the model’s performance on test data.
- **Model Saving and Loading**: Saves and loads the trained model and label encoder for inference.
- **Inference**: Makes predictions on new fingerprint images and displays results.

## Tech Stack

- **TensorFlow**: For building and training the CNN model.
- **Keras**: High-level API used for model creation and training.
- **OpenCV**: For image preprocessing using the Canny edge detection algorithm.
- **Scikit-Learn**: For label encoding and data splitting.
- **Joblib**: For saving and loading the label encoder.
- **Matplotlib**: For visualizing test image predictions.

## Dataset

The dataset should be organized into subdirectories, each representing a different class of fingerprints. The directory structure should be as follows:

```
fingerprints/
    DB1_B/
        image1.tif
        image2.tif
        ...
    DB2_B/
        image1.tif
        image2.tif
        ...
    DB3_B/
        image1.tif
        image2.tif
        ...
    DB4_B/
        image1.tif
        image2.tif
        ...
```

## How It Works

1. **Data Loading**:
   - Loads grayscale images from the specified directory.
   - Applies Canny edge detection to preprocess images.

2. **Image Preprocessing**:
   - Resizes images to 128x128 pixels.
   - Applies Canny edge detection to highlight edges.

3. **Model Architecture**:
   - A CNN with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.
   - Output layer uses softmax activation for classification.

4. **Model Training**:
   - Compiled with Adam optimizer and sparse categorical crossentropy loss.
   - Trained for 10 epochs with validation split.

5. **Model Evaluation**:
   - Evaluates the model on a test set and prints the accuracy.

6. **Model Saving and Loading**:
   - Saves the trained model and label encoder to disk.
   - Loads these artifacts for making predictions.

7. **Inference**:
   - Preprocesses a test image and makes predictions using the trained model.
   - Displays the test image and predicted label.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/maveenbalaji/fingerprint-classification-model.git
   ```

2. Navigate to the project directory:
   ```bash
   cd fingerprint-classification-model
   ```

3. Install dependencies:
   ```bash
   pip install tensorflow opencv-python numpy matplotlib scikit-learn joblib
   ```

## Training the Model

1. Update the `main_folder` variable in the code to point to your dataset directory.
2. Adjust image size and batch size if needed.
3. Run the training script:
   ```bash
   python train_model.py
   ```

## Making Predictions

1. Update the `test_image_path` variable in the code with the path to the image you want to classify.
2. Run the inference script:
   ```bash
   python infer_model.py
   ```

## Example Usage

```python
test_image_path = r'C:\\Users\\mavee\Downloads\\fingerprints\\test_image.tif'
test_image = preprocess_image(test_image_path)
model = tf.keras.models.load_model('fingerprint_model.h5')
label_encoder = joblib.load('label_encoder.pkl')
predicted_label_index = np.argmax(model.predict(test_image))
predicted_label = label_encoder.inverse_transform([predicted_label_index])
print(f'Predicted Label: {predicted_label[0]}')
```

## Future Improvements

- **Expand Dataset**: Include more fingerprint images and categories.
- **Model Tuning**: Experiment with different architectures and hyperparameters.
- **User Interface**: Develop a graphical or web-based interface for interactive classification.

