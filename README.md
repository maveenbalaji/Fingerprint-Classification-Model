Fingerprint Classification Model
Overview

This project involves a Convolutional Neural Network (CNN) for classifying fingerprint images. The model is built using TensorFlow and Keras and is trained on grayscale images of fingerprints. It utilizes edge detection with the Canny algorithm to preprocess images before classification. The project includes code for loading data, preprocessing images, training the model, and making predictions on test images.

Features
Image Preprocessing: Uses the Canny edge detection algorithm to preprocess fingerprint images.
Model Training: Trains a CNN to classify fingerprint images into different categories.
Model Evaluation: Evaluates the model’s performance on test data.
Model Saving and Loading: Saves and loads the trained model and label encoder for inference.
Inference: Makes predictions on new fingerprint images and displays results.
Tech Stack
TensorFlow: For building and training the CNN model.
Keras: High-level API used for model creation and training.
OpenCV: For image preprocessing using the Canny edge detection algorithm.
Scikit-Learn: For label encoding and data splitting.
Joblib: For saving and loading the label encoder.
Matplotlib: For visualizing test image predictions.
How It Works
Data Loading:

Loads grayscale images from the specified directory.
Applies Canny edge detection to preprocess images.
Image Preprocessing:

Resizes images to 128x128 pixels.
Applies Canny edge detection to highlight edges.
Model Architecture:

A CNN with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.
Output layer uses softmax activation for classification.
Model Training:

Compiled with Adam optimizer and sparse categorical crossentropy loss.
Trained for 10 epochs with validation split.
Model Evaluation:

Evaluates the model on a test set and prints the accuracy.
Model Saving and Loading:

Saves the trained model and label encoder to disk.
Loads these artifacts for making predictions.
Inference:

Preprocesses a test image and makes predictions using the trained model.
Displays the test image and predicted label.
