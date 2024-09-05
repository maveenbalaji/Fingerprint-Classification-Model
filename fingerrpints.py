import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def load_images_from_folder(folder, image_size=(128, 128)):
    images = []
    labels = []
    label_names = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                if filename.endswith(".tif"):
                    img_path = os.path.join(label_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        edges = cv2.Canny(img, 100, 200)
                        images.append(edges)
                        labels.append(label_folder)
            label_names.append(label_folder)
    images = np.array(images)
    images = images.reshape(images.shape[0], image_size[0], image_size[1], 1)
    return images, labels, label_names

def preprocess_image(img_path="C:\\Users\\mavee\\Downloads\\fingerprints\\fingerprints\\DB4_B\\107_5.tif", image_size=(128, 128)):
    print(f"Reading image from path: {img_path}")
    if not os.path.exists(img_path):
        raise ValueError(f"Image not found at path: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, image_size)
        edges = cv2.Canny(img, 100, 200)
        edges = edges.reshape(1, image_size[0], image_size[1], 1)
        return edges
    else:
        raise ValueError("Image not found or unable to read the image.")

# Path to the main folder containing DB1_B, DB2_B, DB3_B, and DB4_B
main_folder = r'C:\Users\mavee\Downloads\fingerprints\fingerprints'  # Use raw string (r'path') or double backslashes

# Load images and labels
images, labels, label_names = load_images_from_folder(main_folder)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Save the model and the label encoder
model.save('fingerprint_model.h5')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Path to the test image
test_image_path = r'C:\\Users\\mavee\Downloads\\fingerprints\\test_image.tif'  # Replace with your test image path

# Preprocess the test image
test_image = preprocess_image(test_image_path)

# Load the trained model
model = tf.keras.models.load_model('fingerprint_model.h5')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Make predictions with the model
predicted_label_index = np.argmax(model.predict(test_image))
predicted_label = label_encoder.inverse_transform([predicted_label_index])

print(f'Predicted Label: {predicted_label[0]}')

# Display the test image and the predicted label
plt.imshow(test_image.reshape(128, 128), cmap='gray')
plt.title(f'Predicted Label: {predicted_label[0]}')
plt.show()
