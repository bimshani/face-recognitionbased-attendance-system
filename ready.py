import os
import sys
import urllib.request
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image

# Configuration
VGG_FACE_WEIGHTS = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
MODEL_FOLDER = 'vgg_model'
MODEL_PATH = os.path.join(MODEL_FOLDER, 'vggface_vgg16.h5')

# Create directory if needed
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
    print(f"Created directory: {MODEL_FOLDER}")

# Download weights if needed
if not os.path.exists(MODEL_PATH):
    print(f"Downloading VGGFace weights to {MODEL_PATH}...")
    urllib.request.urlretrieve(VGG_FACE_WEIGHTS, MODEL_PATH)
    print("Download complete!")
else:
    print(f"VGGFace weights file already exists at {MODEL_PATH}")

# Create VGG16 structure (VGGFace model)
def vgg16_model(weights_path):
    img_input = Input(shape=(224, 224, 3))
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # Classification block (We load weights here)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    x = Dense(2622, activation='softmax', name='fc8')(x) 

    model = Model(img_input, x, name='vggface_vgg16')
    
    # Load weights 
    model.load_weights(weights_path) 
    print("Model weights loaded successfully")
    
    # Create a model without the classification layer for feature extraction
    feature_model = Model(inputs=model.input, outputs=model.get_layer('fc7').output)
    
    return model, feature_model

# Build the model and load weights
print("Building VGGFace model...")
vgg_full_model, vgg_feature_model = vgg16_model(MODEL_PATH)

# Save the models
full_model_path = os.path.join(MODEL_FOLDER, 'vggface_full.h5')
feature_model_path = os.path.join(MODEL_FOLDER, 'vggface_features.h5')

vgg_full_model.save(full_model_path)
vgg_feature_model.save(feature_model_path)

print(f"Full VGGFace model saved to: {full_model_path}")
print(f"VGGFace feature extraction model saved to: {feature_model_path}")

# Basic test to verify model works
print("\nVerifying model works with a sample image...")
# Create a black test image
test_img = np.zeros((1, 224, 224, 3))
# Get features
features = vgg_feature_model.predict(test_img)
print(f"Feature vector shape: {features.shape}")
print("Model verification complete!")

