import os
import urllib.request
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Configuration
VGG_FACE_WEIGHTS = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
MODEL_FOLDER = 'vgg_model'
MODEL_PATH = os.path.join(MODEL_FOLDER, 'vggface_vgg16.h5')
FINE_TUNED_PATH = os.path.join(MODEL_FOLDER, 'vggface_finetuned.h5')

# Create directory if needed
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
    print(f"Created directory: {MODEL_FOLDER}")

# Download weights if needed
if not os.path.exists(MODEL_PATH):
    print(f"Downloading VGGFace weights to {MODEL_PATH}...")
    try:
        urllib.request.urlretrieve(VGG_FACE_WEIGHTS, MODEL_PATH)
        print("Download complete!")
    except Exception as e:
        raise Exception(f"Failed to download weights: {e}")
else:
    print(f"VGGFace weights file already exists at {MODEL_PATH}")

def vgg16_model(weights_path, num_classes=None, include_top=True):
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

    # Feature extraction layers
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    features = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(features)
    
    # Classification layer
    if include_top:
        if num_classes:
            x = Dense(num_classes, activation='softmax', name='fc8')(x)
        else:
            x = Dense(2622, activation='softmax', name='fc8')(x)
    else:
        # Return features if no top layer
        return Model(img_input, features, name='vggface_vgg16_features')

    model = Model(img_input, x, name='vggface_vgg16')
    feature_model = Model(inputs=model.input, outputs=features)
    
    # Load pre-trained weights
    try:
        if include_top:
            model.load_weights(weights_path)
            print("Pre-trained weights loaded successfully (full model)")
        else:
            # Load weights excluding the top layer
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print("Pre-trained weights loaded successfully (excluding top)")
    except Exception as e:
        raise Exception(f"Failed to load weights: {e}")
    
    return model, feature_model

# Build and save base models
print("Building VGGFace model...")
base_model, feature_model = vgg16_model(MODEL_PATH, include_top=True)

# Save base models
full_model_path = os.path.join(MODEL_FOLDER, 'vggface_full.h5')
feature_model_path = os.path.join(MODEL_FOLDER, 'vggface_features.h5')
base_model.save(full_model_path)
feature_model.save(feature_model_path)

print(f"Full VGGFace model saved to: {full_model_path}")
print(f"VGGFace feature extraction model saved to: {feature_model_path}")

# Fine-tuning function for transfer learning
def fine_tune_model(num_classes, learning_rate=0.0001):
    if num_classes < 2:
        raise ValueError("Number of classes must be at least 2 (including unknown)")
    
    if os.path.exists(FINE_TUNED_PATH):
        try:
            return tf.keras.models.load_model(FINE_TUNED_PATH)
        except Exception as e:
            print(f"Failed to load fine-tuned model: {e}, rebuilding...")
            os.remove(FINE_TUNED_PATH)
    
    # Create model without the top layer initially
    base_model = vgg16_model(MODEL_PATH, include_top=False)
    
    # Add new classification layer
    x = base_model.output
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', name='fc8')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze earlier layers, fine-tune later ones
    for layer in model.layers[:-4]:
        layer.trainable = False
    
    try:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.save(FINE_TUNED_PATH)
        print(f"Fine-tuned model saved to: {FINE_TUNED_PATH}")
    except Exception as e:
        raise Exception(f"Failed to compile or save fine-tuned model: {e}")
    
    return model

if __name__ == "__main__":
    print("\nVerifying model works with a sample image...")
    test_img = np.zeros((1, 224, 224, 3))
    features = feature_model.predict(test_img)
    print(f"Feature vector shape: {features.shape}")
    print("Model verification complete!")