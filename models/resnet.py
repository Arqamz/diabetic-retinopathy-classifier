# models/resnet.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

def create_resnet_branch(input_layer):
    # Load the ResNet50 model pre-trained on ImageNet, excluding the top layers
    cnn_base = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)
    
    # Freeze the first 100 layers of the ResNet50 model
    for layer in cnn_base.layers[:100]:
        layer.trainable = False
        
    # Reduce the spatial dimensions
    cnn_features = GlobalAveragePooling2D()(cnn_base.output)
    # Dense layer and ReLU activation
    cnn_features = Dense(512, activation='relu')(cnn_features)
    # Dropout layer to prevent overfitting
    cnn_features = Dropout(0.3)(cnn_features)
    
    return cnn_features
