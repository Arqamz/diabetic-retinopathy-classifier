import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Input, Dense, Dropout
from .vit import ViTFeatureExtractorLayer
from .resnet import create_resnet_branch

def build_hybrid_model(input_shape=(224, 224, 3), num_classes=5):
        
    # Dual input layers
    resnet_input = Input(shape=input_shape, name='resnet_input')
    vit_input = Input(shape=input_shape, name='vit_input')

    # ResNet50 branch
    cnn_features = create_resnet_branch(resnet_input)

    # ViT branch
    vit_layer = ViTFeatureExtractorLayer()
    vit_features = vit_layer(vit_input)
    vit_features = Dense(512, activation='relu')(vit_features)
    vit_features = Dropout(0.3)(vit_features)

    # Combined model
    combined = Concatenate()([cnn_features, vit_features])
    combined = Dense(512, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = tf.keras.Model(inputs=[resnet_input, vit_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
