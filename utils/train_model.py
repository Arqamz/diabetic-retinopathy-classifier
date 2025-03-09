from data import create_train_val_test_datasets
from models import build_hybrid_model
import tensorflow as tf

def train_model(model, train_dataset, val_dataset, num_classes, epochs=15):
        
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('hybrid_model_best.keras', save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    # Train model
    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)

    model.save('hybrid_model_final.keras')

    return model, history

if __name__ == "__main__":
    
    # Create datasets
    train_dataset, val_dataset, _, num_classes, class_indices = create_train_val_test_datasets(
        "dataset/train", "dataset/val", "dataset/test", batch_size=32
    )

    # Build model
    model = build_hybrid_model(num_classes=num_classes)

    # Train model
    model, history = train_model(model, train_dataset, val_dataset, num_classes, epochs=15)
