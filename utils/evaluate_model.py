from data import create_train_val_test_datasets
from models import build_hybrid_model
import tensorflow as tf

def evaluate_model(model, test_dataset):
    
    """
    Evaluate the trained model on the test set and print the results.
    """
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Model Test accuracy: {test_acc}, Model Test loss: {test_loss}")

if __name__ == "__main__":
    
    # Create datasets
    _, _, test_dataset, num_classes, class_indices = create_train_val_test_datasets(
        "dataset/train", "dataset/val", "dataset/test", batch_size=32
    )

    # Load the trained model
    model = tf.keras.models.load_model('hybrid_model_best.keras')

    # Evaluate model
    evaluate_model(model, test_dataset)
