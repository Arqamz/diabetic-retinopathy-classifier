from data import create_train_val_test_datasets
from models import build_hybrid_model
from utils import train_model, evaluate_model

if __name__ == "__main__":
    
    # Create datasets
    train_dataset, val_dataset, test_dataset, num_classes, class_indices = create_train_val_test_datasets(
        "dataset/train", "dataset/val", "dataset/test", batch_size=32
    )

    # Build model
    model = build_hybrid_model(num_classes=num_classes)

    # Train model
    model, history = train_model(model, train_dataset, val_dataset, num_classes, epochs=15)

    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, test_dataset)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")