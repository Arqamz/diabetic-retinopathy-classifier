from data import create_train_val_test_datasets
from models import build_hybrid_model, ViTFeatureExtractorLayer
from utils import train_model, evaluate_model, explain_model, analyze_model, visualize_activations

import tensorflow as tf

if __name__ == "__main__":
    
    # Create datasets
    train_dataset, val_dataset, test_dataset, num_classes, class_indices = create_train_val_test_datasets(
        "dataset/train", "dataset/val", "dataset/test", batch_size=32
    )

    # Build model
    model = build_hybrid_model(num_classes=num_classes)

    # Train model (uncomment to re-train)
    
    # model, history = train_model(model, train_dataset, val_dataset, num_classes, epochs=15)

    # Load the trained model
    model = tf.keras.models.load_model('hybrid_model_best.keras', custom_objects={'ViTFeatureExtractorLayer': ViTFeatureExtractorLayer})

    # First evaluate the model
    y_true, y_pred, all_probs = evaluate_model(model, test_dataset, class_indices)

    # Explain model predictions visually
    heatmaps = explain_model(model, test_dataset, class_indices, num_samples=5)

    # Analyze model
    analyze_model(model, test_dataset, class_indices)
    
    # Visualize activations
    visualize_activations(model, test_dataset, layer_name='conv5_block3_out', num_samples=3, num_filters=16)