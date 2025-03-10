import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data import create_train_val_test_datasets
from .evaluate_model import evaluate_model

def analyze_model(model, test_dataset, class_names):
    """
    Comprehensive model performance analysis
    """
    # First run basic evaluation
    y_true, y_pred, all_probs = evaluate_model(model, test_dataset, class_names)
    
    # Analyze per-class performance in detail
    class_analysis = []
    for i, class_name in enumerate(class_names):
        # Get indices for this class
        class_indices = (y_true == i)
        if not any(class_indices):
            continue
            
        # Calculate metrics
        correct_count = np.sum(y_pred[class_indices] == i)
        total_count = np.sum(class_indices)
        accuracy = correct_count / total_count
        
        # Find common misclassifications
        if total_count - correct_count > 0:
            misclassified = y_pred[class_indices & (y_pred != i)]
            most_common_error = np.bincount(misclassified).argmax() if len(misclassified) > 0 else None
            error_count = np.sum(misclassified == most_common_error) if most_common_error is not None else 0
            error_pct = error_count / total_count if total_count > 0 else 0
            most_common_error_class = class_names[most_common_error] if most_common_error is not None else "None"
        else:
            most_common_error_class = "None"
            error_pct = 0
        
        # Calculate confidence statistics for correct predictions
        correct_indices = class_indices & (y_pred == i)
        if any(correct_indices):
            correct_confidences = np.max(all_probs[correct_indices], axis=1)
            avg_confidence = np.mean(correct_confidences)
            min_confidence = np.min(correct_confidences)
            max_confidence = np.max(correct_confidences)
        else:
            avg_confidence = min_confidence = max_confidence = 0
            
        # Store analysis
        class_analysis.append({
            'class': class_name,
            'accuracy': accuracy,
            'sample_count': total_count,
            'most_common_error': most_common_error_class,
            'error_percentage': error_pct,
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence
        })
    
    # Print the analysis as a table
    print("\nDetailed Class Performance Analysis:")
    print("-" * 100)
    print(f"{'Class':<15} | {'Accuracy':<10} | {'Samples':<8} | {'Most Common Error':<20} | {'Error %':<8} | {'Avg Conf':<8} | {'Min Conf':<8} | {'Max Conf':<8}")
    print("-" * 100)
    
    for analysis in class_analysis:
        print(f"{analysis['class']:<15} | {analysis['accuracy']:.2f}      | {analysis['sample_count']:<8} | "
              f"{analysis['most_common_error']:<20} | {analysis['error_percentage']:.2f}    | "
              f"{analysis['avg_confidence']:.2f}    | {analysis['min_confidence']:.2f}    | {analysis['max_confidence']:.2f}")
    
    # Visualize confidence distributions
    plt.figure(figsize=(12, 6))
    
    # Plot confidence distributions for correct vs incorrect predictions
    correct_confidences = np.max(all_probs[y_pred == y_true], axis=1)
    incorrect_confidences = np.max(all_probs[y_pred != y_true], axis=1)
    
    plt.hist(correct_confidences, alpha=0.7, label='Correct Predictions', bins=20, range=(0, 1))
    if len(incorrect_confidences) > 0:
        plt.hist(incorrect_confidences, alpha=0.7, label='Incorrect Predictions', bins=20, range=(0, 1))
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return class_analysis

def visualize_activations(model, test_dataset, layer_name='conv5_block3_out', num_samples=3, num_filters=16):
    """
    Visualize activations of specific layers for insights into what the model sees
    """
    # Find the requested layer
    target_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer_name in layer.name:
            target_layer = layer
            break
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                if layer_name in sublayer.name:
                    target_layer = sublayer
                    break
            if target_layer is not None:
                break
    
    if target_layer is None:
        print(f"Layer {layer_name} not found. Available layers:")
        for layer in model.layers:
            if not isinstance(layer, tf.keras.layers.InputLayer):
                print(f"- {layer.name}")
        return
    
    # Create a model that outputs the target layer's activations
    activation_model = tf.keras.Model(
        inputs=model.input[0],  # Just use the ResNet input
        outputs=target_layer.output
    )
    
    # Get sample images
    for inputs, _ in test_dataset:
        if isinstance(inputs, tuple):
            images = inputs[0]
        else:
            images = inputs
        images = images[:num_samples]
        break
    
    # Get activations
    activations = activation_model.predict(images)
    
    # Visualize the activations of a few filters
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(15, 10))
        
        # Show the original image
        plt.subplot(num_filters//4 + 1, 4, 1)
        img = images[i].numpy()
        img = img[..., ::-1]  # BGR to RGB
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize to [0,1]
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')
        
        # Display activations for selected filters
        for j in range(min(num_filters, activations.shape[-1])):
            plt.subplot(num_filters//4 + 1, 4, j + 2)
            activation = activations[i, :, :, j]
            plt.imshow(activation, cmap='viridis')
            plt.title(f"Filter {j}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Activations for sample {i+1} at layer '{layer_name}'", y=1.02)
        plt.show()

if __name__ == "__main__":
    
    # Create datasets
    _, _, test_dataset, num_classes, class_indices = create_train_val_test_datasets(
        "dataset/train", "dataset/val", "dataset/test", batch_size=32
    )

    # Load the trained model
    model = tf.keras.models.load_model('hybrid_model_best.keras')

    # Analyze model
    analyze_model(model, test_dataset, class_indices)
    
    # Visualize activations
    visualize_activations(model, test_dataset, layer_name='conv5_block3_out', num_samples=3, num_filters=16)
