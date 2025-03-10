from data import create_train_val_test_datasets

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Calculate ROC curves for multiclass (one-vs-rest)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def evaluate_model(model, test_dataset, class_names):
    """
    Enhanced evaluation with comprehensive metrics and visualizations
    """
    # Get true labels and predictions
    y_true = []
    y_pred = []
    all_probs = []

    # Process batches
    for batch in test_dataset:
        inputs, labels = batch
        # Ensure input is properly formatted for the model
        if isinstance(inputs, tuple):
            # Already in the right format (resnet_input, vit_input)
            predictions = model.predict(inputs, verbose=0)
        else:
            # Need to split the input for both branches
            predictions = model.predict([inputs, inputs], verbose=0)
        
        # Extract labels and predictions
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
        all_probs.extend(predictions)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_probs = np.array(all_probs)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # Class performance visualization
    class_accuracy = np.zeros(len(class_names))
    for i in range(len(class_names)):
        idx = (y_true == i)
        if np.sum(idx) > 0:  # Avoid division by zero
            class_accuracy[i] = np.sum(y_pred[idx] == i) / np.sum(idx)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=class_accuracy)
    plt.title('Accuracy by Class')
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Binarize the true labels
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return y_true, y_pred, all_probs

if __name__ == "__main__":
    
    # Create datasets
    _, _, test_dataset, num_classes, class_indices = create_train_val_test_datasets(
        "dataset/train", "dataset/val", "dataset/test", batch_size=32
    )

    # Load the trained model
    model = tf.keras.models.load_model('hybrid_model_best.keras')

    # Evaluate model
    evaluate_model(model, test_dataset)
