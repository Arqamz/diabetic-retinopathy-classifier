from data import create_train_val_test_datasets

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def explain_model(model, test_dataset, class_names, num_samples=5):
    """
    Custom Grad-CAM implementation for hybrid models that manually computes
    the gradients rather than relying on tf_keras_vis
    """
    
    # Find the last convolutional layer in ResNet branch
    last_conv_layer_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # Handle nested models (like ResNet)
            for nested_layer in layer.layers:
                if 'conv5_block3_out' in nested_layer.name:
                    last_conv_layer_name = nested_layer.name
                    break
        elif 'conv5_block3_out' in layer.name:
            last_conv_layer_name = layer.name
            break
    
    if last_conv_layer_name is None:
        # Fallback to any convolutional layer
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                print(f"Using layer: {last_conv_layer_name}")
                break
    
    # Get sample inputs and corresponding labels
    for batch in test_dataset:
        sample_images, sample_labels = batch
        if isinstance(sample_images, tuple):
            resnet_inputs, vit_inputs = sample_images
        else:
            resnet_inputs = vit_inputs = sample_images
            
        resnet_inputs = resnet_inputs[:num_samples]
        vit_inputs = vit_inputs[:num_samples]
        sample_labels = sample_labels[:num_samples]
        break
    
    # Function to generate Grad-CAM for a single image
    def generate_gradcam(img_array, target_class_idx):
        # Create a model that outputs both the last conv layer outputs and the final prediction
        grad_model = tf.keras.models.Model(
            [model.inputs[0], model.inputs[1]],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Cast inputs to float32
        img_array = tf.cast(img_array, tf.float32)
        
        with tf.GradientTape() as tape:
            # Add a batch dimension and process the image
            resnet_input = tf.expand_dims(img_array[0], axis=0)
            vit_input = tf.expand_dims(img_array[1], axis=0)
            
            # Get model predictions and feature maps
            conv_outputs, predictions = grad_model([resnet_input, vit_input])
            target_class_output = predictions[:, target_class_idx]
        
        # Gradient of the target class with respect to the conv layer output
        grads = tape.gradient(target_class_output, conv_outputs)
        
        # Global average pooling of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the incoming gradients
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
        
        # Apply ReLU to focus on features that have a positive influence on the target class
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        
        return heatmap.numpy()
    
    # Make predictions and generate heatmaps
    predictions = model.predict([resnet_inputs, vit_inputs], verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(sample_labels.numpy(), axis=1)
    
    # Generate heatmaps for each sample
    heatmaps = []
    for i in range(num_samples):
        # Use the predicted class as target
        target_class = pred_classes[i]
        
        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam([resnet_inputs[i], vit_inputs[i]], target_class)
        heatmaps.append(heatmap)
    
    # Visualization
    plt.figure(figsize=(20, 4 * num_samples))
    for i in range(num_samples):
        # Process original image
        img = resnet_inputs[i].numpy()
        # Convert BGR to RGB if using ResNet preprocessing
        img = img[..., ::-1]  # BGR to RGB
        # Normalize to [0,1] for visualization
        img = (img - img.min()) / (img.max() - img.min())
        
        # Process heatmap
        heatmap = heatmaps[i]
        # Resize heatmap to match input image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        # Convert to uint8 and colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(np.uint8(255 * img), 0.6, heatmap, 0.4, 0)
        
        # Create subplots
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img)
        plt.title(f"True: {class_names[true_classes[i]]}")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        plt.title(f"Pred: {class_names[pred_classes[i]]}\nConfidence: {predictions[i][pred_classes[i]]*100:.1f}%")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return heatmaps

if __name__ == "__main__":
    
    # Create datasets
    _, _, test_dataset, num_classes, class_indices = create_train_val_test_datasets(
        "dataset/train", "dataset/val", "dataset/test", batch_size=32
    )

    # Load the trained model
    model = tf.keras.models.load_model('hybrid_model_best.keras')

    # Generate heatmaps for 5 samples
    explain_model(model, test_dataset, class_indices, num_samples=5)
