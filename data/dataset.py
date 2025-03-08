# data/dataset.py
import os
import tensorflow as tf

def create_dataset(directory, target_size=(224, 224), batch_size=32, training=False):
    """
    Create a tf.data.Dataset from image directory that provides both CNN and ViT inputs
    """
    # Get class names (subdirectories)
    class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    class_indices = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    
    # List all image files
    image_files = []
    labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(class_dir, filename))
                labels.append(class_indices[class_name])
    
    # Create dataset from file paths
    def process_path(file_path, label):
        # Read and decode image
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        
        # Create CNN input - ResNet50 preprocessing
        cnn_img = tf.cast(img, tf.float32)
        # Replicate ResNet preprocessing
        cnn_img = cnn_img[..., ::-1]  # RGB to BGR
        cnn_img = tf.keras.applications.resnet50.preprocess_input(cnn_img)
        
        # Create ViT input - Simple scaling to [0,1]
        vit_img = tf.cast(img, tf.float32) / 255.0
        
        # One-hot encode the label
        label_one_hot = tf.one_hot(label, depth=num_classes)
        
        return (cnn_img, vit_img), label_one_hot
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))
    
    # Shuffle the dataset if training
    if training:
        dataset = dataset.shuffle(buffer_size=len(image_files), reshuffle_each_iteration=True)
    
    # Process the images and batch the dataset
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Return dataset and metadata
    return dataset, len(image_files), num_classes, class_indices

def create_train_val_test_datasets(train_dir, val_dir, test_dir, batch_size):
    
    train_dataset, _, num_classes, class_indices = create_dataset(
        train_dir, batch_size=batch_size, training=True
    )
    val_dataset, _, _, _ = create_dataset(
        val_dir, batch_size=batch_size, training=False
    )
    test_dataset, _, _, _ = create_dataset(
        test_dir, batch_size=batch_size, training=False
    )
    return train_dataset, val_dataset, test_dataset, num_classes, class_indices

if __name__ == "__main__":
    train_dir = "dataset/train"
    val_dir = "dataset/val"
    test_dir = "dataset/test"
    batch_size = 32

    train_dataset, val_dataset, test_dataset, num_classes, class_indices = create_train_val_test_datasets(
        train_dir, val_dir, test_dir, batch_size
    )

    print("Data has been loaded.")
    print(f"Number of classes: {num_classes}")
    print(f"Class indices: {class_indices}")