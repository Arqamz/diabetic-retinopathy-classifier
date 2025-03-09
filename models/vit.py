# models/vit.py
import tensorflow as tf

class ViTFeatureExtractorLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = 16
        self.hidden_size = 768
        self.num_patches = (224 // self.patch_size) ** 2
        
        # Convolutional layer to create patch embeddings
        self.patch_embedding = tf.keras.layers.Conv2D(
            filters=self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding='valid',
            name='patch_embedding'
        )
        
        # Class token to be appended to the patch embeddings
        self.cls_token = self.add_weight(
            shape=(1, 1, self.hidden_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name='cls_token'
        )
        
        # Positional embeddings for the patches and class token
        self.position_embedding = self.add_weight(
            shape=(1, self.num_patches + 1, self.hidden_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name='position_embedding'
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(self.hidden_size, num_heads=12, mlp_dim=3072)
            for _ in range(6)
        ]
        
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs):
        # Normalize inputs
        x = tf.cast(inputs, tf.float32) / 127.5 - 1
        
        # Create patch embeddings
        patches = self.patch_embedding(x)
        batch_size = tf.shape(patches)[0]
        patches = tf.reshape(patches, [batch_size, -1, self.hidden_size])
        
        # Append class token to patch embeddings
        cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
        patches = tf.concat([cls_tokens, patches], axis=1)
        
        # Add positional embeddings
        patches = patches + self.position_embedding
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            patches = transformer_block(patches)
            
        # Apply layer normalization
        patches = self.layer_norm(patches)
        
        # Return the class token representation
        return patches[:, 0]

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        # Multi-head attention layer
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate
        )
        
        # MLP (Feed-forward neural network)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(dropout_rate),
        ])
        
        # Layer normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs):
        # Apply multi-head attention
        attention_output = self.attention(inputs, inputs)
        
        # Add and normalize
        x = self.layernorm1(inputs + attention_output)
        
        # Apply MLP and normalize
        mlp_output = self.mlp(x)
        return self.layernorm2(x + mlp_output)
