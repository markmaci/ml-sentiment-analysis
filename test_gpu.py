import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Set memory growth for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set for GPU")
    except RuntimeError as e:
        print(e)

# Log device placement
tf.debugging.set_log_device_placement(True)

# Simple test model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(100,), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Test data
import numpy as np
X_train = np.random.rand(1000, 100)
y_train = np.random.randint(0, 2, 1000)

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
