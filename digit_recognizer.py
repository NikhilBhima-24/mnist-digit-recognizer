import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

def main():
    print("TensorFlow Version:", tf.__version__)
    print("Loading MNIST dataset...")
    
    # 1. Load the dataset automatically via Keras
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Normalize the pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(f"Training data shape: {x_train.shape}")
    print(f"Testing data shape: {x_test.shape}")

    # 3. Build a simple Feedforward Neural Network
    print("\nBuilding the Neural Network model...")
    model = Sequential([
        Flatten(input_shape=(28, 28)),          # Flatten the 28x28 images into a 1D array
        Dense(128, activation='relu'),          # Hidden layer with ReLU activation
        Dense(10, activation='softmax')         # Output layer with 10 classes (digits 0-9)
    ])

    # 4. Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. Train the model (Keep epochs low for a fast run)
    print("\nStarting training process...")
    model.fit(x_train, y_train, epochs=2, validation_split=0.1)

    # 6. Evaluate the model on test data
    print("\nEvaluating model on test data...")
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

    # 7. Save the trained model
    model_filename = 'mnist_simple_model.keras'
    model.save(model_filename)
    print(f"Model successfully saved as '{model_filename}'")

if __name__ == "__main__":
    main()
