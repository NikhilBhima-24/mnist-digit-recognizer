# MNIST Digit Recognizer (Deep Learning)

A streamlined Deep Learning project that builds, trains, and evaluates a Feedforward Neural Network to recognize handwritten digits. This project serves as a foundational demonstration of implementing Artificial Neural Networks using **TensorFlow** and **Keras**.

## Features
* **Automated Data Pipeline:** Uses Keras datasets to automatically fetch and load the MNIST dataset (60,000 training images and 10,000 testing images).
* **Data Normalization:** Preprocesses image pixel values (0-255) to a normalized scale (0.0-1.0) to improve training efficiency and model convergence.
* **Neural Network Architecture:** Implements a `Sequential` model utilizing `Flatten` and `Dense` layers with ReLU and Softmax activation functions.
* **Model Export:** Automatically saves the trained model architecture and weights as a `.keras` file for future inference or deployment.

## Tech Stack
* **Python 3.x**
* **TensorFlow & Keras** (Deep Learning Framework)
* **NumPy** (Numerical Computation)

## Installation & Setup

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/NikhilBhima-24/mnist-digit-recognizer.git](https://github.com/NikhilBhima-24/mnist-digit-recognizer.git)
   cd mnist-digit-recognizer
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: TensorFlow is a large package and may take a moment to download depending on your internet connection).*

## Usage

Run the script directly from your terminal. The script will output the training progress, display the final accuracy on the unseen test dataset, and save the model file to your directory.

```bash
python digit_recognizer.py
```

### Expected Output
The model should achieve approximately **96% - 98% accuracy** on the test set within just 2 epochs of training.
