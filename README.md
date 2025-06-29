# QASA-ViT(ViQASA): A Quantum Adaptive Self-Attention Vision Transformer

## Overview

This project explores and compares different Vision Transformer architectures for image classification. It includes implementations for:

1.  **Vision Transformer**: A standard implementation of the Vision Transformer (ViT) model.
2.  **ViQASA (Vision Hybrid Transformer)**: A custom hybrid classical-quantum Vision Transformer that leverages a quantum-inspired attention mechanism using Pennylane.

The models are trained and evaluated on the MNIST and Fashion-MNIST datasets.

## Directory Structure

```
.
├── viqasa_mnist.py            # Train ViQASA on MNIST
├── viqasa_fashion_mnist.py    # Train ViQASA on Fashion-MNIST
├── transformer_mnist.py         # Train standard ViT on MNIST
├── transformer_fashion_mnist.py # Train standard ViT on Fashion-MNIST
└── results/                     # Contains logs from training runs
```

## Models

### Vision Transformer
A standard implementation of the Vision Transformer architecture, adapted for the MNIST/Fashion-MNIST datasets.

### ViQASA (Vision Hybrid Transformer)
This model replaces one of the transformer encoder layers with a `QuantumEncoderLayer`. This layer incorporates a quantum circuit defined with Pennylane to introduce quantum-inspired computational elements into the network.

## Requirements

To run the experiments, you need to install the following Python libraries. It is recommended to use a virtual environment.

```bash
pip install torch torchvision pennylane matplotlib tqdm
```

You can also create a `requirements.txt` file with the following content:
```
torch
torchvision
pennylane
matplotlib
tqdm
```
And install it using `pip install -r requirements.txt`.

## Usage

Each script can be run directly from the command line. The scripts will handle data downloading, model training, evaluation, and saving the results.

### MNIST Dataset

*   **Train Vision Transformer:**
    ```bash
    python transformer_mnist.py
    ```

*   **Train ViQASA:**
    ```bash
    python viqasa_mnist.py
    ```

### Fashion-MNIST Dataset

*   **Train Vision Transformer:**
    ```bash
    python transformer_fashion_mnist.py
    ```

*   **Train ViQASA:**
    ```bash
    python viqasa_fashion_mnist.py
    ```

## Outputs

Running a training script will generate the following outputs:

*   **Checkpoints directory**: e.g., `checkpoints_mnist/` or `checkpoints_fashion_mnist/`
    *   `best_model.pth`: The model weights with the best validation loss.
    *   `history.json`: A JSON file containing the training and validation loss/accuracy history.
*   **Log file**: e.g., `mnist_training.log` or `fashion_mnist_training.log`
    *   Contains detailed logs of the training process, including configuration and per-epoch performance. The user can move these to the `results` folder.
*   **Loss Plot**: e.g., `loss_plot_mnist.png` or `loss_plot_fashion_mnist.png`
    *   A plot showing the training and validation loss curves over epochs.

The `results` directory is intended to store the log files for different runs. For example, `results/transformer_mnist_training.log` which you provided. 