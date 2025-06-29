import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pennylane as qml
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Quantum config
n_qubits = 8
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits + 1)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)
    for i in range(n_qubits):
        qml.RX(weights[0, i], wires=i)
        qml.RZ(weights[1, i], wires=i)
    for l in range(1, n_layers):
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
            qml.RY(weights[l, i], wires=i)
            qml.RZ(weights[l, i], wires=i)
        qml.CNOT(wires=[n_qubits - 1, n_qubits])
        qml.RY(weights[l, -1], wires=n_qubits)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class QuantumLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, n_qubits)
        self.norm = nn.LayerNorm(n_qubits)
        self.output_proj = nn.Linear(n_qubits, output_dim)
        self.weight_shape = (n_layers, n_qubits + 1)
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, {"weights": self.weight_shape})
        self.input_proj.apply(init_weights)
        self.output_proj.apply(init_weights)

    def forward(self, x, timestep):
        x_proj = self.norm(torch.tanh(self.input_proj(x)))
        outputs = [self.qlayer((x_proj[i] + timestep).cpu()).to(x.device) for i in range(x.size(0))]
        return self.output_proj(torch.stack(outputs))

class QuantumEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.v_quantum = QuantumLayer(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        b, s, f = x.shape
        dummy_time = torch.tensor(0.0, device=x.device)
        x_flat = x.view(b * s, f)
        q_out = self.v_quantum(x_flat, dummy_time).view(b, s, f)
        return self.norm2(q_out + self.ffn(q_out))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VisionHybridTransformer(nn.Module):
    def __init__(self, img_size=28, hidden_dim=128, num_layers=4, dropout=0.1, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Sequential(
            nn.Linear(img_size * img_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=1)
        self.encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dropout=dropout)
             for _ in range(num_layers - 1)] + [QuantumEncoderLayer(hidden_dim, dropout)]
        )
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # x: [B, 1, 28, 28]
        x = self.flatten(x)  # [B, 784]
        x = self.embedding(x).unsqueeze(1)  # [B, 1, hidden]
        x = self.pos_enc(x)
        for layer in self.encoder:
            x = layer(x)
        return self.output(x[:, 0])

def get_fashion_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train = datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)
    val = datasets.FashionMNIST(root=".", train=False, download=True, transform=transform)
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(val, batch_size=batch_size)

if __name__ == "__main__":
    # 1. Setup
    epochs = 30
    lr = 1e-4
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs("checkpoints_fashion_mnist", exist_ok=True)
    best_model_path = "checkpoints_fashion_mnist/best_model.pth"
    log_path = "fashion_mnist_training.log"

    # Setup logger
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()

    logger.info("Starting training for Fashion MNIST with VisionHybridTransformer.")
    logger.info(f"Configuration: epochs={epochs}, lr={lr}, batch_size={batch_size}, device={device}")

    # 2. Data
    train_loader, val_loader = get_fashion_mnist_loaders(batch_size=batch_size)
    
    # 3. Model, Loss, Optimizer
    model = VisionHybridTransformer(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 4. Training Loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            
            train_pbar.set_postfix(loss=f"{train_loss / (train_pbar.n + 1):.4f}", acc=f"{train_correct / train_total:.4f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for x, y in val_pbar:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

                val_pbar.set_postfix(loss=f"{val_loss / (val_pbar.n + 1):.4f}", acc=f"{val_correct / val_total:.4f}")
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # 5. Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved to {best_model_path}")

    logger.info("Training finished.")
    # 6. Plotting
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plot_path = "loss_plot_fashion_mnist.png"
    plt.savefig(plot_path)
    logger.info(f"Loss plot saved to {plot_path}") 