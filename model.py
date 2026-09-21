import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
EPOCHS = 30
BATCH_SIZE = 64
HIDDEN_SIZES = 256
DROPOUT_RATE = 0.4
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==========================================
# 2. Data Preprocessing & Loading
# ==========================================
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 50,000 training data, 10,000 validation data, 10,000 testing data
full_train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=test_transform, download=True)

# split the data into training and validation data 
train_dataset, val_dataset = random_split(full_train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. Model Architecture
# ==========================================
class FashionMLP(nn.Module):
    def __init__(self):
        super(FashionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, HIDDEN_SIZES),
            nn.BatchNorm1d(HIDDEN_SIZES),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_SIZES, 10)
        )

    def forward(self, x):
        return self.network(x)

model = FashionMLP().to(DEVICE)

# ==========================================
# 4. Loss and Optimizer
# ==========================================
criterion = nn.CrossEntropyLoss()
# 加入 L2 正则化 (Weight Decay)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ==========================================
# 5. Training and Evaluation Loop
# ==========================================
train_losses = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')

print("Starting Training...")
for epoch in range(EPOCHS):
    
    # --- Training Phase ---
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    epoch_train_loss = train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = (val_correct / len(val_loader.dataset)) * 100
    
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    # --- Checkpointing (Save Best Model) ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'model_proj1.pth')

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
          f"Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | "
          f"Val Acc: {epoch_val_acc:.2f}%")

# ==========================================
# 6. Plotting Evaluation Results
# ==========================================
print("\nTraining completed. Generating evaluation plots...")
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(12, 5))

# Plot 1: Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss', color='#1f77b4', linewidth=2, marker='o', markersize=4)
plt.plot(epochs_range, val_losses, label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--', marker='s', markersize=4)
plt.title('Training & Validation Loss', fontsize=12, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Plot 2: Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_accuracies, label='Val Accuracy', color='#2ca02c', linewidth=2, marker='^', markersize=4)
plt.title('Validation Accuracy Progression', fontsize=12, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300)
print("Plots saved as 'evaluation_results.png'.")

# ==========================================
# 7. Final Test Evaluation (Blind Test)
# ==========================================
print("\n=== Final Test Evaluation ===")

# Reload the best model weights
model.load_state_dict(torch.load('model_proj1.pth', weights_only=True))
model.eval()

test_loss = 0.0
test_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        test_correct += (preds == labels).sum().item()

final_test_loss = test_loss / len(test_loader.dataset)
final_test_acc = (test_correct / len(test_loader.dataset)) * 100

print(f"Total Test Samples: {len(test_loader.dataset)}")
print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")
