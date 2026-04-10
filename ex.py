import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# ===== SETTINGS =====
DATASET_PATH = "roi_output"
BATCH_SIZE = 2
EPOCHS = 10
IMG_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===== DATASET =====
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== FEATURE EXTRACTION =====
def get_features(x):
    x_pad = F.pad(x, (1,1,1,1), mode='reflect')

    n1 = x_pad[:, :, :-2, :-2]
    n2 = x_pad[:, :, :-2, 1:-1]
    n3 = x_pad[:, :, :-2, 2:]
    n4 = x_pad[:, :, 1:-1, :-2]
    n5 = x_pad[:, :, 1:-1, 2:]
    n6 = x_pad[:, :, 2:, :-2]
    n7 = x_pad[:, :, 2:, 1:-1]
    n8 = x_pad[:, :, 2:, 2:]

    features = torch.cat([
        (x - n1), (x - n2), (x - n3), (x - n4),
        (x - n5), (x - n6), (x - n7), (x - n8)
    ], dim=1)  # (B, 8, H, W)

    return features

# ===== PURE ANN MODEL =====
class PureANN(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 8 * IMG_SIZE * IMG_SIZE  

        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 3)  # 3 classes
        )

    def forward(self, x):
        x = get_features(x)             # (B, 8, H, W)
        x = x.view(x.size(0), -1)       # flatten
        x = self.fc(x)
        return x

# ===== MODEL =====
model = PureANN().to(device)

# ===== LOSS & OPTIMIZER =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== TRAINING =====
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

# ===== EVALUATION =====
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total

print("\n===== PURE ANN RESULT =====")
print(f"Accuracy: {accuracy:.4f}")