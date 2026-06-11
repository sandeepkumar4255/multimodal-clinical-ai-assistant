import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class TinyCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    r"C:\multimodal_dataset_backup\chest_xray\train",
    transform=transform
)
print("Total Images:", len(train_dataset))
print("Class Mapping:", train_dataset.class_to_idx)
from collections import Counter

labels = [label for _, label in train_dataset.samples]

distribution = Counter(labels)

print("NORMAL Images:", distribution[0])
print("PNEUMONIA Images:", distribution[1])

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TinyCNN().to(device)

class_weights = torch.tensor(
    [3875 / 1341, 1.0],
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(15):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}:",
        running_loss
    )

torch.save(
    model.state_dict(),
    "models/xray_model.pth"
)
print(train_dataset.class_to_idx)
print("Model Saved")
