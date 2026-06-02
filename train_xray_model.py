from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
import torch

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
train_dataset = datasets.ImageFolder(
    "datasets/chest_xray/train",
    transform=transform
)

# Data Loader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Model
model = models.resnet18(weights="DEFAULT")

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

print("Training Started...")
print(f"Device: {device}")
print(f"Total Images: {len(train_dataset)}")

# Train (1 Epoch for Testing)
for epoch in range(1):

    model.train()

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

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

        if (batch_idx + 1) % 20 == 0:

            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx+1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    print(
        f"Epoch {epoch+1} Completed | "
        f"Average Loss: {running_loss/len(train_loader):.4f}"
    )

# Save Model
torch.save(
    model.state_dict(),
    "models/xray_model.pth"
)

print("Model Saved Successfully!")
print("Saved at: models/xray_model.pth")