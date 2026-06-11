import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms


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


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TinyCNN().to(device)

model.load_state_dict(
    torch.load(
        "models/xray_model.pth",
        map_location=device
    )
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def analyze_xray(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = predicted.item()

    print("Image:", image_path)
    print("Predicted Class:", predicted_class)
    print("Confidence:", confidence.item())

    if predicted_class == 0:

        return {
            "prediction": "NORMAL",
            "confidence": round(
                float(confidence.item()),
                2
            )
        }

    return {
        "prediction": "PNEUMONIA",
        "confidence": round(
            float(confidence.item()),
            2
        )
    }