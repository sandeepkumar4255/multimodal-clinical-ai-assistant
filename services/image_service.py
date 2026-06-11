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


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def load_model():

    device = torch.device("cpu")

    model = TinyCNN()

    model.load_state_dict(
        torch.load(
            "models/xray_model.pth",
            map_location=device
        )
    )

    model.eval()

    return model


def analyze_xray(image_path):

    try:

        device = torch.device("cpu")

        model = load_model()

        image = Image.open(image_path).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0)

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

        class_names = {
            0: "NORMAL",
            1: "PNEUMONIA"
        }

        prediction = class_names[predicted_class]

        print("=" * 50)
        print("Image:", image_path)
        print("Prediction:", prediction)
        print("Confidence:", confidence.item())
        print("=" * 50)

        return {
            "prediction": prediction,
            "confidence": round(
                float(confidence.item()),
                2
            )
        }

    except Exception as e:

        print("XRAY ERROR:", str(e))

        return {
            "prediction": "ERROR",
            "confidence": 0.0
        }
