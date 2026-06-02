import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


_model = None


def get_model():

    global _model

    if _model is None:

        model = TinyCNN()

        model.load_state_dict(
            torch.load(
                "models/xray_model.pth",
                map_location=device
            )
        )

        model.to(device)
        model.eval()

        _model = model

    return _model


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def analyze_xray(image_path):

    try:

        model = get_model()

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0)

        image = image.to(device)

        with torch.no_grad():

            outputs = model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                1
            )

        if prediction.item() == 0:

            return {
                "prediction": "NORMAL",
                "confidence": round(
                    confidence.item(),
                    2
                )
            }

        return {
            "prediction": "PNEUMONIA",
            "confidence": round(
                confidence.item(),
                2
            )
        }

    except Exception as e:

        return {
            "prediction": "ERROR",
            "confidence": 0.0,
            "message": str(e)
        }