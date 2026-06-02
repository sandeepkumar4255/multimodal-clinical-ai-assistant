import torch
from torchvision import transforms, models
from PIL import Image
from torch import nn

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load Model
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "models/xray_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def analyze_xray(image_path):

    try:

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