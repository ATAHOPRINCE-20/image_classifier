import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image

LABELS = ['miner', 'rust', 'phoma']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(LABELS))

model.load_state_dict(torch.load("coffee_multilabel_fine_tuned.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()
preprocess = weights.transforms()

def predict(image: Image.Image):
    image = image.convert("RGB")
    x = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out)[0].cpu().numpy()

    return [
        {"class": LABELS[i], "probability": float(probs[i])}
        for i in range(len(LABELS))
    ]