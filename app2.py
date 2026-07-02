from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import io

app = FastAPI()

# 1. Enable CORS so your React frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your React app's actual URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration & Model Loading ---
model_filename = "coffee_multilabel_fine_tuned.pth"
target_names = ['miner', 'rust', 'phoma']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load model once at startup
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 3)
model.load_state_dict(torch.load(model_filename, map_location=device))
model.eval()
model.to(device)


@app.get("/")
def root():
    return {"status": "Healthy", "model": "Coffee Leaf Classifier"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image bytes
        request_object_content = await file.read()
        image = Image.open(io.BytesIO(request_object_content)).convert("RGB")

        # Transform and predict
        transformed_image = test_transform(image)
        input_tensor = transformed_image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.sigmoid(output)
            predicted_labels = (probabilities > 0.5).int()

        # Format JSON response for React
        results = []
        for i, prob in enumerate(probabilities[0]):
            results.append({
                "disease": target_names[i],
                "probability": float(f"{prob:.4f}"),
                "predicted": bool(predicted_labels[0, i].item())
            })

        return {"success": True, "predictions": results}

    except Exception as e:
        return {"success": False, "error": str(e)}