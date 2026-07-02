import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import os
# --- Configuration (UPDATED FOR CLOUD DEPLOYMENT) ---
# Keep the model file in the same directory as app.py
model_filename = "coffee_multilabel_fine_tuned.pth"
model_path = model_filename  # Looks in the root folder of your project

target_names = ['miner', 'rust', 'phoma']
# Cloud instances usually run on CPU unless you pay for heavy GPU tiers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Image Transformations (must match training/inference) ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Model Definition and Loading ---
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_model():
    # Initialize the pre-trained EfficientNet-B0 model
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Modify the classifier head for your specific task (3 output classes)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 3)

    model.to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        st.success(f"Model '{model_filename}' loaded successfully on {device}!")
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Please ensure it's in the correct Google Drive directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()
    return model

model = load_model()

# --- Inference Function ---
def predict_image(image, model, transform, device, target_names):
    try:
        transformed_image = transform(image)
        input_tensor = transformed_image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.sigmoid(output)
            predicted_labels = (probabilities > 0.5).int()

        results = {}
        for i, prob in enumerate(probabilities[0]):
            results[target_names[i]] = {
                "probability": f"{prob:.4f}",
                "predicted": bool(predicted_labels[0, i].item())
            }
        return results
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(
    page_title="Coffee Leaf Disease Classifier",
    page_icon=":coffee:"
)

st.title("☕ Coffee Leaf Disease Classifier")
st.write("Upload an image of a coffee leaf to predict diseases (miner, rust, phoma).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")
    st.write("Classifying...")

    predictions = predict_image(image, model, test_transform, device, target_names)

    if predictions:
        st.subheader("Prediction Results:")
        for disease, data in predictions.items():
            if data["predicted"]:
                st.markdown(f"- **{disease.capitalize()}**: Detected (Probability: {data['probability']})")
            else:
                st.markdown(f"- {disease.capitalize()}: Not Detected (Probability: {data['probability']})")

st.caption("Model trained on coffee leaf disease dataset.")