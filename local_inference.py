import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import os
import sys
import pandas as pd # Added for CoffeeDataset
from torch.utils.data import Dataset # Added for CoffeeDataset

# Define the base path for your files (this was originally in cell 'xAt0OF6eLomn')
path = '/content/drive/MyDrive/IMAGE CLASSIFICATION/IMAGE CLASSIFICATION'

# Define class paths (originally in G2__Qiwihn8f)
test_classes = os.path.join(path, "test_classes.csv")
train_classes = os.path.join(path, "train_classes.csv")

# --- Define the CoffeeDataset class (originally in T6Ons9WEm-ft) ---
class CoffeeDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["id"]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        labels = torch.tensor([row["miner"], row["rust"], row["phoma"]], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels

# --- Define Image Transformations (originally in n7-RQQjgnEu-) ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Create test_dataset (originally in Q6L8vJIhnTWq) ---
try:
    test_dataset = CoffeeDataset(
        csv_file=test_classes,
        image_dir=os.path.join(path, "coffee-leaf-diseases/test/images"),
        transform=test_transform
    )
except FileNotFoundError:
    print(f"Error: Test dataset CSV or image directory not found. Please ensure files exist at {test_classes} and {os.path.join(path, 'coffee-leaf-diseases/test/images')}")
    test_dataset = None # Set to None if creation fails

# --- 1. Define the Model Architecture (Must match training architecture) ---
# Initialize the pre-trained EfficientNet-B0 model
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# Modify the classifier head for your specific task (3 output classes)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 3)

# --- 2. Load the Saved Model Weights ---
# For Colab, use the full path from Google Drive. For local use, this might be just the filename.
model_path = os.path.join(path, "coffee_multilabel_fine_tuned.pth")

# Set device to CPU for local inference unless GPU is explicitly available and configured
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

try:
    # Load the state dictionary, mapping to the appropriate device
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model successfully loaded from: {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please ensure it's in the correct directory.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    sys.exit(1)

# Set the model to evaluation mode
model.eval()

# --- 3. Define Target Names ---
target_names = ['miner', 'rust', 'phoma']

# --- 4. Inference Function ---
def predict_image(image_path, model, transform, device, target_names):
    try:
        # Load and preprocess the image
        new_image = Image.open(image_path).convert("RGB")
        transformed_image = transform(new_image)
        # Add a batch dimension (model expects batches of images)
        input_tensor = transformed_image.unsqueeze(0).to(device)

        # Make a prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.sigmoid(output)
            predicted_labels = (probabilities > 0.5).int()

        print(f"\nPrediction for image '{os.path.basename(image_path)}':")
        for i, prob in enumerate(probabilities[0]):
            print(f"  {target_names[i]}: Probability = {prob:.4f}, Predicted = {predicted_labels[0, i]}")

        return probabilities, predicted_labels
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Please verify the path.")
        return None, None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None, None

# --- 5. Example Usage (Replace with your local image path) ---
# For demonstration in Colab, we'll use a sample image from the test_dataset
if test_dataset is not None and len(test_dataset) > 0:
    sample_image_id = test_dataset.df['id'].iloc[0] # Using the first image for example
    local_image_path = os.path.join(test_dataset.image_dir, f"{sample_image_id}.jpg")
else:
    print("Warning: test_dataset not found or is empty. Cannot determine a sample image path.")
    print("Please ensure test_dataset is initialized before running this example.")
    local_image_path = "path/to/your/local_image.jpg" # Fallback for clarity

if os.path.exists(local_image_path):
    predict_image(local_image_path, model, test_transform, device, target_names)
else:
    print(f"\nWarning: Cannot run example inference. Please update 'local_image_path' to a valid image file on your machine or in Colab: {local_image_path}")
    print("Make sure the image file exists at the specified path.")