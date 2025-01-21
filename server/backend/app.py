from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T

# Constants
IMAGE_SIZE = 256
CLASSES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites (Two-Spotted Spider Mite)",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy",
    "Powdery Mildew",
]

# Creating a neural network using the ResNet-50 architecture
class RESNET(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(RESNET, self).__init__()
        self.backbone = getattr(models, "resnet50")(pretrained=True)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


model = RESNET(
    num_classes=len(CLASSES),
    freeze_backbone=True,  # freeze pretrained weights initially
)

# Load the saved model weights
model.load_state_dict(torch.load(r"saved_model/trained_model_v1.pth", map_location=torch.device("cpu")))

# Set the model to evaluation mode
model.eval()

# Define image preprocessing
def preprocess_image(image):
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Matching the image size used during training
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Adding batch dimension


# Define function to read the image
def read_image(bytes_data):
    img_pil = Image.open(BytesIO(bytes_data))
    return img_pil


app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="server/static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="server/frontend")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Renders the main HTML page using a Jinja2 template.
    """
    return templates.TemplateResponse("main.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Handles image upload, performs prediction, and returns the result.
    """
    bytes_data = await file.read()  # Read image as bytes
    img = read_image(bytes_data)  # Convert bytes to image
    img_tensor = preprocess_image(img)  # Preprocess image
    with torch.no_grad():
        output = model(img_tensor)  # Get model predictions
        probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
        max_prob, predicted = torch.max(probabilities, 1)  # Get highest probability and class

        # Get class name and confidence percentage
        predicted_class = CLASSES[predicted.item()]
        confidence = max_prob.item() * 100  # Convert to percentage

        return {
            "class": predicted_class,
            "confidence": f"{confidence:.2f}%",
        }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
