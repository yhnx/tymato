from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import torch
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

# Load the full model
model = torch.load(r"saved_model/trained_model_complete_v1.pth", map_location=torch.device("cpu"))
model.eval()  # Set to evaluation mode

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

@app.get("/")
async def root():
    return "Hello World"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    bytes_data = await file.read()  # Read image as bytes
    img = read_image(bytes_data)  # Convert bytes to image
    img_tensor = preprocess_image(img)  # Preprocess image
    with torch.no_grad():
        output = model(img_tensor)  # Get model predictions
        _, predicted = torch.max(output, 1)  # Find the class with highest probability
        return {"class": CLASSES[predicted.item()]}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
