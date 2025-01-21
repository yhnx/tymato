from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image



import torch
import torch.nn as nn
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
    "Powdery Mildew"
]

NUM_CLASSES = len(CLASSES)

# For scaling and resizing incase external images are used
class ScaleAndResizeLayer(nn.Module):
    def __init__(self, size=(IMAGE_SIZE, IMAGE_SIZE), scale=1.0/255.0):
        super(ScaleAndResizeLayer, self).__init__()
        self.resize = T.Resize(size)  # Resize to the target size
        self.scale = scale           # Scale factor

    def forward(self, x):
        # Resize and scale
        x = self.resize(x)
        x = x * self.scale  # Scale the pixel values
        return x
    
#Augmenting the input the data for better modelling
class AugmentationLayer(nn.Module):
    def __init__(self, image_size=(IMAGE_SIZE, IMAGE_SIZE)):
        super(AugmentationLayer, self).__init__()
        self.augmentations = T.Compose([
            T.RandomHorizontalFlip(p=0.5),            # 50% chance of horizontal flip
            T.RandomRotation(degrees=30),            # Random rotation within ±30 degrees
            T.Resize(image_size),                    # Resize to the target size
            T.RandomCrop(image_size),                # Optional: Random crop
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
        ])

    def forward(self, x):
        # Apply augmentations
        return self.augmentations(x)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            # Preprocessing and augmentation layers
            ScaleAndResizeLayer(),                                 # Scaling and Resizing layer
            AugmentationLayer(),                                   # Augmentation layer

            # First convolution block
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Additional convolution blocks
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Flatten layer
            nn.Flatten(),

            # Fully connected (dense) layer
            nn.Linear(64 * 4 * 4, 64),  # Adjust input features based on the final dimensions
            nn.ReLU(),

            # Final classification layer
            nn.Linear(64, NUM_CLASSES),  # Final layer with number of neurons equal to number of classes
            nn.Softmax(dim=1)              # Softmax for multi-class classification
        )

    def forward(self, x):
        assert x.shape[1:] == (3, 256, 256), f"Expected (3, 256, 256), got {x.shape[1:]}"  # Check input shape
        return self.model(x)
    
# Load the model
model = MyModel()
model.load_state_dict(torch.load(r"saved_model/model_v1.pth", map_location=torch.device("cpu")))
model.eval()  # Set to evaluation mode



# Define image preprocessing
def preprocess_image(image):
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Matching the image size used during training
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Adding batch dimension




app = FastAPI()

@app.get("/")
async def root():
    return "Hello World"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    bytes_data = await file.read() # Read image as bytes
    img = read_image(bytes_data)
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        return {"class": CLASSES[predicted.item()]}


def read_image(bytes_data):
    img_pil = Image.open(BytesIO(bytes_data))
    img = np.array(img_pil)
    return img



    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

