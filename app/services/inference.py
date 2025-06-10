import torch
from torchvision import transforms
from PIL import Image
import io

model = torch.load("model/fasterrcnn.pth")
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

def run_inference(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)[0]
    return {"boxes": outputs["boxes"].tolist()}
