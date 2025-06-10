import httpx
from PIL import Image
import io

# Create a dummy image in memory
image = Image.new("RGB", (224, 224), color="white")
buf = io.BytesIO()
image.save(buf, format="JPEG")
buf.seek(0)

url = "http://127.0.0.1:8000/predict"
files = {"file": ("dummy.jpg", buf, "image/jpeg")}

response = httpx.post(url, files=files)
print("Response:", response.json())