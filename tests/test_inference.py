from app.services.inference import run_inference
from PIL import Image
import io

def test_run_inference():
    # Create dummy white image
    image = Image.new("RGB", (224, 224), color="white")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    result = run_inference(buf.read())
    assert "boxes" in result
    assert isinstance(result["boxes"], list)
