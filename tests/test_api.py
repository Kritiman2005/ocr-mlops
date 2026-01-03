from fastapi.testclient import TestClient
from api.main import app
from PIL import Image
import io

client = TestClient(app)

def test_ocr_endpoint():
    image = Image.new("RGB", (100, 50), color="white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/ocr",
        files={"file": ("test.png", buf, "image/png")}
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()
