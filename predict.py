import torch
from torchvision import transforms
from PIL import Image
import io
from model import GarbageCNN

# =====================
# SETUP
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GarbageCNN().to(DEVICE)
model.load_state_dict(torch.load("ai/model.pth", map_location=DEVICE))
model.eval()

class_names = ['high', 'low', 'medium']


# =====================
# PREDICTION FUNCTION
# =====================
def predict_image(file_content: bytes):
    print("FUNCTION CALLED ")

    try:
        image = Image.open(io.BytesIO(file_content)).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probs, 1)

            # DEBUG
            print("PROBS:", probs)
            print("Predicted index:", predicted.item())

        label = class_names[predicted.item()]
        confidence = round(confidence.item() * 100, 2)

        return label, f"Confidence: {confidence}%"

    except Exception as e:
        return None, str(e)