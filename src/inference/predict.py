import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import XceptionNet

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = XceptionNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def predict_image(model, image_path, device):
    """Predict if an image is real or fake"""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0][prediction].item()
    
    result = "Real" if prediction.item() == 1 else "Fake"
    return result, confidence

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        return
    
    image_path = sys.argv[1]
    model_path = '../../models/deepfake_detector.pth'
    
    # Load model
    model, device = load_model(model_path)
    
    # Make prediction
    result, confidence = predict_image(model, image_path, device)
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == '__main__':
    main()
