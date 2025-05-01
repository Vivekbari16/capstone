import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_ensemble import EfficientNetModel, CoAtNetModel, SwinTransformerModel, YOLOModel, FinalEnsemble
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

def load_ensemble(device):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
    # Load all four models with their best weights
    effnet = EfficientNetModel()
    coatnet = CoAtNetModel()
    swin = SwinTransformerModel()
    yolo = YOLOModel()
    effnet.load_state_dict(torch.load(os.path.join(base_dir, 'efficientnet_b0_best.pth'), map_location=device))
    coatnet.load_state_dict(torch.load(os.path.join(base_dir, 'coatnet_0_best.pth'), map_location=device))
    swin_state = torch.load(os.path.join(base_dir, 'swin_transformer_tiny_best.pth'), map_location=device)
    # Fix for SwinTransformerModel state_dict mismatch
    if 'base.fc.weight' in swin_state and 'base.fc.1.weight' not in swin_state:
        swin_state['base.fc.1.weight'] = swin_state['base.fc.weight']
        swin_state['base.fc.1.bias'] = swin_state['base.fc.bias']
        del swin_state['base.fc.weight']
        del swin_state['base.fc.bias']
    swin.load_state_dict(swin_state)
    yolo.load_state_dict(torch.load(os.path.join(base_dir, 'yolov10_small_best.pth'), map_location=device))
    for m in [effnet, coatnet, swin, yolo]:
        m.to(device)
        m.eval()
    # Use the same weights as in evaluation
    weights = [1.5666666666666667, 1.0555555555555556, 0.4111111111111111, 0.9666666666666667]  # Optimized weights from quick_optimize.py
    ensemble_model = FinalEnsemble([effnet, coatnet, swin, yolo], weights=weights)
    ensemble_model.to(device)
    ensemble_model.eval()
    # Set optimal threshold if supported
    if hasattr(ensemble_model, 'mid_conf_threshold'):
        ensemble_model.mid_conf_threshold = 0.54  # Optimal threshold
    return ensemble_model

def predict_image_ensemble(ensemble_model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        # Use the advanced prediction logic from FinalEnsemble
        preds, probs = ensemble_model.predict(x)
        predicted = preds.item()
        confidence = probs[0][predicted].item()
        print('--- DEBUG ENSEMBLE OUTPUT ---')
        print(f'Ensemble probabilities: {probs}')
        print(f'Predicted label: {predicted} (0=Real, 1=Fake)')
        print(f'Confidence: {confidence}')
        print('-----------------------------')
    result = 'Fake' if predicted == 1 else 'Real'
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
