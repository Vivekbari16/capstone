import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io, base64
from flask import Flask, render_template, request, jsonify
from final_ensemble import EfficientNetModel, CoAtNetModel, SwinTransformerModel, YOLOModel, FinalEnsemble, get_transforms, predict_image as ensemble_predict_image

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ensemble_model, ensemble_transform = None, None

def load_ensemble():
    global ensemble_model, ensemble_transform
    base_dir = os.path.join(os.path.dirname(__file__), 'models')
    # Load all four models with their best weights
    effnet = EfficientNetModel()
    coatnet = CoAtNetModel()
    swin = SwinTransformerModel()
    yolo = YOLOModel()
    effnet.load_state_dict(torch.load(os.path.join(base_dir, 'efficientnet_b0_best.pth'), map_location=device))
    coatnet.load_state_dict(torch.load(os.path.join(base_dir, 'coatnet_0_best.pth'), map_location=device))
    # Fix for SwinTransformerModel state_dict mismatch
    swin_state = torch.load(os.path.join(base_dir, 'swin_transformer_tiny_best.pth'), map_location=device)
    # Remove extra keys if present
    if 'base.fc.weight' in swin_state and 'base.fc.1.weight' not in swin_state:
        # The model expects a Sequential [Dropout, Linear], but the checkpoint has only Linear
        # Convert to match model definition
        swin_state['base.fc.1.weight'] = swin_state['base.fc.weight']
        swin_state['base.fc.1.bias'] = swin_state['base.fc.bias']
        del swin_state['base.fc.weight']
        del swin_state['base.fc.bias']
    swin.load_state_dict(swin_state)
    yolo.load_state_dict(torch.load(os.path.join(base_dir, 'yolov10_small_best.pth'), map_location=device))
    for m in [effnet, coatnet, swin, yolo]:
        m.to(device)
        m.eval()
    weights = [0.9, 0.8, 0.6, 0.5]
    ensemble_model = FinalEnsemble([effnet, coatnet, swin, yolo], weights=weights)
    ensemble_model.to(device)
    ensemble_model.eval()
    ensemble_transform = get_transforms()
    print("Ensemble model loaded successfully")

def predict_image_ensemble(image):
    # image is a PIL Image, preprocess and run through ensemble with TTA for robust prediction
    x = ensemble_transform(image).unsqueeze(0).to(device)
    from final_ensemble import test_time_augmentation
    with torch.no_grad():
        preds, probs = test_time_augmentation(ensemble_model, x, device, num_augments=3)
        predicted = preds.item()
        confidence = probs[0][predicted].item()
        print('--- DEBUG ENSEMBLE OUTPUT ---')
        print(f'Ensemble probabilities: {probs}')
        print(f'Predicted label: {predicted} (0=Real, 1=Fake)')
        print(f'Confidence: {confidence}')
        # Print individual model outputs for this image
        if hasattr(ensemble_model, 'models'):
            for idx, m in enumerate(ensemble_model.models):
                out = m(x)
                prob = torch.softmax(out, dim=1)
                print(f'Model {idx} ({type(m).__name__}) probs: {prob}')
        print('-----------------------------')
    return predicted, confidence











from flask import Flask, render_template, request, jsonify
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file uploaded'})
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        label, conf = predict_image_ensemble(img)
        result = 'Fake' if label == 1 else 'Real'
        return jsonify({'result': result, 'confidence': f'{conf:.2%}'})
    except Exception as e:
        print('Analyze Exception:', e)
        return jsonify({'error': str(e)})

from flask import send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import json

@app.route('/report', methods=['POST'])
def report():
    data = request.get_json()
    result = data.get('result', '-')
    confidence = data.get('confidence', '-')
    # Generate PDF
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    c = canvas.Canvas(temp.name, pagesize=letter)
    c.setFont('Helvetica-Bold', 20)
    c.drawString(72, 720, 'Deepfake Detector Report')
    c.setFont('Helvetica', 14)
    c.drawString(72, 680, f'Prediction: {result}')
    c.drawString(72, 650, f'Confidence Score: {confidence}')
    c.setFont('Helvetica', 12)
    c.drawString(72, 610, 'This report summarizes the result of the deepfake analysis.')
    c.save()
    temp.seek(0)
    return send_file(temp.name, as_attachment=True, download_name='deepfake_report.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    load_ensemble()
    app.run(debug=True)


