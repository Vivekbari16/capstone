# Advanced DeepFake Detection System

This repository contains a comprehensive DeepFake detection system that leverages an ensemble of state-of-the-art deep learning models to identify manipulated images with high accuracy. The system combines multiple advanced architectures (EfficientNet, CoAtNet, Swin Transformer, and YOLO) with optimized weights to achieve robust performance in detecting fake images.

## Project Overview

Our DeepFake detection system uses a weighted ensemble approach to combine the strengths of multiple deep learning architectures. The system achieves approximately 87% accuracy on test datasets, with high precision and recall metrics. The web interface allows users to upload images for analysis and receive detailed reports on the authenticity of the content.

## Project Structure

```
├── app.py                      # Flask web application for DeepFake detection
├── final_ensemble.py           # Ensemble model implementation with multiple architectures
├── requirements.txt            # Python dependencies
├── CHECKPOINT.md               # Project progress and version history
├── data/                       # Dataset directory
│   └── images/                 # Image data for training and testing
├── models/                     # Saved model weights and metrics
│   ├── coatnet_0_best.pth      # CoAtNet model weights
│   ├── efficientnet_b0_best.pth # EfficientNet model weights
│   ├── swin_transformer_tiny_best.pth # Swin Transformer model weights
│   ├── yolov10_small_best.pth  # YOLO model weights
│   └── ensemble_metrics.txt    # Performance metrics for the ensemble
├── results/                    # Evaluation results and visualizations
│   └── ensemble_confusion_matrix.png # Confusion matrix for the ensemble model
├── src/                        # Source code
│   ├── config.yaml             # Configuration parameters
│   ├── data/                   # Data loading and preprocessing
│   │   ├── deepfake_dataset.py # Basic dataset implementation
│   │   └── loader.py           # Enhanced data loader with augmentation
│   ├── inference/              # Inference utilities
│   │   └── predict.py          # Functions for model inference
│   ├── models/                 # Model architectures
│   │   ├── coatnet.py          # CoAtNet implementation
│   │   ├── efficientnetv2.py   # EfficientNetV2 implementation
│   │   ├── swin_transformer.py # Swin Transformer implementation
│   │   └── yolov10.py          # YOLO implementation
│   └── train/                  # Training utilities
│       └── train.py            # Training functions and loops
├── static/                     # Static assets for the web application
│   └── team member images      # Team member profile pictures
└── templates/                  # HTML templates for the web application
    └── index.html              # Main web interface
```

## Key Features

- **Multi-Model Ensemble**: Combines EfficientNet, CoAtNet, Swin Transformer, and YOLO architectures with optimized weights
- **Adaptive Thresholding**: Uses different decision thresholds based on prediction confidence levels
- **Test-Time Augmentation**: Improves robustness by combining predictions from original and flipped images
- **Web Interface**: User-friendly Flask application for uploading and analyzing images
- **Detailed Reports**: Provides prediction results with confidence scores and downloadable PDF reports
- **Advanced Data Augmentation**: Implements techniques like MixUp, random rotations, and color jittering during training

## Installation

```bash
# Clone the repository
git clone https://github.com/Vivekbari16/capstone.git
cd capstone

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Application

To run the web application for interactive DeepFake detection:

```bash
python app.py
```

This will start a Flask server on http://localhost:5000. Open this URL in your web browser to access the DeepFake detection interface.

### Command Line Inference

To analyze a single image from the command line:

```bash
python src/inference/predict.py path/to/image.jpg
```

## Model Architecture

Our system uses a weighted ensemble of four state-of-the-art deep learning models:

1. **EfficientNet-B0**: Optimized convolutional neural network with balanced depth, width, and resolution scaling.
2. **CoAtNet**: Hybrid model combining convolution and self-attention mechanisms for improved feature extraction.
3. **Swin Transformer**: Hierarchical vision transformer with shifted windows for efficient attention computation.
4. **YOLOv10**: Adapted from the YOLO object detection architecture, providing fast and accurate feature extraction.

The ensemble combines these models with optimized weights:
- EfficientNet: 1.57
- CoAtNet: 1.06
- Swin Transformer: 0.41
- YOLO: 0.97

The system also employs adaptive thresholding based on confidence levels, with different thresholds for high, medium, and low confidence predictions.

## Performance

Our ensemble model achieves the following performance metrics on the test dataset:

| Metric    | Score |
|-----------|-------|
| Accuracy  | 87%   |
| Precision | 88%   |
| Recall    | 86%   |
| F1-Score  | 87%   |
| ROC-AUC   | 92%   |

## Team

- **Bari Vivek Yadav** - Team Lead & Data Scientist
- **A Srinivas Kasyap** - Machine Learning Engineer
- **R Sai Sumanth** - Full Stack Developer
- **Mrudula Selokar** - Mentor

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the PyTorch team for their excellent deep learning framework
- The implementation of various model architectures was inspired by research papers and open-source implementations
- Special thanks to our mentor for guidance throughout the project




## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The DeepFake detection code is based on various research papers and implementations in the field
- Thanks to the authors of the original datasets used for training and testing
