# Enhanced DeepFake Detection System

This repository contains a comprehensive DeepFake detection system that uses state-of-the-art deep learning models to identify manipulated images and videos. The system includes multiple advanced model architectures, optimized training pipelines, and tools for inference on both images and videos. Our enhanced system achieves 75-90% accuracy on DeepFake detection tasks.

## Project Structure

```
├── data/
│   ├── images/
│   │   ├── real/       # Real face images
│   │   └── fake/       # Fake face images (with easy, mid, hard categories)
│   └── videos/         # Video files for testing
├── models/             # Saved model weights
├── results/            # Evaluation results and plots
├── src/                # Source code
│   ├── dataset.py      # Dataset classes and data loaders
│   ├── inference.py    # Functions for inference on images and videos
│   ├── models.py       # Model architectures
│   ├── train.py        # Training functions
│   └── utils.py        # Utility functions
├── batch_image_detection.py    # Script for batch processing of images
├── demo_app.py                 # GUI application for DeepFake detection
├── model_comparison.py         # Script to compare different models
├── train_difficulty_analysis.py # Analysis of model performance on different difficulty levels
├── train_efficientnet.py       # Script to train EfficientNet model
├── train_ensemble.py           # Script to create and evaluate ensemble model
├── train_resnet.py             # Script to train ResNet model
├── train_xception.py           # Script to train Xception model
├── video_detection.py          # Script for DeepFake detection in videos
└── README.md                   # This file
```

## Installation

1. Clone this repository:
```
git clone https://github.com/username/deepfake-detection.git
cd deepfake-detection
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Install additional dependencies for the state-of-the-art models:
```
pip install timm albumentations ultralytics optuna
```

## Dataset

The dataset is organized as follows:
- `data/images/real/`: Contains real face images
- `data/images/fake/`: Contains fake face images, with filenames prefixed by difficulty level:
  - `easy_*.jpg`: Easier to detect fakes
  - `mid_*.jpg`: Medium difficulty fakes
  - `hard_*.jpg`: Harder to detect fakes
- `data/videos/`: Contains video files for testing

## Usage

### Training Models

To train the different models, run the following scripts:

```
# Train Swin Transformer V2 model
python train_swin.py

# Train CoAtNet model
python train_coatnet.py

# Train EfficientNetV2 model
python train_efficientnetv2.py

# Train YOLOv10 model
python train_yolov10.py

# Create and evaluate ensemble model (requires all models above)
python train_ensemble.py
```

You can customize the training process with command-line arguments:

```
python train_swin.py --model_size tiny --batch_size 16 --lr 1e-4 --epochs 20
```

For hyperparameter tuning:

```
python src/tune_hyperparams.py --config src/config.yaml --n_trials 50
```

### Analyzing Model Performance

To analyze model performance on different difficulty levels:

```
python src/evaluate.py --model swin_transformer_tiny --analyze_difficulty
```

To compare all models:

```
python src/evaluate.py
```

To optimize ensemble weights:

```
python train_ensemble.py --ensemble_type weighted --optimize
```

To perform grid search for optimal ensemble weights:

```
python train_ensemble.py --ensemble_type weighted --grid_search
```

### Inference

For batch processing of images:

```
python batch_image_detection.py --image_dir path/to/images --model ensemble --output results.csv
```

For video processing:

```
python video_detection.py --video path/to/video.mp4 --model ensemble --output processed_video.avi
```

### GUI Application

To use the graphical user interface:

```
python demo_app.py
```

## Model Architectures

The system includes the following state-of-the-art model architectures:

1. **Swin Transformer V2**: A hierarchical vision transformer with shifted windows for efficient attention computation
2. **CoAtNet**: A hybrid model that combines convolution and self-attention for improved performance
3. **EfficientNetV2**: An improved version of EfficientNet with better parameter efficiency and accuracy
4. **YOLOv10**: A backbone from the YOLO object detection architecture adapted for binary classification
5. **Weighted Ensemble**: An optimized ensemble that combines predictions from multiple models with learned weights

We also include the previous generation models for comparison:
- **Xception**: A custom Xception-like architecture with separable convolutions
- **ResNet50**: A pre-trained ResNet50 model with a custom classification head
- **EfficientNet-B3**: A pre-trained EfficientNet-B3 model with a custom classification head

## Performance

The models are evaluated on metrics such as:
- Accuracy
- Precision and Recall
- F1 Score
- ROC curves and AUC
- Precision-Recall curves

Results are saved in the `results/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The DeepFake detection code is based on various research papers and implementations in the field
- Thanks to the authors of the original datasets used for training and testing
