# u{1F43E} Animal Classifier - Deep Learning Image Classification

A deep learning application that classifies animal images using **Convolutional Neural Networks (CNN)**. Built with Python as a Software Laboratory final project.

## u{1F680} Features
- CNN-based image classification model
- Automated data preparation pipeline
- Model training with configurable parameters
- Interactive classification application
- Pre-trained model outputs for quick inference

## u{1F6E0}u{FE0F} Tech Stack
| Technology | Purpose |
|---|---|
| Python | Primary language |
| TensorFlow/Keras | Deep learning framework |
| NumPy | Numerical computing |
| OpenCV | Image processing |
| Matplotlib | Visualization |

## u{1F4E6} Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python data_preparation.py
```

### 3. Train Model
```bash
python train_model.py
```

### 4. Run Classifier
```bash
python animal_classifier_app.py
```

## u{1F4C2} Project Structure
```
u{251C}u{2500}u{2500} main.py                  # Main entry point
u{251C}u{2500}u{2500} animal_classifier_app.py # Classification application
u{251C}u{2500}u{2500} data_preparation.py      # Dataset preparation
u{251C}u{2500}u{2500} train_model.py           # Model training script
u{251C}u{2500}u{2500} class_mapping.json       # Class label mappings
u{251C}u{2500}u{2500} src/                     # Additional source modules
u{2514}u{2500}u{2500} model_outputs/           # Trained model artifacts
```