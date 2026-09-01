# Plant Disease Detection System

An AI-powered web application for early plant disease detection using deep learning and Explainable Artificial Intelligence (XAI). The application allows users to upload an image of a plant leaf, predicts the disease affecting the plant, displays a confidence score, generates a Grad-CAM heatmap showing the regions that influenced the prediction, and provides useful disease information including symptoms, causes, treatment, and prevention.

## Overview

Plant diseases are a major cause of reduced agricultural productivity worldwide. Early and accurate disease detection can help farmers take timely action, minimize crop losses, and improve food production.

This project uses deep learning and Explainable Artificial Intelligence (XAI) to provide plant disease predictions while making the model's decision-making process more transparent and easier to understand.

## Features

- Upload plant leaf images
- Predict plant diseases using a trained deep-learning model
- Display prediction confidence as a percentage
- Provide a warning for low-confidence predictions
- Generate Grad-CAM heatmaps for model explainability
- Display disease symptoms
- Display possible causes
- Display treatment recommendations
- Display prevention measures
- Explore sample images of predicted disease classes
- Clean and user-friendly Streamlit interface

## Model

The current disease classification model uses **EfficientNetB0 with transfer learning**, implemented using TensorFlow/Keras.

EfficientNetB0 is a Convolutional Neural Network (CNN) architecture. Transfer learning allows the project to use previously learned visual features and adapt them to the plant disease classification task.

The model was trained using data augmentation and a two-stage training process involving initial training and fine-tuning.

The application loads the production model from:

```text
checkpoints/production_model.keras
```

Model configuration and class information are stored in:

```text
checkpoints/model_config.json
class_names.json
```

The model supports **38 PlantVillage disease and healthy classes** across multiple crop species.

## Explainability

The application uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations for predictions.

Grad-CAM produces a heatmap highlighting areas of the uploaded leaf image that contributed to the model's prediction. This helps users understand which visual regions influenced the classification instead of receiving only a disease label.

## Dataset

The project uses the **PlantVillage dataset**, which contains thousands of labelled plant leaf images covering multiple crops, diseases, and healthy classes.

PlantVillage images are generally captured under relatively controlled conditions. Therefore, images taken in real-world environments may differ from the training data because of factors such as:

- background variation
- lighting conditions
- camera angle
- image quality
- leaf orientation
- disease severity

These differences can affect model confidence and prediction performance.

## Model Evaluation

The retrained model was evaluated using an independent class-per-folder test dataset.

A total of **4,242 leaf images** from the represented test classes were evaluated after excluding 115 background-only images.

The model correctly classified:

```text
4,004 / 4,242 images
```

This produced an overall accuracy of:

```text
94.39%
```

The independent test dataset did not contain examples for every one of the model's 38 output classes. Therefore, the 94.39% result represents performance on the classes available in the independent test set and should not be interpreted as independent evaluation of all 38 classes.

Detailed evaluation artifacts are available in:

```text
classification_report.txt
confusion_matrix.png
evaluation_predictions.csv
```

The evaluation can also be run using:

```bash
python evaluate_model.py --test-dir path/to/test
```

## Confidence and Real-World Images

Prediction confidence represents how strongly the model favors its predicted class relative to the other available classes. It is not the same as the model's overall accuracy.

During informal testing, some images obtained from outside the PlantVillage-style dataset, including internet images, produced lower confidence scores than images from the independent test dataset.

This suggests that the model performs best on images similar to its training distribution and may have reduced confidence when presented with more varied real-world photographs.

The application therefore warns users when a prediction has low confidence rather than presenting uncertain predictions as highly reliable.

## Project Structure

```text
Plant Disease Detector Prototype/
│
├── app/
│   ├── app.py
│   ├── predict.py
│   ├── gradcam.py
│   ├── disease_info_fallback.py
│   └── samples/
│
├── checkpoints/
│   ├── production_model.keras
│   └── model_config.json
│
├── class_names.json
├── train_model.py
├── evaluate_model.py
├── classification_report.txt
├── confusion_matrix.png
├── evaluation_predictions.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies Used

- Python 3.11
- TensorFlow
- Keras
- EfficientNetB0
- Streamlit
- NumPy
- Pillow (PIL)
- Matplotlib
- Grad-CAM
- JSON

## Installation

Clone the repository:

```bash
git clone https://github.com/dzifad/plant-disease-detector-prototype.git
```

Navigate into the project:

```bash
cd plant-disease-detector-prototype
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

## How to Use

1. Launch the Streamlit application.
2. Upload an image of a plant leaf.
3. Wait for the model to process the image.
4. View the predicted disease and confidence score.
5. Check any low-confidence warning displayed by the application.
6. Examine the Grad-CAM heatmap to see which areas influenced the prediction.
7. Explore sample images associated with the predicted class.
8. Read the disease information, including symptoms, causes, treatment, and prevention.

## Training

The model can be retrained using:

```bash
python train_model.py
```

A short pipeline test can be performed using:

```bash
python train_model.py --quick --epochs-head 1 --epochs-finetune 1
```

The training process generates the production model and its associated model configuration.

Class names and model configuration should remain consistent with the trained model and should not be manually reordered.

## Limitations

- The model is limited to the plant classes it was trained to recognize.
- The independent evaluation dataset did not represent all 38 model classes.
- PlantVillage-style images may not fully represent real agricultural environments.
- Lighting, backgrounds, image quality, multiple leaves, camera angles, and disease severity can affect predictions.
- Low confidence indicates that the model is uncertain between possible classes.
- Predictions should be treated as decision-support information rather than a replacement for professional agricultural diagnosis.

## Future Improvements

- Train and evaluate using more diverse real-world field images
- Obtain independent test images covering all 38 supported classes
- Support additional crop species and diseases
- Improve generalization under different lighting and background conditions
- Add camera-based image capture
- Add multilingual support
- Deploy and evaluate the application in real agricultural environments

## Authors

**Dzifa Esi Dotse and Enock Daniel Essoun**

Final Year Information Technology Students  
University of Ghana

## License

This project was developed for academic purposes as a final-year undergraduate project.