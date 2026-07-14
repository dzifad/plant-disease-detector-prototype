# PLANT DISEASE DETECTION SYSTEM
An AI-powered web application for early plant disease detection using Convolutional Neural Networks (CNNs). The application allows users to upload an image of a plant leaf, predicts the disease affecting the plant, generates a Grad-CAM heatmap showing which regions influenced the prediction and provides useful disease information including symptoms, causes, treatment and prevention.


## OVERVIEW
Plant diseases are a major cause of reduced agricultural productivity worldwide. Early and accurate disease detection helps farmers take timely action, minimize crop losses, and improve food production.

This project uses Deep Learning and Explainable Artificial Intelligence (XAI) to make disease prediction more transparent and easier to understand.

## FEATURES
- Upload plant leaf images
- Predict plant diseases using a trained CNN model
- Display prediction confidence score
- Generate Grad-CAM heatmaps for model explainability
- Display disease symptoms
- Display possible causes
- Display treatment recommendations
- Display prevention measures
- Clean and user-friendly Streamlit interface


## MODEL
The disease classification model was trained using TensorFlow/Keras.

Model Architecture:
- Convolutional Neural Network (CNN)

Framework:
- TensorFlow
- Keras

Explainability:
- Grad-CAM (Gradient-weighted Class Activation Mapping)

## PROJECT STRUCTURE
```
Plant Disease Detector Prototype/
│
├── app/
│   ├── app.py
│   ├── predict.py
│   ├── gradcam.py
│   ├── disease_info_fallback.py
│   ├── banner.jpg
│   └── samples/
│
├── checkpoints/
│   ├── best_model.h5
│   └── final_saved_model/
│
├── class_names.json
├── requirements.txt
├── .gitignore
└── README.md
```

## TECHNOLOGIES USED
- Python 3.11
- Streamlit
- TensorFlow
- Keras
- NumPy
- Pillow (PIL)
- Matplotlib
- JSON

## INSTALLATION
Clone the repository
```bash
git clone https://github.com/dzifad/plant-disease-detector-prototype.git
```

Navigate into the project
```bash
cd plant-disease-detector-prototype
```

Create a virtual environment
```bash
python3 -m venv .venv
```

Activate the virtual environment
```bash
source .venv/bin/activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Run the application
```bash
streamlit run app/app.py
```

## HOW TO USE
1. Launch the Streamlit application.
2. Upload an image of a plant leaf.
3. Wait for the prediction.
4. View the predicted disease and confidence score.
5. Examine the Grad-CAM heatmap to understand which regions influenced the prediction.
6. Read the disease information, including symptoms, causes, treatment, and prevention.

## DATASET
This project uses the PlantVillage dataset, which contains thousands of labeled images covering multiple crops and plant diseases.

##  Future Improvements
- Support additional crop species
- Improve prediction accuracy with larger datasets
- Deploy the application online
- Add multilingual support

## AUTHORS
** Dzifa Esi Dotse, Enock Daniel Essoun **
Final Year Information Technology Student,
University of Ghana

## LICENSE
This project was developed for academic purposes as a final-year undergraduate project.