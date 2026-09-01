# Plant Disease Detection System

EfficientNetB0 transfer-learning application for classifying PlantVillage leaf images and displaying Grad-CAM explanations in Streamlit.

## Important limitations

- The model is only valid for the classes saved in `checkpoints/model_config.json`.
- PlantVillage images are controlled and may not represent real mobile-phone field photographs.
- Predictions are advisory and should be confirmed by a qualified agricultural professional.

## Dataset layout

Place the PlantVillage colour folders at:

```text
data/plantvillage/PlantVillage-Dataset-master/raw/color/<class_name>/*.jpg
```

Each class must have its own folder. The training script creates deterministic, stratified train, validation and independent test splits in memory.

## Train

```powershell
python train_model.py
```

For a short pipeline check only:

```powershell
python train_model.py --quick --epochs-head 1 --epochs-finetune 1
```

Training produces:

```text
checkpoints/production_model.keras
checkpoints/model_config.json
checkpoints/test_metrics.json
class_names.json
```

The class file and model configuration are generated automatically. Do not manually reorder or remove class names.

## Run the application

```powershell
pip install -r requirements.txt
streamlit run app/app.py
```

## Evaluate an independent class-per-folder test set

```powershell
python evaluate_model.py --test-dir path/to/test
```

The evaluator fails immediately if test-folder names, model outputs and metadata do not match.
