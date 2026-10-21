# NeuroVision — Brain MRI Analysis and Tumor Segmentation

NeuroVision is a final-year engineering project that uses deep learning to analyze brain MRI images. It combines tumor classification, segmentation, and Grad-CAM visualization in a web application.

**Purpose:** Educational and experimental use only. NeuroVision is not a clinically validated diagnostic system.

## Features

- **Tumor classification:** Predicts one of four classes: glioma, meningioma, no tumor, or pituitary.
- **Tumor segmentation:** Generates a binary mask highlighting the region predicted by the U-Net model.
- **Grad-CAM visualization:** Generates a heatmap to help visualize image regions influencing the classification model.
- **Interactive web interface:** Upload MRI images and view analysis results.
- **Reference-mask comparison:** Supports a configured demonstration image when the local reference dataset is available.
- **GPU inference:** Uses CUDA when compatible hardware and software are available.

## Technology Stack

**AI and Machine Learning**
- Python
- PyTorch and TorchVision
- U-Net segmentation architecture
- Classification model
- Grad-CAM visualization
- NumPy and Pillow

**Backend**
- FastAPI
- Uvicorn

**Frontend**
- React
- Vite
- Axios
- Lucide React

## Project Structure

```text
NeuroVision/
├── ai-service/
│   ├── scripts/
│   │   ├── unet_model.py
│   │   ├── train_classification.py
│   │   └── evaluation scripts
│   └── outputs/
│       └── segmentation/
├── backend/
│   └── app/
│       ├── main.py
│       ├── classification_service.py
│       └── gradcam_service.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── index.html
├── requirements.txt
├── .gitignore
└── README.md
```

Training datasets, trained checkpoints, and local virtual environments are not included in the Git repository.

## Requirements

- Python compatible with the installed PyTorch and TorchVision versions
- Node.js and npm
- Python dependencies from `requirements.txt`
- Trained model checkpoints
- A compatible CUDA GPU for GPU inference; CPU inference may also be possible

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/sHimanshu22/NeuroVision-.git
cd NeuroVision-
```

### 2. Set up Python

From the project root, create and activate a virtual environment:

**Windows CMD**
```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Install a PyTorch and TorchVision build compatible with your machine if the versions installed from the requirements file do not support your desired hardware.

### 3. Configure model checkpoints

Place the trained checkpoints at the locations expected by the corresponding model services.

The segmentation backend expects:

```text
ai-service/outputs/segmentation/continuation_lr1e4/best_continued_unet.pt
```

The classification service and Grad-CAM service also require their configured classification checkpoint files.

Model checkpoints and datasets are excluded from this repository. Obtain them from your own training artifacts and verify the paths in the service configuration before starting the application.

### 4. Start the backend

From the project root:

```cmd
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Backend URL: `http://127.0.0.1:8000`

Health check: `http://127.0.0.1:8000/api/health`

Interactive API documentation: `http://127.0.0.1:8000/docs`

### 5. Start the frontend

Open a second terminal:

```cmd
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/api/health` | Model readiness and inference configuration |
| GET | `/api/reference-mask` | Retrieve the configured demonstration reference mask |
| POST | `/api/segment` | Generate a segmentation mask |
| POST | `/api/classify` | Classify an uploaded MRI |
| POST | `/api/gradcam` | Generate a Grad-CAM visualization |

## Model Evaluation

### Classification

The classification model achieved **98.9% accuracy** on the evaluated test set. Exact duplicate images were identified across the classification training and test splits during the dataset audit. This potential data leakage can inflate the measured accuracy, so the result should be interpreted cautiously and should not be treated as a reliable estimate of generalization to unseen patients.

### Segmentation

The selected U-Net checkpoint was evaluated on 860 official test images.

| Metric | Result |
|---|---:|
| Mean Dice score | 0.8075 |
| Median Dice score | 0.9083 |
| Mean IoU | 0.7282 |

Performance varies across individual images. Some cases have poor or empty predictions even when the overall metrics are strong. The application should therefore be considered an experimental prototype.

## Limitations

- The models have not been clinically validated.
- Classification predictions and segmentation masks can be incorrect.
- Grad-CAM is a visualization aid, not proof that a model's prediction is correct.
- Results depend on image preprocessing, training data, model checkpoints, and inference settings.
- A high evaluation score does not guarantee reliable performance on new data.
- The configured reference-mask feature depends on local dataset files.

## Future Improvements

- Evaluate on independent data without train/test leakage.
- Improve segmentation robustness on difficult MRI cases.
- Add automated backend and frontend tests.
- Improve checkpoint and dependency management.
- Expand documentation and usability testing.

## Disclaimer

NeuroVision is a final-year engineering and educational project. It is not a medical device and must not be used to diagnose disease or guide treatment. All results require qualified clinical interpretation.
