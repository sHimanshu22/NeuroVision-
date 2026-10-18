

\# NeuroVision — Brain MRI Segmentation



NeuroVision is a deep learning project focused on brain MRI image segmentation. It uses a U-Net-based model to generate a binary mask highlighting the region predicted by the model.



\## Features



\- Brain MRI image upload

\- Deep learning inference using a U-Net model

\- Predicted binary segmentation mask

\- Visual comparison of MRI, prediction, overlay, and reference mask

\- FastAPI backend for model inference

\- React-based frontend dashboard

\- GPU inference support when compatible CUDA hardware is available



\## Technology Stack



\*\*AI / Machine Learning\*\*

\- Python

\- PyTorch

\- NumPy

\- Pillow

\- U-Net architecture



\*\*Backend\*\*

\- FastAPI

\- Uvicorn

\- Python



\*\*Frontend\*\*

\- React

\- Vite

\- Axios

\- Lucide React



\## Project Structure



```text

NeuroVision/

├── ai-service/

│   └── scripts/

│       └── unet\_model.py

├── backend/

│   └── app/

│       └── main.py

├── frontend/

│   ├── src/

│   ├── package.json

│   └── index.html

├── .gitignore

└── README.md

```



Training artifacts, datasets, virtual environments, and large archives are excluded from this repository.



\## Requirements



\- Python 3.10 or a compatible version supported by the installed dependencies

\- Node.js and npm

\- A trained model checkpoint

\- CUDA-compatible GPU for GPU inference; CPU inference may also be possible



\## Local Setup



\### 1. Clone the repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd NeuroVision

```



Replace `YOUR\_GITHUB\_REPOSITORY\_URL` with your repository URL.



\### 2. Set up the Python environment



From the project root, run:



```bash

python -m venv ai-service/.venv

ai-service\\.venv\\Scripts\\activate

```



Install the Python dependencies used by the backend and model code. A complete dependency list should be maintained in a requirements file.



\### 3. Configure the trained model



Place the trained checkpoint at the path expected by the backend:



```text

ai-service/outputs/segmentation/continuation\_lr1e4/best\_continued\_unet.pt

```



The checkpoint is not included in this repository. Update the backend configuration if you store it elsewhere.



\### 4. Start the backend



From the project root:



```bash

python -m uvicorn backend.app.main:app --reload

```



If the application uses a different import configuration, run Uvicorn from the appropriate working directory.



Backend URL:



```text

http://127.0.0.1:8000

```



Interactive API documentation:



```text

http://127.0.0.1:8000/docs

```



\### 5. Start the frontend



Open a second terminal:



```bash

cd frontend

npm install

npm run dev

```



Open the local URL printed by Vite, usually:



```text

http://localhost:5173

```



\## Model Evaluation



The selected U-Net checkpoint was evaluated on a held-out segmentation test set containing 860 cases.



| Metric | Result |

|---|---:|

| Mean Dice score | 0.8075 |

| Median Dice score | 0.9083 |

| Mean IoU | 0.7282 |



These results describe the evaluated test set and do not guarantee performance on new data.



A separate single-image diagnostic produced a Dice score of 0.1418 and an IoU of 0.0763, illustrating that performance can vary substantially between cases.



\## Limitations



\- This is an experimental research and educational project.

\- The model is not clinically validated and must not be used for diagnosis or treatment decisions.

\- Predictions may under-segment or over-segment regions.

\- The displayed binary mask depends on the preprocessing and prediction threshold.

\- The trained checkpoint and dataset are not distributed with this repository.



\## Future Improvements



\- Improve robustness across MRI acquisition conditions.

\- Add more comprehensive error analysis.

\- Package dependencies in reproducible requirements files.

\- Improve model configuration and checkpoint management.

\- Add automated API and frontend tests.





\## Model and Reference Data Configuration



The backend expects the trained U-Net checkpoint at:



`ai-service/outputs/segmentation/continuation\_lr1e4/best\_continued\_unet.pt`



The checkpoint is excluded from Git. Obtain the checkpoint separately from your own saved training artifacts and place it at the expected location.



The `/api/reference-mask` endpoint is configured for one demonstration image and reads its reference mask from the local BRISC2025 segmentation test dataset. The dataset is excluded from this repository. If the dataset is unavailable, the reference-mask panel will not be available.



\## API Endpoints



| Method | Endpoint | Purpose |

|---|---|---|

| GET | `/` | API welcome response |

| GET | `/api/health` | Check model readiness |

| POST | `/api/segment` | Generate a segmentation mask for an uploaded image |

| GET | `/api/reference-mask` | Return the configured example's reference mask |



Interactive API documentation is available at `/docs` when the backend is running.




\## Disclaimer



NeuroVision is intended for research and educational purposes only. It is not a medical device and is not intended to replace professional medical judgment.



