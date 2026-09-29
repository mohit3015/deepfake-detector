# Deepfake Detector

AI-powered deepfake detection web app built with **EfficientNet-B4** (PyTorch + timm) and a **Streamlit** UI. Upload a face image or a short video and get a Real/Fake prediction with a confidence score.

**Live demo:** _(link add karna baad me)_
**Model weights:** [mohitbharti1530/deepfake-detector-model](https://huggingface.co/mohitbharti1530/deepfake-detector-model)

## Features
- Image detection with Real/Fake probability
- Video detection: samples frames and averages the fake probability
- Model weights loaded automatically from the Hugging Face Hub
- Runs on CPU

## Model performance
Evaluated on a balanced test set of 20,000 images (10,000 Real + 10,000 Fake) from the [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset.

| Metric | Score |
|---|---|
| Accuracy | 94.13% |
| Precision (Fake) | 95.96% |
| Recall (Fake) | 92.13% |
| F1-score | 94.01% |
| ROC-AUC | 0.9873 |

Confusion matrix (rows = actual, columns = predicted):

|  | Pred Real | Pred Fake |
|---|---|---|
| **Actual Real** | 9612 | 388 |
| **Actual Fake** | 787 | 9213 |

## Tech stack
Python, PyTorch, timm (EfficientNet-B4), Streamlit, OpenCV, Hugging Face Hub

## Run locally
```bash
git clone https://github.com/mohit3015/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Project files
- `Deepfake_Detection_Project.ipynb`: training and evaluation notebook
- `streamlit_app.py`: web app
- `requirements.txt`: dependencies

## Limitations
- Metrics are on the test split of the same dataset used for training; accuracy may be lower on other deepfake generators, compressed videos or real-world photos.
- Works best on cropped, front-facing face images similar to the training data.
- This is a demo and should not be used as sole evidence to judge whether media is authentic.
