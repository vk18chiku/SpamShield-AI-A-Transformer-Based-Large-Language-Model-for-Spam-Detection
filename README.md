---
title: SpamShield AI
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
---

# SpamShield AI: Advanced LLM Spam Classification System 🛡️

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)

## 🌐 Live Demo
Experience the application live here: **[Spam-Shield-AI on Hugging Face Spaces](https://huggingface.co/spaces/Uttam1695/Spam-Shield-AI)**

---

Welcome to **SpamShield AI**, an intelligent, end-to-end spam classification web application. This project demonstrates the implementation of a custom-built Large Language Model (LLM) utilizing a transformer architecture to classify text messages (SMS/Emails) into **Spam** or **Ham (Not Spam)**.

Every component of this pipeline—from data preprocessing and model architecture to the modern web interface—was engineered entirely from scratch.

## 🚀 Features

- **Custom Transformer Architecture**: Implemented entirely in PyTorch, leveraging attention mechanisms optimized for natural language processing.
- **High-Performance Inference**: Real-time classification of text streams via an integrated Streamlit frontend.
- **Modern User Interface**: A responsive, beautifully designed frontend using Streamlit with custom CSS injection featuring glassmorphism and animated backgrounds.
- **Robust Training Pipeline**: Includes custom PyTorch Datasets/DataLoaders, automated validation loops, and learning rate scheduling (detailed in the Jupyter Notebook).

## 🧠 Model Architecture & Training

The core of SpamShield AI is the `review_classifier.pth` model. It was trained on the benchmark **SMS Spam Collection dataset**.

### Key Highlights:
1. **Data Engineering**: Processed over 5,500 SMS messages (`sms_spam_collection`), split into `train.csv`, `validation.csv`, and `test.csv`. Handling class imbalances and applying advanced tokenization strategies.
2. **Custom Embeddings**: Built custom word and positional embeddings to capture deep semantic meaning and sequence context.
3. **Loss & Accuracy Tracking**: The training process achieved exceptional accuracy, significantly reducing false positives (as visualized in `accuracy-plot.pdf` and `loss-plot.pdf`).
4. **Model Optimization**: The trained weights (`review_classifier.pth`) were converted to FP16 (Half-Precision - `review_classifier_fp16.pth`) and INT8 (`review_classifier_int8.pth`) to reduce memory footprint, enabling lightning-fast, seamless cloud deployments using `@st.cache_resource`.

## 📂 Project Structure

- `app.py`: The main Streamlit web application script providing the modern UI and integrating the model for real-time inference.
- `model_def.py`: Contains the custom PyTorch Transformer model definition.
- `extracted_code.py` / `file.ipynb`: The core research notebook and extracted code showing the step-by-step training process, data loading, and model architecture from scratch.
- `train.csv`, `validation.csv`, `test.csv`: The processed dataset splits used for training and evaluating the model.
- `gpt_download.py` & `gpt2/`: Utilities to download and work with tokenizers/weights if leveraging external pretrained structures.
- `*.pth`: The saved PyTorch model weights in different precisions (standard, fp16, int8).
- `accuracy-plot.pdf` & `loss-plot.pdf`: Visualizations of the model's performance during training.

## 💻 Tech Stack

- **Deep Learning**: PyTorch
- **Backend & Frontend**: Python, Streamlit
- **Data Processing**: Pandas, NumPy
- **Environment**: Jupyter Notebook (for model research and development)
- **Deployment**: Hugging Face Spaces, Docker

## 🛠️ Installation & Setup

To run this application locally on your machine, follow these steps:

### Prerequisites

Make sure you have Python 3.8+ installed.

### 1. Clone the repository
```bash
git clone https://github.com/vk18chiku/SpamShield-AI-A-Transformer-Based-Large-Language-Model-for-Spam-Detection.git
cd SpamShield-AI-A-Transformer-Based-Large-Language-Model-for-Spam-Detection
```

### 2. Install Dependencies
Install the required Python packages (it is recommended to use a virtual environment):
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Streamlit application:
```bash
streamlit run app.py
```

The interface will automatically open in your default web browser (typically at `http://localhost:8501`).

## 📸 Screenshots & UI Design

The web interface was meticulously designed to provide a premium user experience. It features:
- **Glassmorphism**: Sleek, transparent UI elements that blur the dynamic background.
- **Real-time Feedback**: Animated loading states and dynamic result cards that adapt based on the prediction confidence.
- **Responsive Layout**: Flawless execution across desktop and mobile devices.

## 📈 Future Enhancements

- **API Rate Limiting**: Implementing Redis to manage request throttling.
- **INT8 Quantization**: Already explored, but further edge deployment using dynamic quantization without sacrificing accuracy.
- **Multi-lingual Support**: Expanding the vocabulary to detect spam in multiple languages.

---
*Designed and engineered by Uttam Kumar Mahato*
