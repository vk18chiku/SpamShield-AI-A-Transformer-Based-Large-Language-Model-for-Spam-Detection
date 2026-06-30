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
1. **Data Engineering**: Processed over 5,500 SMS messages, handling class imbalances and applying advanced tokenization strategies.
2. **Custom Embeddings**: Built custom word and positional embeddings to capture deep semantic meaning and sequence context.
3. **Loss & Accuracy Tracking**: The training process achieved exceptional accuracy, significantly reducing false positives (as visualized in `accuracy-plot.pdf` and `loss-plot.pdf`).
4. **State Dict**: The trained weights (`review_classifier.pth`) are loaded seamlessly into the Streamlit application for low-latency predictions using `@st.cache_resource`.

## 💻 Tech Stack

- **Deep Learning**: PyTorch
- **Backend & Frontend**: Python, Streamlit
- **Data Processing**: Pandas, NumPy
- **Environment**: Jupyter Notebook (for model research and development)

## 🛠️ Installation & Setup

To run this application locally on your machine, follow these steps:

### Prerequisites

Make sure you have Python 3.8+ installed.

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Spam_Classifier_using_LLM_from_scratch.git
cd Spam_Classifier_using_LLM_from_scratch
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
- **Model Quantization**: Reducing the model size for edge deployment without sacrificing accuracy.
- **Multi-lingual Support**: Expanding the vocabulary to detect spam in multiple languages.

---
*Designed and engineered by Uttam Kumar Mahato*
