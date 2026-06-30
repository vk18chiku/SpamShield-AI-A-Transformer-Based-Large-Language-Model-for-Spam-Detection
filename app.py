import streamlit as st
import torch
import tiktoken
from model_def import GPTModel, BASE_CONFIG

# Page Config
st.set_page_config(page_title="SpamShield AI", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPTModel(BASE_CONFIG).half()
    model.load_state_dict(torch.load('review_classifier.pth', map_location=device))
    model.to(device)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    return model, tokenizer, device

model, tokenizer, device = load_model()
max_length = 120
pad_token_id = 50256

def predict_spam(text):
    try:
        input_ids = tokenizer.encode(text)
        supported_context_length = model.pos_emb.weight.shape[0]
        input_ids = input_ids[:min(max_length, supported_context_length)]
        input_ids += [pad_token_id] * (max_length - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(input_tensor)[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            
        predicted_label = torch.argmax(probas, dim=-1).item()
        confidence = probas[0][predicted_label].item()
        
        return {
            "prediction": "Spam" if predicted_label == 1 else "Ham (Not Spam)", 
            "confidence": confidence
        }
    except Exception as e:
        return {"prediction": "Error", "confidence": 0.0}

# Custom CSS for styling
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(37, 99, 235, 0.1) 0%, transparent 20%),
                radial-gradient(circle at 90% 80%, rgba(147, 51, 234, 0.1) 0%, transparent 20%);
}
.title-container {
    text-align: center;
    padding-bottom: 2rem;
}
.title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 1rem;
}
.highlight {
    background: -webkit-linear-gradient(45deg, #60a5fa, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    background-color: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
}
.dot {
    width: 8px;
    height: 8px;
    background-color: #3b82f6;
    border-radius: 50%;
}
.subtitle {
    color: #9ca3af;
    font-size: 1.125rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-container">
    <div class="badge"><div class="dot"></div> Powered by Custom LLM</div>
    <div class="title">Intelligent <span class="highlight">Spam Detection</span></div>
    <div class="subtitle">Built from scratch using PyTorch. Enter any email, SMS, or message below and let our state-of-the-art transformer model analyze it in real-time.</div>
</div>
""", unsafe_allow_html=True)

message = st.text_area("Message Content", placeholder="Paste the message text here to analyze...", height=150, label_visibility="collapsed")

if st.button("✨ Analyze Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message to analyze.")
    else:
        with st.spinner("Running inference through layers..."):
            result = predict_spam(message)
            
        if result["prediction"] == "Error":
            st.error("An error occurred during prediction.")
        else:
            st.markdown("### Analysis Result")
            is_spam = "Spam" in result["prediction"] and "Not" not in result["prediction"]
            
            if is_spam:
                st.error(f"🚨 **{result['prediction']}** (Confidence: {result['confidence']*100:.1f}%)")
            else:
                st.success(f"✅ **{result['prediction']}** (Confidence: {result['confidence']*100:.1f}%)")
            
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>&copy; 2026 Advanced AI Spam Classifier. Built from scratch with PyTorch and Streamlit.</div>", unsafe_allow_html=True)
