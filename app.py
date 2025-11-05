import streamlit as st
import torch
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Load model
@st.cache_resource
def load_model():
    model = torch.load("model.pt", map_location="cpu")
    model.eval()
    return model

# Load tokenizer and other components
@st.cache_resource
def load_components():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return tokenizer, mlb, metadata

def preprocess_code(code):
    # Your preprocessing function here
    code = code.lower()
    # ... rest of preprocessing
    return code

# Streamlit UI
st.title("🧪 Pharmaceutical Code Efficiency Analyzer")
st.write("Paste your code to detect inefficient patterns")

code_input = st.text_area("Code Input", height=200, 
                         placeholder="Paste your pharmaceutical/biotech code here...")

if st.button("Analyze Code"):
    if code_input.strip():
        with st.spinner("Analyzing code patterns..."):
            try:
                # Load components
                model = load_model()
                tokenizer, mlb, metadata = load_components()
                
                # Preprocess and predict
                processed_code = preprocess_code(code_input)
                sequence = tokenizer.texts_to_sequences([processed_code])
                padded_sequence = pad_sequences(sequence, maxlen=metadata['max_len'], padding='post')
                
                # Convert to torch tensor and predict
                input_tensor = torch.tensor(padded_sequence, dtype=torch.float32)
                with torch.no_grad():
                    predictions = model(input_tensor)
                
                # Display results
                predictions_np = predictions.numpy()
                binary_predictions = (predictions_np > 0.5).astype(int)
                predicted_labels = mlb.inverse_transform(binary_predictions)
                
                if predicted_labels[0]:
                    st.error("🚨 Inefficiencies Detected:")
                    for label in predicted_labels[0]:
                        st.write(f"- {label}")
                else:
                    st.success("✅ No inefficiencies detected!")
                    
            except Exception as e:
                st.error(f"Error analyzing code: {e}")
    else:
        st.warning("Please enter some code to analyze")
