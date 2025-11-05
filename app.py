import streamlit as st
import torch
import numpy as np

st.title("🔮 Simple Streamlit Model Demo")

# --- Load model once and cache it ---
@st.cache_resource
def load_model():
    model = torch.load("model.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

# --- UI for input ---
user_input = st.number_input("Enter a number:", value=0.0, format="%.2f")

if st.button("Predict"):
    with torch.no_grad():
        x = torch.tensor([[user_input]], dtype=torch.float32)
        y = model(x)
        st.success(f"Model output: {y.item():.4f}")
