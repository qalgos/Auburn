import streamlit as st
import torch
import numpy as np

# -------------------------------
# Load the model
# -------------------------------
@st.cache_resource
def load_model():
    model = torch.load("MODEL/model.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

# -------------------------------
# UI Configuration
# -------------------------------
st.set_page_config(
    page_title="Auburn",
    page_icon="O",
    layout="centered"
)

# Custom CSS for a smoother design
st.markdown("""
    <style>
        .main {
            background-color: #F9FAFB;
        }
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 1px solid #ccc;
            padding: 0.5rem;
        }
        .stButton > button {
            background-color: #4F46E5;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton > button:hover {
            background-color: #4338CA;
        }
        .suggestion {
            background-color: #EEF2FF;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            margin: 0.2rem;
            display: inline-block;
            cursor: pointer;
            color: #4338CA;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.title(" Auburn")
st.caption("Try Auburn AI — type a query or click a suggestion!")

# -------------------------------
# Suggested Queries
# -------------------------------
st.markdown("**Suggested Queries:**")
suggestions = [
    "Sample input 1",
    "Sample input 2",
    "Sample input 3"
]

cols = st.columns(len(suggestions))
for i, s in enumerate(suggestions):
    if cols[i].button(s):
        st.session_state["user_query"] = s

# -------------------------------
# Query Input
# -------------------------------
query = st.text_input("Enter your query:", st.session_state.get("user_query", ""))

# -------------------------------
# Run Prediction
# -------------------------------
if st.button("🔍 Predict"):
    try:
        # Example: adapt this depending on your model
        x = torch.tensor([[float(len(query))]], dtype=torch.float32)  # placeholder
        with torch.no_grad():
            y = model(x)
        st.success(f"Model output: {y.item():.4f}")
    except Exception as e:
        st.error(f"Error: {e}")
