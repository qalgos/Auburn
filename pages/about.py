
import streamlit as st

st.set_page_config(
    page_title="About - Pharma Analyzer",
    page_icon="🧪",
    layout="wide"
)

# Same menu CSS as main app
st.markdown("""
<style>
    .main-menu {
        background-color: #f8f9fa;
        padding: 1rem 2rem;
        border-bottom: 2px solid #A52A2A;
        margin-bottom: 2rem;
        border-radius: 0 0 10px 10px;
    }
    .menu-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
    }
    .menu-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #A52A2A;
        text-decoration: none;
    }
    .menu-links {
        display: flex;
        gap: 2rem;
    }
    .menu-link {
        color: #2C2C2C;
        text-decoration: none;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .menu-link:hover {
        background-color: #A52A2A;
        color: white;
        text-decoration: none;
    }
    .menu-link.active {
        background-color: #A52A2A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Navigation Menu
st.markdown("""
<div class="main-menu">
    <div class="menu-container">
        <a href="/" class="menu-title">🧪 Pharma Code Analyzer</a>
        <div class="menu-links">
            <a href="/" class="menu-link">Home</a>
            <a href="/about" class="menu-link active">About</a>
            <a href="https://github.com/yourusername/your-repo" class="menu-link" target="_blank">GitHub</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.title("About Pharma Code Efficiency Analyzer")
st.write("Content from your about.html page would go here...")
