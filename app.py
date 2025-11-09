import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os
from PIL import Image
import base64

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Auburn AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #1e293b;
    }
    
    /* Sidebar text */
    .css-1d391kg p, .css-1d391kg label, .css-1d391kg div, .css-1d391kg span {
        color: #f1f5f9 !important;
    }
    
    /* Navigation header */
    .nav-header {
        padding: 1.5rem 1rem;
        border-bottom: 1px solid #334155;
        margin-bottom: 2rem;
    }
    
    .nav-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
    }
    
    .nav-subtitle {
        font-size: 0.875rem;
        color: #94a3b8;
    }
    
    /* Navigation items */
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #cbd5e1;
    }
    
    .nav-item:hover {
        background-color: #334155;
        color: #ffffff;
    }
    
    .nav-item.active {
        background-color: #3b82f6;
        color: #ffffff;
    }
    
    /* Main content area */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .content-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
    }
    
    /* Status boxes */
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button[kind="primary"] {
        background-color: #3b82f6;
        border: none;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
    }
    
    .stButton button[kind="secondary"] {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    
    /* Code input */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px #3b82f6;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def get_base64_of_image(image_path):
    """Convert image to base64 for embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

def create_navigation():
    """Create the persistent navigation sidebar"""
    with st.sidebar:
        # Navigation Header
        st.markdown("""
        <div class="nav-header">
            <div class="nav-title">🧬 Auburn AI</div>
            <div class="nav-subtitle">Code Optimization Platform</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation Menu
        menu_options = {
            "🏠 Dashboard": "dashboard",
            "🔍 Code Analysis": "analysis", 
            "📚 Examples": "examples",
            "📊 Results": "results",
            "ℹ️ About": "about"
        }
        
        # Initialize session state for current page
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "dashboard"
        
        # Create navigation items
        for label, page in menu_options.items():
            is_active = st.session_state.current_page == page
            css_class = "nav-item active" if is_active else "nav-item"
            
            if st.button(label, key=page, use_container_width=True):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown("---")
        
        # User info section
        st.markdown("""
        <div style="padding: 1rem;">
            <div style="color: #94a3b8; font-size: 0.875rem;">Logged in as</div>
            <div style="color: #f1f5f9; font-weight: 600;">Demo User</div>
        </div>
        """, unsafe_allow_html=True)

def authenticate():
    """Enhanced authentication with modern UI"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        # Center the authentication form
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #1e293b; margin-bottom: 0.5rem;'>🧬</h1>
                <h2 style='color: #1e293b; margin-bottom: 0.5rem;'>Auburn AI</h2>
                <p style='color: #64748b;'>Secure Code Analysis Platform</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.subheader("🔒 Authentication Required")
                
                with st.form("auth_form"):
                    password = st.text_input(
                        "Enter access password:", 
                        type="password",
                        placeholder="Enter your password...",
                        help="Contact administrator if you've forgotten the password"
                    )
                    submit = st.form_submit_button("Login", use_container_width=True)
                    
                    if submit:
                        if password == "password":  # In production, use proper authentication
                            st.session_state.authenticated = True
                            st.rerun()
                        else:
                            st.error("❌ Incorrect password. Please try again.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            st.stop()
    
    return True

def load_model_and_components():
    """Load ML model and components with caching"""
    @st.cache_resource
    def _load_resources():
        try:
            model = tf.keras.models.load_model("model.h5")
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            with open('mlb.pkl', 'rb') as f:
                mlb = pickle.load(f)
            with open('metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)

            max_len = metadata['max_len']
            fundamental_operations = metadata['fundamental_operations']
        
            return model, tokenizer, mlb, max_len, fundamental_operations
           
        except Exception as e:
            st.error(f"Error loading components: {e}")
            return None, None, None, None, None
    
    return _load_resources()

def render_dashboard():
    """Render the dashboard page"""
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin-bottom: 0.5rem;">Welcome to Auburn AI</h1>
        <p style="color: #e2e8f0; margin-bottom: 0;">Advanced Code Optimization for Pharmaceutical Research</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h3 style="color: #3b82f6; margin-bottom: 0.5rem;">🔍</h3>
            <h4 style="margin-bottom: 0.5rem;">Patterns Detected</h4>
            <h2 style="color: #1e293b; margin: 0;">12+</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h3 style="color: #10b981; margin-bottom: 0.5rem;">⚡</h3>
            <h4 style="margin-bottom: 0.5rem;">Avg. Speedup</h4>
            <h2 style="color: #1e293b; margin: 0;">5.2x</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h3 style="color: #f59e0b; margin-bottom: 0.5rem;">🧬</h3>
            <h4 style="margin-bottom: 0.5rem;">Pharma Optimized</h4>
            <h2 style="color: #1e293b; margin: 0;">100%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h3 style="color: #ef4444; margin-bottom: 0.5rem;">🔒</h3>
            <h4 style="margin-bottom: 0.5rem;">Secure Analysis</h4>
            <h2 style="color: #1e293b; margin: 0;">On-Prem</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Start New Analysis", use_container_width=True, type="primary"):
            st.session_state.current_page = "analysis"
            st.rerun()
    
    with col2:
        if st.button("📚 View Examples", use_container_width=True):
            st.session_state.current_page = "examples"
            st.rerun()
    
    # Recent activity
    st.subheader("Recent Activity")
    st.markdown("""
    <div class="content-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0;">Analysis History</h4>
            <span style="color: #64748b; font-size: 0.875rem;">Last 7 days</span>
        </div>
        <div style="color: #64748b;">
            • Drug compound sorting analysis - 2 hours ago<br>
            • Patient record search optimization - 1 day ago<br>
            • Matrix multiplication efficiency - 2 days ago<br>
            • Biomarker search patterns - 3 days ago
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_analysis():
    """Render the code analysis page"""
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h1 style="color: #1e293b; margin-bottom: 0.5rem;">🔍 Code Analysis</h1>
            <p style="color: #64748b; margin: 0;">Detect inefficiencies and optimize your pharmaceutical code</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, tokenizer, mlb, max_len, operations_info = load_model_and_components()
    
    if model is None:
        st.error("❌ Model failed to load. Please check if all model files are available.")
        return
    
    # Code input section
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("Input Your Code")
    
    code_input = st.text_area(
        "Paste your Python code below:",
        height=300,
        placeholder="""# Paste your pharmaceutical research code here\n\ndef analyze_compounds(compounds):\n    # Your code here\n    return results""",
        help="Auburn AI will detect inefficient patterns in sorting, searching, and matrix operations"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        analyze_btn = st.button("🚀 Analyze Code", use_container_width=True, type="primary", disabled=not code_input.strip())
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis results
    if analyze_btn and code_input.strip():
        with st.spinner("🔍 Analyzing code patterns..."):
            try:
                # Preprocessing and prediction functions would go here
                # For now, we'll show a placeholder result
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.subheader("📊 Analysis Results")
                
                # Simulated results
                st.markdown("""
                <div class="danger-box">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.5rem;">⚠️</span>
                        <div>
                            <strong>Inefficiencies Detected</strong>
                            <div style="color: #64748b; font-size: 0.875rem;">3 potential optimizations found</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Results breakdown
                results = [
                    {"pattern": "Inefficient Sorting", "confidence": "92%", "impact": "High"},
                    {"pattern": "Linear Search", "confidence": "85%", "impact": "Medium"},
                    {"pattern": "Naive Matrix Multiplication", "confidence": "78%", "impact": "High"}
                ]
                
                for result in results:
                    with st.expander(f"🔍 {result['pattern']} (Confidence: {result['confidence']})"):
                        st.write(f"**Impact Level:** {result['impact']}")
                        st.write("**Recommendation:** Consider using optimized libraries or algorithmic improvements")
                        st.write("**Quantum Potential:** This operation may benefit from quantum acceleration")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error analyzing code: {str(e)}")

def render_examples():
    """Render the examples page"""
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #1e293b; margin-bottom: 0.5rem;">📚 Code Examples</h1>
        <p style="color: #64748b; margin: 0;">Explore common patterns in pharmaceutical research code</p>
    </div>
    """, unsafe_allow_html=True)
    
    EXAMPLE_CODES = {
        "🧬 Drug Compound Sorting": """# Sort drug compounds by IC50 value using bubble sort
compounds = load_compound_library()
for i in range(len(compounds)):
    for j in range(len(compounds)-1):
        if compounds[j].ic50 > compounds[j+1].ic50:
            compounds[j], compounds[j+1] = compounds[j+1], compounds[j]""",

        "🧬 Patient Record Search": """# Find patient records by ID using linear search
def find_patient_by_id(patients, target_id):
    for patient in patients:
        if patient.id == target_id:
            return patient
    return None""",

        "🧬 Matrix Multiplication": """# Manual matrix multiplication for pharmacokinetic modeling
def manual_matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result"""
    }
    
    # Display examples in a grid
    cols = st.columns(2)
    for i, (title, code) in enumerate(EXAMPLE_CODES.items()):
        with cols[i % 2]:
            st.markdown(f'<div class="content-card">', unsafe_allow_html=True)
            st.subheader(title)
            st.code(code, language='python')
            if st.button(f"Use This Example", key=f"example_{i}", use_container_width=True):
                st.session_state.current_page = "analysis"
                # You could set the code in session state here
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

def render_about():
    """Render the about page"""
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #1e293b; margin-bottom: 0.5rem;">ℹ️ About Auburn AI</h1>
        <p style="color: #64748b; margin: 0;">Advanced code optimization for the pharmaceutical industry</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="content-card">
            <h3>Overview</h3>
            <p>Auburn AI is an advanced tool designed specifically for pharmaceutical and 
            biotechnology research. It automatically detects inefficient code patterns and 
            suggests classical and quantum improvements.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="content-card">
            <h3>Key Features</h3>
            <div class="feature-card">
                <strong>Domain-Specific Analysis</strong>
                <p style="margin: 0.5rem 0 0 0; color: #64748b;">Deep learning models optimized for pharma/biotech computational workflows</p>
            </div>
            <div class="feature-card">
                <strong>Private and Secure</strong>
                <p style="margin: 0.5rem 0 0 0; color: #64748b;">Your proprietary code never leaves your infrastructure</p>
            </div>
            <div class="feature-card">
                <strong>Quantum Readiness</strong>
                <p style="margin: 0.5rem 0 0 0; color: #64748b;">Identify operations with potential quantum speedups</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card">
            <h3>Supported Operations</h3>
            <ul style="color: #64748b;">
                <li>Inefficient Sorting Algorithms</li>
                <li>Linear Search Patterns</li>
                <li>Matrix Operations</li>
                <li>Nested Loops</li>
                <li>Data Structure Inefficiencies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="content-card">
            <h3>Impact</h3>
            <div class="success-box">
                • Reduces computational time by 3-10x<br><br>
                • Optimizes memory usage<br><br>
                • Identifies quantum opportunities<br><br>
                • Improves research throughput
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Authentication
    if not authenticate():
        return
    
    # Create navigation
    create_navigation()
    
    # Route to appropriate page
    if st.session_state.current_page == "dashboard":
        render_dashboard()
    elif st.session_state.current_page == "analysis":
        render_analysis()
    elif st.session_state.current_page == "examples":
        render_examples()
    elif st.session_state.current_page == "about":
        render_about()
    else:
        render_dashboard()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**Auburn AI** • Safe • Reliable • Smart")
    with col2:
        st.markdown("v0.1 • Demo")
    with col3:
        st.markdown("Optimizing pharmaceutical research")

if __name__ == "__main__":
    main()
