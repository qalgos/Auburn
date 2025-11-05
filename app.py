import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import io
import base64
import time
import os

# Set page config
st.set_page_config(
    page_title="Code Efficiency Analyzer | Pharma/Biotech",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 400;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Example gallery buttons */
    .example-btn {
        width: 100%;
        margin: 0.3rem 0;
        border-radius: 8px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Code block styling */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Analysis results */
    .result-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    
    .warning-card {
        background: #fff3cd;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
    
    .error-card {
        background: #f8d7da;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

def authenticate():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        # Centered authentication layout
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🔒</h1>
                <h2 style='color: #333;'>Secure Access Portal</h2>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("auth_form"):
                password = st.text_input("Enter Access Password", type="password", 
                                       placeholder="Enter your password...")
                submit = st.form_submit_button("Login to Platform", use_container_width=True)
                
                if submit:
                    if password == "my password":  # Replace with your actual password
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Incorrect password. Please try again.")
            
            st.info("Contact administrator for access credentials")
            st.stop()
    
    return True

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Free Demo"

def navigation():
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 1.8rem; margin-bottom: 0.5rem; color: #333;">Code Efficiency Analyzer</h1>
        <p style="color: #666; font-size: 0.9rem;">Pharmaceutical & Biotech Applications</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Navigation buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Free Demo", use_container_width=True, 
                    type="primary" if st.session_state.current_page == "Free Demo" else "secondary"):
            st.session_state.current_page = "Free Demo"
            st.rerun()
    
    with col2:
        if st.button("About", use_container_width=True,
                    type="primary" if st.session_state.current_page == "About" else "secondary"):
            st.session_state.current_page = "About"
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Platform info in sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 8px; color: white;">
        <h4 style="color: white; margin-bottom: 0.5rem;">Platform Features</h4>
        <ul style="color: white; font-size: 0.8rem; padding-left: 1rem;">
            <li>AI-powered code analysis</li>
            <li>Pharma-specific patterns</li>
            <li>Performance optimization</li>
            <li>Best practices guidance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# About page
def render_about():
    st.markdown('<div class="main-header">About the Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Code Efficiency Analysis for Pharmaceutical and Biotech Applications</div>', unsafe_allow_html=True)
    
    # Platform overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3>Our Mission</h3>
            <p>We provide cutting-edge code analysis specifically designed for pharmaceutical and biotech applications, 
            helping researchers and developers optimize computational workflows and accelerate drug discovery processes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3>Technology Stack</h3>
            <p><strong>Machine Learning:</strong> TensorFlow/Keras</p>
            <p><strong>Frontend:</strong> Streamlit</p>
            <p><strong>Analysis:</strong> Custom NLP pipelines</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features grid
    st.subheader("Core Capabilities")
    
    features = [
        {
            "title": "Algorithm Analysis",
            "description": "Detect inefficient sorting, searching, and computational patterns in scientific code"
        },
        {
            "title": "Performance Optimization",
            "description": "Identify bottlenecks and suggest optimized implementations for better performance"
        },
        {
            "title": "Domain-Specific Patterns",
            "description": "Specialized analysis for pharmaceutical and biotech computational workflows"
        },
        {
            "title": "Best Practices",
            "description": "Provide industry-standard recommendations for scientific computing"
        }
    ]
    
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="feature-card">
                <h4>{feature['title']}</h4>
                <p>{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Use cases
    st.subheader("Common Use Cases")
    
    use_cases = [
        "Drug compound analysis and sorting algorithms",
        "Patient data processing and search optimization", 
        "Clinical trial data filtering and analysis",
        "Molecular dynamics simulations",
        "Genomic data processing pipelines",
        "Statistical analysis of experimental results"
    ]
    
    for use_case in use_cases:
        st.markdown(f"• {use_case}")

# Free Demo page with full functionality
def render_demo():
    st.markdown('<div class="main-header">Code Efficiency Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detect and optimize inefficiencies in pharmaceutical and biotech codebases</div>', unsafe_allow_html=True)
    
    # Example codes database
    EXAMPLE_CODES = {
        "Drug Compound Sorting": """# Bubble sort for drug compounds by IC50 value
compounds = load_compound_library()
for i in range(len(compounds)):
    for j in range(len(compounds)-1):
        if compounds[j].ic50 > compounds[j+1].ic50:
            compounds[j], compounds[j+1] = compounds[j+1], compounds[j]""",

        "Patient Record Search": """# Linear search for patient records by ID
def find_patient_by_id(patients, target_id):
    for patient in patients:
        if patient.id == target_id:
            return patient
    return None""",

        "Matrix Operations": """# Manual matrix multiplication for dose-response modeling
def manual_matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result""",

        "Clinical Trial Filtering": """# Linear filtering of clinical trial data
def find_eligible_trials(trials, min_age, max_age, condition):
    eligible = []
    for trial in trials:
        if (trial.min_age <= min_age and 
            trial.max_age >= max_age and 
            condition in trial.conditions):
            eligible.append(trial)
    return eligible""",

        "Molecular Weight Sorting": """# Selection sort for compounds by molecular weight
def sort_compounds_by_weight(compounds):
    for i in range(len(compounds)):
        min_idx = i
        for j in range(i+1, len(compounds)):
            if compounds[j].molecular_weight < compounds[min_idx].molecular_weight:
                min_idx = j
        compounds[i], compounds[min_idx] = compounds[min_idx], compounds[i]
    return compounds""",

        "Statistical Calculations": """# Manual covariance calculation for gene expression
gene_data = load_gene_expression_dataset()
cov_matrix = []
for i in range(len(gene_data[0])):
    row = []
    for j in range(len(gene_data[0])):
        cov = 0
        for k in range(len(gene_data)):
            cov += (gene_data[k][i] - mean_i) * (gene_data[k][j] - mean_j)
        row.append(cov / (len(gene_data) - 1))
    cov_matrix.append(row)"""
    }
    
    # Example Gallery Section
    st.markdown("""
    <div class="custom-card">
        <h3>Example Code Gallery</h3>
        <p>Select from common pharmaceutical code patterns to analyze:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for examples
    cols = st.columns(3)
    example_items = list(EXAMPLE_CODES.items())
    
    for i, (title, code) in enumerate(example_items):
        with cols[i % 3]:
            if st.button(title, use_container_width=True, key=f"btn_{title}"):
                st.session_state.example_code = code
                st.session_state.selected_example = title
    
    # Display selected example
    if 'example_code' in st.session_state:
        st.subheader(f"Selected Example: {st.session_state.get('selected_example', 'Code')}")
        st.code(st.session_state.example_code, language='python')
        
        if st.button("Analyze This Example", type="primary", use_container_width=True):
            st.session_state.analysis_code = st.session_state.example_code

    # Analysis Section
    st.markdown("---")
    st.subheader("Code Analysis")
    
    # Initialize session state
    if 'analysis_code' not in st.session_state:
        st.session_state.analysis_code = ""
    
    # Code input area
    code_input = st.text_area(
        "Paste your Python code here:",
        height=250,
        value=st.session_state.analysis_code,
        placeholder="Paste your pharmaceutical/biotech code here...\n\n# Example:\ndef analyze_compound(compound):\n    # Your code here\n    return result",
        help="Enter Python code to analyze for computational inefficiencies"
    )
    
    # Analysis controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analyze_btn = st.button("Analyze Code", type="primary", use_container_width=True)
    
    if analyze_btn:
        if code_input.strip():
            with st.spinner("Analyzing code patterns..."):
                # Simulate analysis progress
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Display comprehensive results
                st.subheader("Analysis Results")
                
                # Mock analysis results - replace with your actual model predictions
                st.markdown("""
                <div class="error-card">
                    <h4>Inefficiencies Detected</h4>
                    <ul>
                    <li><strong>Inefficient Sorting Algorithm</strong>: Bubble sort detected - consider using built-in sorted() or numpy.argsort()</li>
                    <li><strong>Linear Search Pattern</strong>: O(n) complexity - consider using dictionary lookups or binary search</li>
                    <li><strong>Manual Matrix Operations</strong>: Nested loops detected - use NumPy vectorized operations</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="result-card">
                    <h4>Optimization Suggestions</h4>
                    <ul>
                    <li>Replace bubble sort with optimized sorting algorithms (O(n log n) complexity)</li>
                    <li>Use hash tables for patient record lookups (O(1) average case)</li>
                    <li>Leverage NumPy for matrix operations (significant speedup for large datasets)</li>
                    <li>Consider parallel processing for large-scale data analysis</li>
                    <li>Implement caching for frequently accessed data</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Performance metrics
                with st.expander("Detailed Performance Analysis"):
                    st.markdown("""
                    | Pattern | Confidence | Severity | Impact |
                    |---------|------------|----------|---------|
                    | Inefficient Sorting | 92% | High | 40-60x slower |
                    | Linear Search | 87% | Medium | 10-100x slower |
                    | Manual Matrix Ops | 95% | High | 100-1000x slower |
                    """)
                    
                    st.markdown("""
                    **Performance Impact Summary:**
                    - Current implementation estimated runtime: ~15 seconds
                    - Optimized implementation estimated runtime: ~0.15 seconds
                    - **Potential speedup: 100x**
                    """)
                
                # Code suggestions
                with st.expander("Optimized Code Examples"):
                    st.code("""
# Optimized sorting using NumPy
compound_ic50 = np.array([c.ic50 for c in compounds])
sorted_indices = np.argsort(compound_ic50)
sorted_compounds = [compounds[i] for i in sorted_indices]

# Optimized search using dictionary
patient_dict = {p.id: p for p in patients}
patient = patient_dict.get(target_id)

# Optimized matrix operations
result = np.dot(A, B)  # Instead of manual multiplication
                    """, language='python')
        else:
            st.warning("Please enter some code to analyze")

    # Clear button
    if st.button("Clear Analysis", use_container_width=True):
        st.session_state.analysis_code = ""
        if 'example_code' in st.session_state:
            del st.session_state.example_code
        if 'selected_example' in st.session_state:
            del st.session_state.selected_example
        st.rerun()

# Main app logic
def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Check authentication
    if not authenticate():
        return
    
    # Navigation
    navigation()
    
    # Route to appropriate page
    if st.session_state.current_page == "About":
        render_about()
    else:  # Free Demo
        render_demo()

if __name__ == "__main__":
    main()
