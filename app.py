import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import io
import base64
import time
import re
import os

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Auburn",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def authenticate():
    """Enhanced authentication with better UI"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        # Custom CSS for auth page
        st.markdown("""
            <style>
            .auth-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.container():
                st.markdown('<div class="auth-container">', unsafe_allow_html=True)
                st.title("🔒 Secure Access")
                st.write("Authentication required to access the Code Efficiency Analyzer")
                
                with st.form("auth_form"):
                    password = st.text_input(
                        "Enter access password:", 
                        type="password",
                        help="Contact administrator if you've forgotten the password"
                    )
                    submit = st.form_submit_button("🚀 Login", use_container_width=True)
                    
                    if submit:
                        if password == "AuburninYC2026!":  # Replace with your actual password
                            st.session_state.authenticated = True
                            st.rerun()
                        else:
                            st.error("❌ Incorrect password. Please try again.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            st.stop()
    
    return True

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
    .danger-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
    }
    .example-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .example-card:hover {
        background: #e9ecef;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Check authentication before running app
if authenticate():
    # Navigation
    st.sidebar.title("🧬 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Demo", "ℹ️ About"], index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.title("🔧 Quick Actions")
    
    # Load resources with caching
    @st.cache_resource
    def load_model():
        try:
            model = tf.keras.models.load_model("model.h5")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    @st.cache_resource
    def load_components():
        try:
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            with open('mlb.pkl', 'rb') as f:
                mlb = pickle.load(f)
            with open('metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            return tokenizer, mlb, metadata
        except Exception as e:
            st.error(f"Error loading components: {e}")
            return None, None, None
    
    def preprocess_code(code):
        # Keep your existing preprocessing logic
        return code

    # Example codes database
    EXAMPLE_CODES = {
        "🧬 Drug Compound Sorting": """# Bubble sort for drug compounds by IC50 value
compounds = load_compound_library()
for i in range(len(compounds)):
    for j in range(len(compounds)-1):
        if compounds[j].ic50 > compounds[j+1].ic50:
            compounds[j], compounds[j+1] = compounds[j+1], compounds[j]""",

        "🔍 Patient Record Search": """# Linear search for patient records by ID
def find_patient_by_id(patients, target_id):
    for patient in patients:
        if patient.id == target_id:
            return patient
    return None""",

        "🧪 Manual Matrix Operations": """# Manual matrix multiplication for dose-response modeling
def manual_matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result""",
    }

    # ABOUT PAGE
    if page == "About":
        st.markdown('<h1 class="main-header">Auburn</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Overview")
            st.markdown("""
            <div class="feature-card">
            The **Code Efficiency Analyzer** is an advanced AI-powered tool specifically designed for 
            the pharmaceutical and biotechnology industries. It automatically detects inefficient 
            code patterns in scientific computing and data analysis workflows.
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Key Features")
            
            features = [
                ("🧠 AI-Powered Analysis", "Deep learning models trained on thousands of code examples"),
                ("⚡ Performance Optimization", "Identifies bottlenecks in sorting, searching, and matrix operations"),
                ("🧬 Domain-Specific", "Optimized for pharma/biotech computational workflows"),
                ("📊 Detailed Reporting", "Comprehensive analysis with confidence scores and improvement suggestions")
            ]
            
            for feature, description in features:
                with st.expander(f"**{feature}**"):
                    st.write(description)
        
        with col2:
            st.subheader("🔍 Supported Patterns")
            st.markdown("""
            - **Inefficient Sorting Algorithms**
            - **Linear Search Patterns**  
            - **Manual Matrix Operations**
            - **Nested Loop Inefficiencies**
            - **Suboptimal Data Structures**
            - **Redundant Calculations**
            """)
            
            st.subheader("📈 Impact")
            st.markdown("""
            <div class="success-box">
            **Potential performance improvements**: 3-10x  
            **Reduced computation time**:   
            **Memory optimization**: 
            **Quantum speedup potential estimation**
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🛠️ Technical Architecture")
        
        tech_cols = st.columns(3)
        with tech_cols[0]:
            st.markdown("""
            **Machine Learning Stack**
            - TensorFlow/Keras
            - Custom NLP Pipeline
            - Multi-label Classification
            """)
        with tech_cols[1]:
            st.markdown("""
            **Data Processing**
            - Abstract Syntax Trees
            - Code Tokenization
            - Pattern Recognition
            """)
        with tech_cols[2]:
            st.markdown("""
            **Performance Metrics**
            - Confidence Scoring
            - Pattern Matching
            - Optimization Suggestions
            """)

    # DEMO PAGE
    else:
        st.markdown('<h1 class="main-header">Auburn</h1>', unsafe_allow_html=True)
        st.subheader("Detect inefficiencies in pharma/biotech codebases")
        
        # Quick stats in sidebar
        with st.sidebar:
            st.info("""
            **📊 Detection Capabilities**
            - 5+ inefficiency patterns
            - Real-time analysis
            - Confidence scoring
            - Optimization suggestions
            - Quantum speedup potential evaluation
            """)
            
            if st.button("🔄 Clear Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Example Gallery Section
        st.subheader("📚 Example Code Gallery")
        st.write("Click on examples to load them into the analyzer:")
        
        # Create columns for examples
        cols = st.columns(2)
        example_titles = list(EXAMPLE_CODES.keys())
        
        for i, title in enumerate(example_titles):
            with cols[i % 2]:
                if st.button(
                    title, 
                    use_container_width=True, 
                    key=f"btn_{title}",
                    help=f"Load {title} example"
                ):
                    st.session_state.example_code = EXAMPLE_CODES[title]
                    st.session_state.selected_example = title
                    st.session_state.analysis_code = EXAMPLE_CODES[title]
        
        # Display selected example
        if 'example_code' in st.session_state:
            st.markdown(f"**📋 Example Loaded:** {st.session_state.get('selected_example', 'Selected Code')}")
            st.code(st.session_state.example_code, language='python')
            
            if st.button("🔍 Analyze This Example", type="primary", use_container_width=True):
                st.session_state.analyze_clicked = True
        
        st.markdown("---")
        
        # Main Analysis Section
        st.subheader("🔍 Code Analysis")
        
        # Initialize session state
        if 'analysis_code' not in st.session_state:
            st.session_state.analysis_code = ""
        
        # Code input area
        code_input = st.text_area(
            "Paste or modify your Python code here:", 
            height=250,
            value=st.session_state.analysis_code,
            placeholder="""# Paste your code here or use an example above\n\ndef your_function():\n    # Your code here\n    return result""",
            help="The analyzer will detect inefficient patterns in sorting, searching, and matrix operations"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            analyze_clicked = st.button(
                "Analyze Code", 
                type="primary", 
                use_container_width=True,
                disabled=not code_input.strip()
            )
        
        with col2:
            if st.button("🗑️ Clear Code", use_container_width=True):
                st.session_state.analysis_code = ""
                if 'example_code' in st.session_state:
                    del st.session_state.example_code
                if 'selected_example' in st.session_state:
                    del st.session_state.selected_example
                st.rerun()
        
        # Analysis execution
        if analyze_clicked or st.session_state.get('analyze_clicked', False):
            if 'analyze_clicked' in st.session_state:
                del st.session_state.analyze_clicked
                
            if code_input.strip():
                with st.spinner("🔍 Analyzing code patterns..."):
                    try:
                        model = load_model()
                        tokenizer, mlb, metadata = load_components()
                        
                        if model is None or tokenizer is None:
                            st.error("❌ Required model files not found. Please ensure all model files are available.")
                            st.stop()
                        
                        # Preprocess and predict
                        processed_code = preprocess_code(code_input)
                        sequence = tokenizer.texts_to_sequences([processed_code])
                        padded_sequence = pad_sequences(sequence, maxlen=metadata['max_len'], padding='post')
                        
                        # Keras prediction
                        predictions = model.predict(padded_sequence, verbose=0)
                        
                        # Get results
                        binary_predictions = (predictions > 0.5).astype(int)
                        predicted_labels = mlb.inverse_transform(binary_predictions)
                        
                        # Get confidence scores
                        confidence_scores = {}
                        for i, label in enumerate(mlb.classes_):
                            confidence_scores[label] = float(predictions[0][i])
                        
                        # Display results
                        st.subheader("📊 Analysis Results")
                        
                        if predicted_labels[0]:
                            st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                            st.error("🚨 Inefficiencies Detected")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            for label in predicted_labels[0]:
                                confidence = confidence_scores.get(label, 0) * 100
                                with st.container():
                                    col_a, col_b = st.columns([3, 1])
                                    with col_a:
                                        st.write(f"**{label}**")
                                    with col_b:
                                        st.write(f"`{confidence:.1f}%`")
                                
                                if label in metadata.get('fundamental_operations', {}):
                                    info = metadata['fundamental_operations'][label]
                                    with st.expander(f"📖 Details & Recommendations for {label}"):
                                        st.write(f"**Description**: {info.get('description', 'N/A')}")
                                        st.write(f"**Quantum Speedup**: {info.get('quantum_speedup', 'N/A')}")
                                        st.write(f"**Optimization**: {info.get('optimization_notes', 'N/A')}")
                        else:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.success("✅ No inefficiencies detected!")
                            st.write("The code appears to use efficient implementations.")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        # Detailed confidence scores
                        with st.expander("🔍 Detailed Confidence Scores"):
                            st.write("All detected patterns with confidence levels:")
                            for label, confidence in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
                                progress_value = confidence
                                st.write(f"**{label}**")
                                st.progress(progress_value, text=f"{confidence:.1%} confidence")
                                
                    except Exception as e:
                        st.error(f"❌ Error analyzing code: {str(e)}")
                        st.info("""
                        **Troubleshooting tips:**
                        - Ensure all model files (model.h5, tokenizer.pkl, mlb.pkl, metadata.pkl) are present
                        - Check that the code is valid Python syntax
                        - Try using one of the example codes above
                        """)
            else:
                st.warning("⚠️ Please enter some code to analyze")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**Code Efficiency Analyzer** • Built with Streamlit & TensorFlow")
    with col2:
        st.markdown("Detects inefficient patterns in scientific computing")
    with col3:
        st.markdown("v2.0 • Professional Edition")
