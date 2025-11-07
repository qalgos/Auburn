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
from PIL import Image

# Load your logo
logo = Image.open("image0.jpeg")

import base64

def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_of_image("image0.jpeg")

st.markdown(f"""
<style>
    .header-container {{
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 15px;
        margin-bottom: 2rem;
    }}
    .header-logo {{
        width: 140px;
        height: auto;
    }}
    .header-title {{
        margin: 0;
        color: #000000;
        font-size: 2.5rem;
        font-weight: 700;
    }}
</style>

<div class="header-container">
    <img src="data:image/png;base64,{logo_base64}" class="header-logo">
    <h1 class="header-title">Auburn</h1>
</div>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Auburn",
    page_icon= logo,
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
                st.write("Authentication required to access Auburn")
                
                with st.form("auth_form"):
                    password = st.text_input(
                        "Enter access password:", 
                        type="password",
                        help="Contact administrator if you've forgotten the password"
                    )
                    submit = st.form_submit_button("Login", use_container_width=True)
                    
                    if submit:
                        if password == "AuburninYC2026!":  # Replace with your actual password
                            st.session_state.authenticated = True
                            st.rerun()
                        else:
                            st.error("❌ Incorrect password. Please try again.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            st.stop()
    
    return True

st.markdown("""
<style>
    /* Main background - pure white */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Headers - black */
    .main-header {
        font-size: 3rem;
        color: #000000;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* All text black */
    .stMarkdown, .stText, p, div, span {
        color: #000000 !important;
    }
    
    /* Cards and containers - white with subtle borders */
    .feature-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #F0F0F0;
        margin-bottom: 1rem;
    }
    
    /* Status boxes - white with delicate colored accents */
    .success-box {
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #90EE90;
        border: 1px solid #F0F0F0;
    }
    
    .warning-box {
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #FFD700;
        border: 1px solid #F0F0F0;
    }
    
    .danger-box {
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #FFB6C1;
        border: 1px solid #F0F0F0;
    }
    
    /* Example cards - white with subtle borders */
    .example-card {
        background: #FFFFFF;
        border: 1px solid #F0F0F0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .example-card:hover {
        background: #FAFAFA;
        transform: translateY(-1px);
        border-color: #E8E8E8;
    }
    
    /* Sidebar styling - white */
    .css-1d391kg {
        background-color: #FFFFFF;
    }
    
    /* Buttons - delicate colors with white text */
    .stButton button {
        background-color: #F8F8FF;
        color: #000000;
        border: 1px solid #E8E8E8;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #F0F0F0;
        color: #000000;
        border-color: #D0D0D0;
    }
    
    /* Primary buttons - delicate purple */
    .stButton button[kind="primary"] {
        background-color: #F0E6FF;
        color: #000000;
        border: 1px solid #E0D6FF;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #E8DCFF;
        color: #000000;
    }
    
    /* Text areas and inputs - white */
    .stTextArea textarea, .stTextInput input {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        color: #000000;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #C0C0C0;
        box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.1);
    }
    
    /* Expanders - white */
    .streamlit-expanderHeader {
        background-color: #FFFFFF;
        border: 1px solid #F0F0F0;
        border-radius: 6px;
        color: #000000;
    }
    
    /* Progress bars - delicate purple */
    .stProgress > div > div > div {
        background-color: #E0D6FF;
    }
    
    /* Authentication container - delicate gradient */
    .auth-container {
        background: linear-gradient(135deg, #F0E6FF 0%, #F8F8FF 100%);
        padding: 2rem;
        border-radius: 15px;
        color: #000000;
        border: 1px solid #E8E8E8;
    }
    
    /* Radio buttons and other form elements */
    .stRadio > div {
        background-color: #FFFFFF;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #FAFAFA;
        border: 1px solid #F0F0F0;
    }
</style>
""", unsafe_allow_html=True)

# Check authentication before running app
if authenticate():
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Demo", "About"], index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.title("Auburn ai")
    
    # Load resources with caching
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
         
        # Convert to lowercase first for consistent matching
        code = code.lower()
        
        code = re.sub(r'\s+', ' ', code)
   
        
        # Normalize numbers
        code = re.sub(r'\b\d+\b', ' num ', code)
        
        # Clean up spaces
        code = re.sub(r'\s+', ' ', code).strip()
    
        
        return code

    # Example codes database
    EXAMPLE_CODES = {
        "🧬 Drug Compound Sorting (bubble sort mistake)": """#drug compounds by IC50 value
    compounds = load_compound_library()
    for i in range(len(compounds)):
        for j in range(len(compounds)-1):
            if compounds[j].ic50 > compounds[j+1].ic50:
                compounds[j], compounds[j+1] = compounds[j+1], compounds[j]""",
    
        "🔍 Patient Record Search (inefficient for a large database of patients)": """#for patient records by ID
    def find_patient_by_id(patients, target_id):
        for patient in patients:
            if patient.id == target_id:
                return patient
        return None""",
    
        "🧪Matrix Multiplication (not vectorized)": """# matrix multiplication for dose-response modeling
    def manual_matrix_multiply(A, B):
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result""",
    
        "📊 Clinical Trial Filtering": """# Linear filtering of clinical trial data
    def find_eligible_trials(trials, min_age, max_age, condition):
        eligible = []
        for trial in trials:
            if (trial.min_age <= min_age and 
                trial.max_age >= max_age and 
                condition in trial.conditions):
                eligible.append(trial)
        return eligible""",
    
        "⚗️ Molecular Weight Sorting": """# Selection sort for compounds by molecular weight
    def sort_compounds_by_weight(compounds):
        for i in range(len(compounds)):
            min_idx = i
            for j in range(i+1, len(compounds)):
                if compounds[j].molecular_weight < compounds[min_idx].molecular_weight:
                    min_idx = j
            compounds[i], compounds[min_idx] = compounds[min_idx], compounds[i]
        return compounds""",
    
        "🧫 Manual Statistical Calculations": """# Manual covariance calculation for gene expression
    gene_data = load_gene_expression_dataset()
    cov_matrix = []
    for i in range(len(gene_data[0])):
        row = []
        for j in range(len(gene_data[0])):
            cov = 0
            for k in range(len(gene_data)):
                cov += (gene_data[k][i] - mean_i) * (gene_data[k][j] - mean_j)
            row.append(cov / (len(gene_data) - 1))
        cov_matrix.append(row)""",
    
        "💊 Drug Interaction Search": """# Nested loop search for drug interactions
    def find_drug_interactions(drug, drug_library):
        interactions = []
        for other_drug in drug_library:
            if drug != other_drug:
                affinity = calculate_binding_affinity(drug, other_drug)
                if affinity < 10:  # Strong binding
                    interactions.append(other_drug)
        return interactions"""
    }
    

    # ABOUT PAGE
    if page == "About":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Overview")
            st.markdown("""
            <div class="feature-card">
            Auburn is an advanced AI-powered tool designedspecifically for 
            the pharmaceutical and biotechnology industries. It automatically detects inefficient 
            code patterns and suggests improvements. It also evaluates quantum speedup potential, and quantifies how much your business can benefit from the adoption of quantum computers.
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Key Features")
            
            features = [
                ("🧠 AI-Powered Analysis", "Deep learning models trained to recognise inefficiencies in many programming languages"),
                ("⚡ Performance Optimization", "This demo identifies bottlenecks in sorting, searching, and matrix operations"),
                ("🧬 Domain-Specific", "Optimized for pharma/biotech computational workflows"),
                ("🔒 Private and Secure", "You can scan your codebase without worrying about leaks"),
                ("📊 Detailed Reporting", "Comprehensive analysis with improvement suggestions and quantum potential evaluation.")
            ]
            
            for feature, description in features:
                with st.expander(f"**{feature}**"):
                    st.write(description)
        
        with col2:
            st.subheader("Supported operations")
            st.markdown("""
            - **Sorting**
            - **Database Search**  
            - **Matrix Operations**
            - **Nested Loops**
            - **Suboptimal Data Structures**
            - **Redundant Calculations**
            """)
            
            st.subheader("📈 Impact")
            st.markdown("""
            <div class="success-box">
            Average performance improvement
            Reduced computational time 
            Memory optimization 
            Quantum speedup potential
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("How it works?")
        
        tech_cols = st.columns(2)
        with tech_cols[0]:
            st.markdown("""
            **Machine Learning Stack**
            - TensorFlow/Keras
            - State-of-the-art Natural Language Processing tools
            - Knowledge of quantum algorithms
            """)
        with tech_cols[1]:
            st.markdown("""
            **Processing capabilities**
            - Capable of detecting many inefficient operations with just one run
            - Pre-trained model guarantess fast processing time
            - Accepts many programming languages
            """)
            
        st.markdown("---")
        st.subheader("See who trusted us")
        st.markdown("""
        **Reviews**
        - Name Surname, MEng Chemical Engineering: ""
        - Name Surname, PhD Computational Chemistry: ""
        - Name Surname, PhD Computational Chemistry: ""
        - Name Surname, MD: ""
        """)

    # DEMO PAGE
    else:

        #st.subheader("Detect inefficiencies in pharma/biotech codebases")
        st.text(
            "Auburn AI detects inefficient code implementation and screens for classical and quantum speedups available in your code.")
        # Quick stats in sidebar
        with st.sidebar:
            st.info("""
            **Capabilities**
            - 5+ inefficiency patterns
            - Real-time analysis
            - Confidence scoring
            - Optimization suggestions
            - Evaluating quantum speedup potential 
            """)
            
            if st.button("🔄 Clear Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Example Gallery Section
        st.subheader("Example Code Gallery")
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
            st.markdown(f"**Example Loaded:** {st.session_state.get('selected_example', 'Selected Code')}")
            st.code(st.session_state.example_code, language='python')
            
           
        
        st.markdown("---")
        
        # Main Analysis Section
        st.subheader("Code Analysis")
        
        # Initialize session state
        if 'analysis_code' not in st.session_state:
            st.session_state.analysis_code = ""
        
        # Code input area
        code_input = st.text_area(
            "Paste or modify your Python code here:", 
            height=250,
            value=st.session_state.analysis_code,
            placeholder="""# Paste your code here or use an example above\n\ndef your_function():\n    # Your code here\n    return result""",
            help="Auburn will detect inefficient patterns in sorting, searching, and matrix operations"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            analyze_clicked = st.button(
                "🚀 Analyze Code", 
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
                        
                        processed_code = preprocess_code(code_input)
                        sequence = tokenizer.texts_to_sequences([processed_code])
                        padded_sequence = pad_sequences(sequence, padding='post')
                        
                        predictions = model.predict(padded_sequence, verbose=0)
                        binary_predictions = (predictions > 0.5).astype(int)
                        predicted_labels = mlb.inverse_transform(binary_predictions)
                        
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
            
           
        
        st.markdown("---")
        
        # Main Analysis Section
        st.subheader("Code Analysis")
        
        # Initialize session state
        if 'analysis_code' not in st.session_state:
            st.session_state.analysis_code = ""
        
        # Code input area
        code_input = st.text_area(
            "Paste or modify your Python code here:", 
            height=250,
            value=st.session_state.analysis_code,
            placeholder="""# Paste your code here or use an example above\n\ndef your_function():\n    # Your code here\n    return result""",
            help="Auburn will detect inefficient patterns in sorting, searching, and matrix operations"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            analyze_clicked = st.button(
                "🚀 Analyze Code", 
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
                        
                        processed_code = preprocess_code(code_input)
                        sequence = tokenizer.texts_to_sequences([processed_code])
                        padded_sequence = pad_sequences(sequence, padding='post')
                        
                        predictions = model.predict(padded_sequence, verbose=0)
                        binary_predictions = (predictions > 0.5).astype(int)
                        predicted_labels = mlb.inverse_transform(binary_predictions)
                        
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
        st.markdown("**Auburn** • Built with Streamlit & TensorFlow")
    with col2:
        st.markdown("Detects inefficient patterns in scientific computing")
    with col3:
        st.markdown("v2.0 • Professional Edition")
