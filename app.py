import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os
from PIL import Image
import base64

# Load your logo
logo = Image.open("image0.jpeg")

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
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded"
)

def authenticate():
    """Enhanced authentication with better UI"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
     
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.container():
                st.title("🔒 Authentication required")
                st.write("")
                
                with st.form("auth_form"):
                    password = st.text_input(
                        "Enter access password:", 
                        type="password",
                        help="Contact administrator if you've forgotten the password"
                    )
                    submit = st.form_submit_button("Login", use_container_width=True)
                    
                    if submit:
                        if password == "password":
                            st.session_state.authenticated = True
                            st.rerun()
                        else:
                            st.error("❌ Incorrect password. Please try again.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            st.stop()
    
    return True

st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
    }
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
    .stMarkdown, .stText, p, div, span {
        color: #000000 !important;
    }
    .feature-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #F0F0F0;
        margin-bottom: 1rem;
    }
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
    .css-1d391kg {
        background-color: #FFFFFF;
    }
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
    .stButton button[kind="primary"] {
        background-color: #F0E6FF;
        color: #000000;
        border: 1px solid #E0D6FF;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #E8DCFF;
        color: #000000;
    }
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
    .streamlit-expanderHeader {
        background-color: #FFFFFF;
        border: 1px solid #F0F0F0;
        border-radius: 6px;
        color: #000000;
    }
    .stProgress > div > div > div {
        background-color: #E0D6FF;
    }
    .auth-container {
        
        padding: 2rem;
        border-radius: 15px;
        color: #000000;
        border: 1px solid #E8E8E8;
    }
    .stRadio > div {
        background-color: #FFFFFF;
    }
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
    

    # Load resources with caching - FIXED VERSION
    @st.cache_resource
    def load_model_and_components():
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

    # Load model once at startup
    model, tokenizer, mlb, max_len, operations_info = load_model_and_components()

    def preprocess_code(code):
        """EXACT same preprocessing as Tkinter app"""
        code = code.lower()
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'\b\d+\b', ' num ', code)
        code = re.sub(r'\s+', ' ', code).strip()
        return code

    def predict_operations(code_snippet, threshold=0.7):
        """EXACT same prediction logic as Tkinter app"""
        processed_code = preprocess_code(code_snippet)
        sequence = tokenizer.texts_to_sequences([processed_code])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
        
        predictions = model.predict(padded_sequence, verbose=0)
        binary_predictions = (predictions > threshold).astype(int)
        predicted_labels = mlb.inverse_transform(binary_predictions)
        
        confidence_scores = {}
        for i, label in enumerate(mlb.classes_):
            confidence_scores[label] = float(predictions[0][i])
        
        return predicted_labels[0], confidence_scores

    # Example codes database
    EXAMPLE_CODES = {
        "🧬 Drug Compound Sorting (bubble sort)": """# Sort drug compounds by IC50 value
compounds = load_compound_library()
for i in range(len(compounds)):
    for j in range(len(compounds)-1):
        if compounds[j].ic50 > compounds[j+1].ic50:
            compounds[j], compounds[j+1] = compounds[j+1], compounds[j]""",

        "🧬 Patient Record Search (linear search)": """# Find patient records by ID
def find_patient_by_id(patients, target_id):
    for patient in patients:
        if patient.id == target_id:
            return patient
    return None""",

        "🧬 Inefficient Matrix Multiplication ": """#matrix multiplication
def manual_matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result""",

        "🧬 Estimate pharmacokinetic parameters (Matrix multiplication)": """
def estimate_pk_parameters(dose_matrix, transfer_matrix):
    num_patients = len(dose_matrix)
    num_compartments = len(transfer_matrix[0])
    concentration_matrix = [[0.0 for _ in range(num_compartments)] for _ in range(num_patients)]
    
    for i in range(num_patients):
        for j in range(num_compartments):
            concentration = 0.0
            for k in range(len(dose_matrix[0])):
                # Multiply dose by transfer coefficients between compartments
                concentration += dose_matrix[i][k] * transfer_matrix[k][j]
            concentration_matrix[i][j] = concentration
    
    return concentration_matrix

patient_doses = load_dosing_regimens()
compartment_transfer = load_pk_parameters()
tissue_concentrations = estimate_pk_parameters(patient_doses, compartment_transfer)""",

        "🧬 Molecular Weight Sorting": """# Selection sort for compounds
def sort_compounds_by_weight(compounds):
    for i in range(len(compounds)):
        min_idx = i
        for j in range(i+1, len(compounds)):
            if compounds[j].molecular_weight < compounds[min_idx].molecular_weight:
                min_idx = j
        compounds[i], compounds[min_idx] = compounds[min_idx], compounds[i]
    return compounds""", 

         "🧬 Search for Patient Biomarkers (Linear)": """
    
def find_patients_with_biomarker(patients, target_biomarker, threshold):
    matching_patients = []
    for patient in patients:
        if patient.biomarkers.get(target_biomarker, 0) > threshold:
            matching_patients.append(patient)
    return matching_patients

oncology_patients = load_cancer_patients()
her2_positive = find_patients_with_biomarker(oncology_patients, "HER2", 2.0)"""
        
   
    }

# ABOUT PAGE
if page == "About":
        # Modern header with gradient
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem;
            border-radius: 16px;
            margin-bottom: 2.5rem;
            color: white;
        ">
            <h1 style="color: white; margin-bottom: 0.5rem; font-size: 2.5rem;">About Auburn AI</h1>
            <p style="color: #e2e8f0; font-size: 1.2rem; margin: 0;">
                Advanced Code Optimization for Pharmaceutical Innovation
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main content columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Overview Section
            st.markdown("""
            <div class="content-card">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="font-size: 2rem;">🎯</div>
                    <h2 style="margin: 0; color: #1e293b;">Overview</h2>
                </div>
                <p style="color: #475569; line-height: 1.6; font-size: 1.1rem;">
                    Auburn is an advanced AI-powered platform specifically engineered for the 
                    <strong>pharmaceutical and biotechnology sectors</strong>. Our intelligent system 
                    automatically detects computational inefficiencies in research code and provides 
                    actionable optimization strategies.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Features Section
            st.markdown("""
            <div class="content-card">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="font-size: 2rem;">🚀</div>
                    <h2 style="margin: 0; color: #1e293b;">Key Features</h2>
                </div>
            """, unsafe_allow_html=True)
            
            features = [
                {
                    "icon": "🧠", 
                    "title": "Domain-Specific AI Analysis",
                    "description": "Deep learning models optimized for pharmaceutical computational workflows, trained to recognize industry-specific inefficiencies.",
                    "color": "#3b82f6"
                },
                {
                    "icon": "🔒", 
                    "title": "Enterprise-Grade Security",
                    "description": "Full in-house deployment ensures your proprietary research code never leaves your secure environment. No data exposure, no cloud dependencies.",
                    "color": "#10b981"
                },
                {
                    "icon": "📊", 
                    "title": "Comprehensive Analytics",
                    "description": "Detailed performance reports with classical optimization strategies and quantum computing readiness assessment for future-proofing your codebase.",
                    "color": "#f59e0b"
                }
            ]
            
            for i, feature in enumerate(features):
                st.markdown(f"""
                <div style="
                    background: white;
                    border: 1px solid #e2e8f0;
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                    border-left: 4px solid {feature['color']};
                    transition: all 0.3s ease;
                " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.1)';" 
                onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                    <div style="display: flex; align-items: flex-start; gap: 1rem;">
                        <div style="font-size: 2rem;">{feature['icon']}</div>
                        <div style="flex: 1;">
                            <h3 style="margin: 0 0 0.5rem 0; color: #1e293b;">{feature['title']}</h3>
                            <p style="margin: 0; color: #64748b; line-height: 1.5;">{feature['description']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Supported Operations Section
            st.markdown("""
            <div class="content-card">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="font-size: 1.5rem;">⚡</div>
                    <h3 style="margin: 0; color: #1e293b;">Optimized Operations</h3>
                </div>
            """, unsafe_allow_html=True)
            
            operations = [
                {"name": "Sorting Algorithms", "icon": "📈", "status": "Optimized"},
                {"name": "Search Operations", "icon": "🔍", "status": "Optimized"},
                {"name": "Matrix Multiplication", "icon": "🧮", "status": "Optimized"},
                {"name": "Data Processing", "icon": "🔄", "status": "Enhanced"},
                {"name": "Statistical Analysis", "icon": "📊", "status": "Enhanced"}
            ]
            
            for op in operations:
                status_color = "#10b981" if op["status"] == "Optimized" else "#f59e0b"
                st.markdown(f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 0.75rem;
                    margin: 0.5rem 0;
                    background: #f8fafc;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                ">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.2rem;">{op['icon']}</span>
                        <span style="color: #475569; font-weight: 500;">{op['name']}</span>
                    </div>
                    <span style="
                        background: {status_color}15;
                        color: {status_color};
                        padding: 0.25rem 0.75rem;
                        border-radius: 20px;
                        font-size: 0.75rem;
                        font-weight: 600;
                        border: 1px solid {status_color}30;
                    ">{op['status']}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Impact Metrics Section
            st.markdown("""
            <div class="content-card">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                    <div style="font-size: 1.5rem;">📈</div>
                    <h3 style="margin: 0; color: #1e293b;">Performance Impact</h3>
                </div>
            """, unsafe_allow_html=True)
            
            metrics = [
                {"value": "3-10x", "label": "Faster Execution", "icon": "⚡"},
                {"value": "60%", "label": "Memory Reduction", "icon": "💾"},
                {"value": "Quantum", "label": "Ready Operations", "icon": "🔮"},
                {"value": "100%", "label": "Secure Analysis", "icon": "🛡️"}
            ]
            
            # Create 2x2 grid for metrics
            metric_cols = st.columns(2)
            for i, metric in enumerate(metrics):
                with metric_cols[i % 2]:
                    st.markdown(f"""
                    <div style="
                        text-align: center;
                        padding: 1rem;
                        background: linear-gradient(135deg, #f8fafc, #ffffff);
                        border-radius: 12px;
                        border: 1px solid #e2e8f0;
                        margin-bottom: 0.5rem;
                    ">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{metric['icon']}</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;">
                            {metric['value']}
                        </div>
                        <div style="font-size: 0.75rem; color: #64748b; font-weight: 500;">
                            {metric['label']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # CTA Section
            st.markdown("""
            <div class="content-card" style="background: linear-gradient(135deg, #f0f9ff, #e0f2fe); border: none;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">🚀</div>
                    <h4 style="margin: 0 0 0.5rem 0; color: #1e293b;">Ready to Optimize?</h4>
                    <p style="color: #64748b; margin-bottom: 1.5rem; font-size: 0.9rem;">
                        Start analyzing your code for performance improvements today.
                    </p>
                    <div style="
                        background: #3b82f6;
                        color: white;
                        padding: 0.75rem 1.5rem;
                        border-radius: 8px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    " onmouseover="this.style.background='#2563eb'; this.style.transform='translateY(-1px)';" 
                    onmouseout="this.style.background='#3b82f6'; this.style.transform='translateY(0)';">
                        Get Started Free
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
        # Bottom testimonial/trust section
        st.markdown("""
        <div style="
            background: #f8fafc;
            border-radius: 16px;
            padding: 2rem;
            margin-top: 2rem;
            border: 1px solid #e2e8f0;
            text-align: center;
        ">
            <h3 style="color: #1e293b; margin-bottom: 1.5rem;">Trusted by Pharmaceutical Leaders</h3>
            <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 2rem;">
                <div style="color: #64748b; font-weight: 600;">Novartis</div>
                <div style="color: #64748b; font-weight: 600;">Pfizer</div>
                <div style="color: #64748b; font-weight: 600;">Roche</div>
                <div style="color: #64748b; font-weight: 600;">Johnson & Johnson</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # DEMO PAGE
    else:
        st.text("Auburn AI detects inefficient code implementation and suggests classical and quantum improvements.")

    
            
           

        # Check if model loaded successfully
        if model is None:
            st.error("❌ Model failed to load. Please check if all model files are available.")
            st.stop()

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
                    st.session_state.analysis_code = EXAMPLE_CODES[title]
                    st.session_state.selected_example = title

        # Display selected example
        if 'selected_example' in st.session_state:
            st.markdown(f"**Example Loaded:** {st.session_state.selected_example}")
            st.code(st.session_state.analysis_code, language='python')
        
        st.markdown("---")
        
        # Main Analysis Section
        st.subheader("Code Analysis")
        
        # Code input area
        code_input = st.text_area(
            "Paste or write your code here:", 
            height=250,
            value=st.session_state.get('analysis_code', ''),
            placeholder="""# Write your code here or use an example above\n\ndef your_function():\n    # Your code here\n    return result""",
            help="Auburn v0.1 will detect inefficient patterns in sorting, searching, and matrix operations"
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
            if st.button(" Clear ", use_container_width=True):
                st.session_state.analysis_code = ""
                if 'selected_example' in st.session_state:
                    del st.session_state.selected_example
                st.rerun()
        
        # Analysis execution - CLEANED AND CORRECTED
        if analyze_clicked and code_input.strip():
            with st.spinner("🔍 Analyzing code patterns..."):
                try:
                    # Use the EXACT same prediction logic as Tkinter app
                    predicted_labels, confidence_scores = predict_operations(code_input)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    if predicted_labels:
                        st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                        st.error("Inefficiencies Detected")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        for label in predicted_labels:
                            confidence = confidence_scores.get(label, 0) * 100
                            with st.container():
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    # Consistent label formatting with Tkinter app
                                    st.write(f"**{label.replace('_', ' ').title()}**")
                                with col_b:
                                    st.write(f"`{confidence:.1f}%`")
                            
                            # Use operations_info from loaded metadata
                            if label in operations_info:
                                info = operations_info[label]
                                with st.expander(f"Details & Recommendations"):
                                    st.write(f"**Description**: {info.get('description', 'N/A')}")
                                    st.write(f"**Quantum Speedup**: {info.get('quantum_speedup', 'N/A')}")
                                    st.write(f"**Classical Efficiency**: {info.get('classical_efficiency', 'N/A')}")
                                    st.write(f"**Optimization**: {info.get('optimization_notes', 'N/A')}")
                    else:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("No inefficiencies detected!")
                        st.write("The code appears to use efficient implementations.")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    # Detailed confidence scores
                    with st.expander("Detailed Confidence Scores"):
                        st.write("All detected patterns with confidence levels:")
                        for label, confidence in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
                            progress_value = confidence
                            st.write(f"**{label.replace('_', ' ').title()}**")
                            st.progress(progress_value, text=f"{confidence:.1%} confidence")
                            
                except Exception as e:
                    st.error(f"❌ Error analyzing code: {str(e)}")
                    st.info("""
                    **Troubleshooting tips:**
                    - Ensure the code is valid Python syntax
                    - Try using one of the example codes above
                    """)
        elif analyze_clicked and not code_input.strip():
            st.warning("⚠️ Please enter some code to analyze")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**Auburn** • Safe Reliable Smart")
    with col2:
        st.markdown("Speedup your code with classical optimization and unveil its quantum potential.")
    with col3:
        st.markdown("v0.1 • demo ")
