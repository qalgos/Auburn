import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os
from PIL import Image
import base64
from fpdf import FPDF
from datetime import datetime
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
def create_analysis_pdf(code_snippet, predicted_labels, confidence_scores, operations_info):
    """Generate a professional PDF report"""
    
    pdf = FPDF()
    pdf.add_page()
    
    # Set up fonts
    pdf.set_font("Arial", size=12)
    
    # Header with gradient-like effect (using colors)
    pdf.set_fill_color(59, 130, 246)  # Blue
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, "Auburn AI - Code Analysis Report", ln=True, align='C', fill=True)
    pdf.ln(5)
    
    # Report metadata
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Analysis ID: {hash(code_snippet) % 10000:04d}", ln=True)
    pdf.ln(5)
    
    # Executive Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(0, 0, 0)
    
    if predicted_labels:
        summary_text = f"Analysis detected {len(predicted_labels)} potential inefficiencies in your code."
        pdf.multi_cell(0, 6, summary_text)
    else:
        pdf.multi_cell(0, 6, "No significant inefficiencies detected. Code appears well-optimized.")
    pdf.ln(5)
    
    # Code Snippet Section
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 8, "Analyzed Code", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.set_text_color(100, 100, 100)
    
    # Code with background
    pdf.set_fill_color(248, 250, 252)
    pdf.cell(0, 6, "", ln=True, fill=True)
    
    # Split code into lines and add to PDF
    code_lines = code_snippet.split('\n')
    for line in code_lines[:20]:  # Limit to first 20 lines
        pdf.cell(0, 4, line, ln=True)
    
    if len(code_lines) > 20:
        pdf.cell(0, 4, "... (code truncated for report)", ln=True)
    
    pdf.cell(0, 6, "", ln=True, fill=True)
    pdf.ln(5)
    
    # Detected Issues Section
    if predicted_labels:
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(239, 68, 68)  # Red for issues
        pdf.cell(0, 10, "Detected Inefficiencies", ln=True)
        
        for i, label in enumerate(predicted_labels, 1):
            confidence = confidence_scores.get(label, 0) * 100
            
            # Issue header
            pdf.set_font("Arial", 'B', 11)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, f"{i}. {label.replace('_', ' ').title()}", ln=True)
            
            # Confidence level
            pdf.set_font("Arial", 'I', 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, f"Confidence: {confidence:.1f}%", ln=True)
            
            # Detailed analysis for each operation
            if label in operations_info:
                info = operations_info[label]
                
                # Description
                pdf.set_font("Arial", 'B', 9)
                pdf.set_text_color(30, 64, 175)  # Dark blue
                pdf.cell(0, 6, "Description:", ln=True)
                pdf.set_font("Arial", size=9)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 5, info.get('description', 'N/A'))
                
                # Quantum Speedup
                pdf.set_font("Arial", 'B', 9)
                pdf.set_text_color(124, 58, 237)  # Purple
                pdf.cell(0, 6, "Quantum Speedup:", ln=True)
                pdf.set_font("Arial", size=9)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 5, info.get('quantum_speedup', 'N/A'))
                
                # Classical Efficiency
                pdf.set_font("Arial", 'B', 9)
                pdf.set_text_color(21, 128, 61)  # Green
                pdf.cell(0, 6, "Classical Efficiency:", ln=True)
                pdf.set_font("Arial", size=9)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 5, info.get('classical_efficiency', 'N/A'))
                
                # Optimization
                pdf.set_font("Arial", 'B', 9)
                pdf.set_text_color(180, 83, 9)  # Orange
                pdf.cell(0, 6, "Optimization Recommendations:", ln=True)
                pdf.set_font("Arial", size=9)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 5, info.get('optimization_notes', 'N/A'))
            
            pdf.ln(3)
    
    else:
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(34, 197, 94)  # Green for success
        pdf.cell(0, 10, "✓ No Inefficiencies Detected", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, "Your code appears to be well-optimized. No significant performance issues were found.")
    
    # Recommendations Section
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 10, "Overall Recommendations", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(0, 0, 0)
    
    recommendations = [
        "Implement suggested classical optimizations for immediate performance gains",
        "Consider quantum-ready algorithms for future scalability",
        "Regularly profile and optimize computational hotspots",
        "Use appropriate data structures for your specific use case"
    ]
    
    for rec in recommendations:
        pdf.cell(5)  # Indent
        pdf.cell(0, 6, f"• {rec}", ln=True)
    
    # Footer
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Generated by Auburn AI - Advanced Code Optimization for Pharmaceutical Research", ln=True, align='C')
    pdf.cell(0, 5, "Confidential Report - For authorized use only", ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin1')

def get_download_link(pdf_data, filename):
    """Generate a download link for the PDF"""
    b64 = base64.b64encode(pdf_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download PDF Report</a>'

# Integration into your existing analysis section:
def render_analysis_with_pdf():
    """Your existing analysis function enhanced with PDF download"""
    
    # ... your existing analysis code ...
    
    if analyze_clicked and code_input.strip():
        with st.spinner("🔍 Analyzing code patterns..."):
            try:
                predicted_labels, confidence_scores = predict_operations(code_input)
                
                # Display results (your existing code)
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
                                st.write(f"**{label.replace('_', ' ').title()}**")
                            with col_b:
                                st.write(f"`{confidence:.1f}%`")
                        
                        if label in operations_info:
                            # Your existing detailed analysis display...
                            pass
                
                # ADD PDF GENERATION BUTTON
                st.markdown("---")
                st.subheader("📊 Generate Report")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if st.button("📄 Generate PDF Report", use_container_width=True, type="secondary"):
                        with st.spinner("Generating professional report..."):
                            try:
                                pdf_data = create_analysis_pdf(
                                    code_input, 
                                    predicted_labels, 
                                    confidence_scores, 
                                    operations_info
                                )
                                
                                # Create download link
                                filename = f"auburn_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                                download_link = get_download_link(pdf_data, filename)
                                
                                st.markdown(download_link, unsafe_allow_html=True)
                                st.success("✅ Report generated successfully!")
                                
                            except Exception as e:
                                st.error(f"❌ Failed to generate PDF: {str(e)}")
                
                with col2:
                    st.info("""
                    **Professional Report Includes:**
                    • Executive summary
                    • Code analysis details  
                    • Confidence scores
                    • Optimization recommendations
                    • Quantum computing insights
                    """)
                
            except Exception as e:
                st.error(f"❌ Error analyzing code: {str(e)}")

# Alternative: Add download button right after analysis results
def add_pdf_download_section(code_input, predicted_labels, confidence_scores, operations_info):
    """Add PDF download section to your analysis results"""
    
    st.markdown("---")
    
    # PDF Generation Section
    st.markdown("""
    <div class="content-card">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem;">📊</div>
            <h3 style="margin: 0; color: #1e293b;">Export Analysis Report</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔄 Generate Comprehensive PDF Report", use_container_width=True, type="primary"):
            with st.spinner("🔄 Creating professional report..."):
                try:
                    pdf_data = create_analysis_pdf(
                        code_input, 
                        predicted_labels, 
                        confidence_scores, 
                        operations_info
                    )
                    
                    filename = f"Auburn_AI_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    download_link = get_download_link(pdf_data, filename)
                    
                    st.markdown("""
                    <div style="
                        background: #f0fdf4;
                        border: 1px solid #bbf7d0;
                        border-radius: 8px;
                        padding: 1rem;
                        margin-top: 1rem;
                    ">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span style="color: #16a34a;">✅</span>
                            <strong style="color: #166534;">Report Generated Successfully!</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(download_link, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Failed to generate PDF: {str(e)}")
    
    with col2:
        st.markdown("""
        **Report Features:**
        - Professional formatting
        - Detailed analysis breakdown  
        - Confidence metrics
        - Optimization strategies
        - Quantum computing potential
        - Executive summary
        - Code snippets
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Usage in your main analysis function:
# After displaying analysis results, call:
# add_pdf_download_section(code_input, predicted_labels, confidence_scores, operations_info)
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
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Overview")
            st.markdown("""
            <div class="feature-card">
            Auburn is an advanced AI-powered tool designed specifically for 
            the pharmaceutical and biotechnology industries. It automatically detects inefficient 
            code patterns and suggests improvements.
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Key Features")
            
            features = [
                (" Domain-Specific AI-Powered Analysis ", "Deep learning models optimized for pharma/biotech computational workflows, trained to recognise inefficiencies."),
                (" Private and Secure ", "Scan your codebase without worrying about leaks. Optimize your codebase fully in-house, without your proprietary code ever leaving your company.  "),
                (" Detailed Reporting ", "Comprehensive analysis with improvement suggestions. Learn if your business can benefit from quantum computers!")
            ]
            
            for feature, description in features:
                with st.expander(f"{feature}"):
                    st.write(description)
        
        with col2:
            st.subheader("Supported Operations")
            st.markdown("""
            - **Inefficient Sorting**
            - **Inefficient Search**  
            - **Inefficient Matrix Multiplication**
    
            """)
            
            st.subheader("Impact")
            st.markdown("""
            <div class="success-box">
            • Reduces computational time <br>
            • Optimizes memory usage <br>
            • Detects operations with quantum speedups <br>
            </div>
            """, unsafe_allow_html=True)

    #    # DEMO PAGE
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
                           # In your analysis results section, replace the details display with:
if label in operations_info:
    info = operations_info[label]
    with st.expander(f"🔍 Detailed Analysis: {label.replace('_', ' ').title()}"):
        
        # Description with blue background
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #eff6ff, #dbeafe);
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            border-left: 4px solid #3b82f6;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <div style="font-size: 1.2rem;">📝</div>
                <h4 style="margin: 0; color: #1e40af;">Description</h4>
            </div>
            <p style="margin: 0; color: #374151; line-height: 1.5;">
                {info.get('description', 'N/A')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quantum Speedup with purple background
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #faf5ff, #f3e8ff);
            border: 1px solid #e9d5ff;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            border-left: 4px solid #8b5cf6;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <div style="font-size: 1.2rem;">⚛️</div>
                <h4 style="margin: 0; color: #7c3aed;">Quantum Speedup</h4>
            </div>
            <p style="margin: 0; color: #374151; line-height: 1.5;">
                {info.get('quantum_speedup', 'N/A')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Classical Efficiency with green background
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f0fdf4, #dcfce7);
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            border-left: 4px solid #22c55e;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <div style="font-size: 1.2rem;">⚡</div>
                <h4 style="margin: 0; color: #15803d;">Classical Efficiency</h4>
            </div>
            <p style="margin: 0; color: #374151; line-height: 1.5;">
                {info.get('classical_efficiency', 'N/A')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Optimization with orange background
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #fffbeb, #fef3c7);
            border: 1px solid #fde68a;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            border-left: 4px solid #f59e0b;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <div style="font-size: 1.2rem;">🎯</div>
                <h4 style="margin: 0; color: #b45309;">Optimization</h4>
            </div>
            <p style="margin: 0; color: #374151; line-height: 1.5;">
                {info.get('optimization_notes', 'N/A')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Impact Summary with teal background
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f0fdfa, #ccfbf1);
            border: 1px solid #99f6e4;
            border-radius: 12px;
            padding: 1.25rem;
            border-left: 4px solid #14b8a6;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <div style="font-size: 1.2rem;">💡</div>
                <h4 style="margin: 0; color: #0f766e;">Impact Summary</h4>
            </div>
            <p style="margin: 0; color: #374151; line-height: 1.5;">
                Implementing the recommended classical improvements can significantly reduce computational time. 
                Depending on data size, time saved can range from <strong>seconds to hours or even days</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
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
