import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Set page config
st.set_page_config(
    page_title="Auburn free demo",
    page_icon="",
    layout="wide"
)

# Load Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

# Load other components
@st.cache_resource
def load_components():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return tokenizer, mlb, metadata

    # Keep your existing preprocessing function and model training code exactly as before
def preprocess_code(code):
        """
        Enhanced preprocessing for pharmaceutical code patterns
        Focuses on preserving operation-specific patterns while normalizing noise
        """
        # Convert to lowercase first for consistent matching
        code = code.lower()
        
        # Step 1: Preserve critical operation patterns with context
        # Matrix operations with context preservation
        matrix_ops = [
            (r'np\.dot\s*\(', ' dot '),
            (r'tf\.matmul\s*\(', ' matmul '),
            (r'torch\.mm\s*\(', ' mm '),
            (r'torch\.matmul\s*\(', ' matmul '),
            (r'np\.matmul\s*\(', ' matmul '),
            (r'jnp\.dot\s*\(', ' dot '),
            (r'paddle\.matmul\s*\(', ' matmul '),
            (r'tf\.linalg\.matmul\s*\(', ' matmul '),
            (r'@\s*\w', ' at_operator '),
            (r'\.dot\s*\(', ' dot '),
            (r'tf\.tensordot\s*\(', ' tensordot '),
            (r'torch\.dot\s*\(', ' dot '),
            (r'np\.inner\s*\(', ' inner '),
            (r'tf\.linalg\.matvec\s*\(', ' matvec '),
            (r'np\.vdot\s*\(', ' vdot '),
            (r'linear_algebra\.dot_product\s*\(', ' dot_product ')
        ]
        for pattern, replacement in matrix_ops:
            code = re.sub(pattern, replacement, code)
        
        # Differential equation solvers
        diff_eq_ops = [
            (r'solve_ivp\s*\(', ' diff_eq_solver '),
            (r'odeint\s*\(', ' diff_eq_solver '),
            (r'solve_ode\s*\(', ' diff_eq_solver '),
            (r'pde_solver\s*\(', ' diff_eq_solver '),
            (r'solve_bvp\s*\(', ' diff_eq_solver '),
            (r'scipy\.integrate\s*\.\s*(solve_ivp|odeint)', ' diff_eq_solver ')
        ]
        for pattern, replacement in diff_eq_ops:
            code = re.sub(pattern, replacement, code)
        
        # Matrix inversion operations
        inversion_ops = [
            (r'np\.linalg\.inv\s*\(', ' matrix_inversion '),
            (r'tf\.linalg\.inv\s*\(', ' matrix_inversion '),
            (r'torch\.inverse\s*\(', ' matrix_inversion '),
            (r'scipy\.linalg\.inv\s*\(', ' matrix_inversion '),
            (r'scipy\.linalg\.pinv\s*\(', ' matrix_inversion '),
            (r'torch\.linalg\.pinv\s*\(', ' matrix_inversion ')
        ]
        for pattern, replacement in inversion_ops:
            code = re.sub(pattern, replacement, code)
        
        # Database operations - preserve full context
        db_ops = [
            (r'select\s+.*?\s+from', ' database_query '),
            (r'db\.query\s*\(', ' database_query '),
            (r'cursor\.execute\s*\(', ' database_query '),
            (r'session\.query\s*\(', ' database_query '),
            (r'pd\.read_sql\s*\(', ' database_query '),
            (r'db\.collection\.find\s*\(', ' database_query '),
            (r'db\.find\s*\(', ' database_query '),
            (r'fetch\s*\(', ' database_query '),
            (r'conn\.query\s*\(', ' database_query '),
            (r'orm\.execute\s*\(', ' database_query ')
        ]
        for pattern, replacement in db_ops:
            code = re.sub(pattern, replacement, code)
        
        # Sorting operations
        sort_ops = [
            (r'sorted\s*\(', ' sorting_op '),
            (r'\.sort\s*\(', ' sorting_op '),
            (r'order\s+by', ' sorting_op '),
            (r'arrange\s*\(', ' sorting_op '),
            (r'np\.sort\s*\(', ' sorting_op '),
            (r'torch\.sort\s*\(', ' sorting_op '),
            (r'tf\.sort\s*\(', ' sorting_op '),
            (r'df\.sort_values\s*\(', ' sorting_op '),
            (r'df\.orderBy\s*\(', ' sorting_op ')
        ]
        for pattern, replacement in sort_ops:
            code = re.sub(pattern, replacement, code)
        
        # Addition operations - be more specific to avoid false positives
        addition_ops = [
            (r'\+\s*\w', ' addition_op '),
            (r'\+=', ' addition_op '),
            (r'np\.sum\s*\(', ' addition_op '),
            (r'tf\.add\s*\(', ' addition_op '),
            (r'torch\.sum\s*\(', ' addition_op '),
            (r'pandas\.Series\.add\s*\(', ' addition_op '),
            (r'jnp\.add\s*\(', ' addition_op '),
            (r'tf\.reduce_sum\s*\(', ' addition_op ')
        ]
        for pattern, replacement in addition_ops:
            code = re.sub(pattern, replacement, code)
        
        # Step 2: Preserve structural elements
        code = re.sub(r'([=<>(),.!;{}[\]])', r' \1 ', code)
        
        # Step 3: Handle literals and variables
        # Preserve string literals
        code = re.sub(r'("[^"]*"|\'[^\']*\')', ' _str_literal_ ', code)
        
        # Replace numbers (including scientific notation and decimals)
        code = re.sub(r'\b\d+\.?\d*([eE][-+]?\d+)?\b', ' _number_ ', code)
        
        # Replace variable names but preserve common pharmaceutical terms
        pharma_terms = [
            'concentration', 'dosage', 'compound', 'patient', 'clinical', 'trial',
            'drug', 'molecular', 'protein', 'gene', 'toxicity', 'efficacy',
            'binding', 'affinity', 'receptor', 'ligand', 'pharmacokinetic',
            'pharmacodynamic', 'ic50', 'ld50', 'biomarker', 'expression',
            'pathway', 'metabolic', 'enzyme', 'kinetic', 'cell_growth',
            'dose', 'response', 'treatment', 'control', 'placebo', 'therapy'
        ]
        
        # Replace generic variable names
        for term in pharma_terms:
            code = re.sub(r'\b' + term + r'\b', f' _pharma_{term}_ ', code)
        
        # Replace remaining variable names
        code = re.sub(r'\b[a-z_][a-z0-9_]{2,}\b', ' _variable_ ', code)
        
        # Step 4: Clean up and normalize
        # Collapse multiple spaces and trim
        code = re.sub(r'\s+', ' ', code).strip()
    
        return code


# UI
st.title("Auburn free demo")
st.write("Detect inefficiencies in pharma/biotech codebases")

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



# Example Gallery Section - ABOVE the analysis
st.subheader("📚 Example Code Gallery")
st.write("Click on examples to load them into the analyzer below:")

# Create columns for the example gallery
cols = st.columns(2)

# Distribute examples across columns
example_titles = list(EXAMPLE_CODES.keys())
for i, title in enumerate(example_titles):
    with cols[i % 2]:  # Alternate between columns
        if st.button(title, use_container_width=True, key=f"btn_{title}"):
            st.session_state.example_code = EXAMPLE_CODES[title]
            st.session_state.selected_example = title

# Display selected example code
if 'example_code' in st.session_state:
    st.subheader(f"📋 Example: {st.session_state.get('selected_example', 'Selected Code')}")
    st.code(st.session_state.example_code, language='python')
    
    # Add a button to use this code for analysis
    if st.button("🔍 Analyze This Example", type="primary"):
        st.session_state.analysis_code = st.session_state.example_code
        st.rerun()

# Analysis Section - BELOW the examples
st.markdown("---")
st.subheader("🔍 Code Analysis")

# Initialize session state for analysis code
if 'analysis_code' not in st.session_state:
    st.session_state.analysis_code = ""

# Text area for code input - prefill with selected example
code_input = st.text_area(
    "Paste or modify your code here:", 
    height=200,
    value=st.session_state.analysis_code,
    placeholder="Paste your code here or use an example above..."
)

# Analysis button
if st.button("Analyze Code", type="primary", use_container_width=True):
    if code_input.strip():
        with st.spinner("🔍 Analyzing code patterns..."):
            try:
                model = load_model()
                tokenizer, mlb, metadata = load_components()
                
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
                    st.error("🚨 Inefficiencies Detected:")
                    for label in predicted_labels[0]:
                        confidence = confidence_scores.get(label, 0) * 100
                        st.write(f"**{label.replace('_', ' ').title()}** (Confidence: {confidence:.1f}%)")
                        
                        if label in metadata.get('fundamental_operations', {}):
                            info = metadata['fundamental_operations'][label]
                            with st.expander(f"Details for {label}"):
                                st.write(f"**Description**: {info.get('description', 'N/A')}")
                                st.write(f"**Quantum Speedup**: {info.get('quantum_speedup', 'N/A')}")
                                st.write(f"**Optimization**: {info.get('optimization_notes', 'N/A')}")
                else:
                    st.success("✅ No inefficiencies detected!")
                    st.info("The code appears to use efficient implementations.")
                    
            except Exception as e:
                st.error(f"❌ Error analyzing code: {str(e)}")
    else:
        st.warning("Please enter some code to analyze")

# Clear button
if st.button("Clear All", use_container_width=True):
    st.session_state.analysis_code = ""
    if 'example_code' in st.session_state:
        del st.session_state.example_code
    if 'selected_example' in st.session_state:
        del st.session_state.selected_example
    st.rerun()

# Footer

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Detects inefficient sorting, matrix multiplication, and database search")
