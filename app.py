import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Set page config
st.set_page_config(
    page_title="Pharma Code Analyzer",
    page_icon="🧪",
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
st.title("🧪 Pharmaceutical Code Efficiency Analyzer")
st.write("Detect inefficient patterns in pharmaceutical/biotech code")

with st.expander("📋 Example Code"):
    st.code("""# Inefficient bubble sort
compounds = load_compound_library()
for i in range(len(compounds)):
    for j in range(len(compounds)-1):
        if compounds[j].ic50 > compounds[j+1].ic50:
            compounds[j], compounds[j+1] = compounds[j+1], compounds[j]

# Inefficient linear search
for compound in compounds:
    if compound.smiles == target_smiles:
        target_compound = compound
        break""")

code_input = st.text_area("Paste your code here:", height=200, placeholder="Paste your pharmaceutical code here...")

if st.button("Analyze Code", type="primary"):
    if code_input.strip():
        with st.spinner("🔍 Analyzing code patterns..."):
            try:
                model = load_model()
                tokenizer, mlb, metadata = load_components()
                
                # Preprocess and predict
                processed_code = preprocess_code(code_input)
                sequence = tokenizer.texts_to_sequences([processed_code])
                padded_sequence = pad_sequences(sequence, maxlen=metadata['max_len'], padding='post')
                
                # Keras prediction (much simpler!)
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
                        st.write(f"**{label}** (Confidence: {confidence:.1f}%)")
                        
                        if label in metadata.get('fundamental_operations', {}):
                            info = metadata['fundamental_operations'][label]
                            with st.expander(f"Details for {label}"):
                                st.write(f"**Description**: {info.get('description', 'N/A')}")
                                st.write(f"**Quantum Speedup**: {info.get('quantum_speedup', 'N/A')}")
                                st.write(f"**Optimization**: {info.get('optimization_notes', 'N/A')}")
                else:
                    st.success("✅ No inefficiencies detected!")
                    st.info("The code appears to use efficient implementations.")
                    
                # Show raw confidence scores
                with st.expander("🔍 Detailed Confidence Scores"):
                    for label, confidence in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"{label}: {confidence:.3f}")
                        
            except Exception as e:
                st.error(f"❌ Error analyzing code: {str(e)}")
                st.info("Make sure all model files (model.h5, tokenizer.pkl, mlb.pkl, metadata.pkl) are in the repository.")
    else:
        st.warning("Please enter some code to analyze")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Detects inefficient sorting, matrix operations, and search patterns")
