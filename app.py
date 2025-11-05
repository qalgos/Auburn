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

def authenticate():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔒 Auburn AI- Authentication required")
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            with st.form("auth_form"):
                password = st.text_input("Enter access password:", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    # Replace with your actual password
                    if password == "AuburninYC2026":
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Incorrect password")
            st.stop()
    
    return True

# Check authentication before running app
if authenticate():
    # Set page config - MUST be the first Streamlit command
    # Set page config
    st.set_page_config(
        page_title="Auburn free demo",
        page_icon="",
        layout="wide"
    )

   #
        
            # Custom CSS for modern styling
        def apply_custom_css():
            st.markdown("""
            <style>
            /* Main styling */
            .main-header {
                font-size: 3rem;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }
            
            .sub-header {
                font-size: 1.2rem;
                color: var(--text-color);
                margin-bottom: 2rem;
                opacity: 0.8;
            }
            
            /* Card styling */
            .custom-card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid var(--border-color);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .custom-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
            }
            
            /* Button styling */
            .stButton button {
                border-radius: 8px;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .stButton button:hover {
                transform: translateY(-1px);
            }
            
            /* Example gallery buttons */
            .example-btn {
                width: 100%;
                margin: 0.3rem 0;
                border-radius: 8px !important;
            }
            
            /* Code block styling */
            .stCodeBlock {
                border-radius: 8px;
                border: 1px solid var(--border-color);
            }
            
            /* Success/Error/Warning styling */
            .stAlert {
                border-radius: 8px;
            }
            
            /* Progress bar */
            .stProgress > div > div {
                background: linear-gradient(90deg, #667eea, #764ba2);
            }
            
            /* Hide default Streamlit elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """, unsafe_allow_html=True)
        
        def authenticate():
            if 'authenticated' not in st.session_state:
                st.session_state.authenticated = False
            
            if not st.session_state.authenticated:
                # Centered authentication layout
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    st.markdown('<h2 style="text-align: center; margin-bottom: 2rem;">🔒 Secure Access</h2>', unsafe_allow_html=True)
                    
                    with st.form("auth_form"):
                        password = st.text_input("**Enter Access Password**", type="password", 
                                               placeholder="Enter your password...")
                        submit = st.form_submit_button("🚀 Login", use_container_width=True)
                        
                        if submit:
                            # Replace with your actual password
                            if password == "my password":
                                st.session_state.authenticated = True
                                st.rerun()
                            else:
                                st.error("❌ Incorrect password. Please try again.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.info("💡 *Contact administrator for access credentials*")
                    st.stop()
            
            return True
        
        # Navigation function
        def navigation():
            with st.sidebar:
                st.markdown("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h1 style="font-size: 1.8rem; margin-bottom: 0.5rem;">🧬</h1>
                    <h2 style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0;">Code Efficiency</h2>
                    <p style="opacity: 0.7; font-size: 0.9rem;">Pharma/Biotech Analyzer</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Navigation menu
                page = st.radio(
                    "Navigate to:",
                    ["🚀 Free Demo", "📊 Analysis", "ℹ️ About"],
                    key="navigation"
                )
                
                st.markdown("---")
                
                # Theme selector
                st.subheader("🎨 Theme")
                theme = st.selectbox(
                    "Select theme:",
                    ["Light 🌞", "Dark 🌙", "Auto 🤖"],
                    key="theme_selector"
                )
                
                st.markdown("---")
                
                # Footer in sidebar
                st.markdown("""
                <div style="text-align: center; margin-top: 2rem; opacity: 0.7;">
                    <small>Built with Streamlit • v2.0</small>
                </div>
                """, unsafe_allow_html=True)
            
            return page
        
        # About page
        def render_about():
            st.markdown('<div class="main-header">About the Platform</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Advanced Code Efficiency Analysis for Pharmaceutical and Biotech Applications</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div class="custom-card">
                    <h3>🎯 Our Mission</h3>
                    <p>We provide cutting-edge code analysis specifically designed for pharmaceutical and biotech applications, 
                    helping researchers and developers optimize computational workflows and accelerate drug discovery.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="custom-card">
                    <h3>🔬 Key Features</h3>
                    <ul>
                    <li><strong>AI-Powered Analysis:</strong> Detect inefficient algorithms and suggest optimizations</li>
                    <li><strong>Domain-Specific Patterns:</strong> Specialized in pharma/biotech computational patterns</li>
                    <li><strong>Performance Metrics:</strong> Quantify potential speedup and efficiency gains</li>
                    <li><strong>Best Practices:</strong> Industry-standard recommendations for scientific computing</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="custom-card">
                    <h3>📈 Technology Stack</h3>
                    <p><strong>Machine Learning:</strong> TensorFlow/Keras</p>
                    <p><strong>Frontend:</strong> Streamlit</p>
                    <p><strong>Analysis:</strong> Custom NLP pipelines</p>
                    <p><strong>Deployment:</strong> Cloud-native architecture</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="custom-card">
                    <h3>👥 Team</h3>
                    <p>Developed by experts in computational biology, software engineering, and machine learning 
                    with decades of combined experience in pharmaceutical research and development.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Free Demo page
        def render_demo():
            st.markdown('<div class="main-header">Free Demo</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Experience the power of our code analysis platform</div>', unsafe_allow_html=True)
            
            # Demo information
            st.markdown("""
            <div class="custom-card">
                <h3>🎪 Welcome to the Demo</h3>
                <p>This interactive demo allows you to analyze code snippets for common inefficiencies found in 
                pharmaceutical and biotech applications. Try the examples below or paste your own code!</p>
            </div>
            """, unsafe_allow_html=True)
            
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
            
            # Example Gallery
            st.subheader("📚 Example Code Gallery")
            st.write("Click on examples to load them into the analyzer:")
            
            # Create columns for examples
            cols = st.columns(3)
            for i, (title, code) in enumerate(EXAMPLE_CODES.items()):
                with cols[i % 3]:
                    if st.button(title, use_container_width=True, key=f"demo_btn_{title}"):
                        st.session_state.demo_code = code
                        st.session_state.selected_demo = title
            
            # Display selected demo code
            if 'demo_code' in st.session_state:
                st.subheader(f"📋 {st.session_state.get('selected_demo', 'Selected Code')}")
                st.code(st.session_state.demo_code, language='python')
                
                if st.button("🔍 Analyze This Code", type="primary", use_container_width=True):
                    st.session_state.analysis_code = st.session_state.demo_code
                    st.session_state.page = "📊 Analysis"
                    st.rerun()
        
        # Analysis page (your existing analysis functionality)
        def render_analysis():
            st.markdown('<div class="main-header">Code Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Detect inefficiencies in pharma/biotech codebases</div>', unsafe_allow_html=True)
            
            # Load model and components (cached)
            @st.cache_resource
            def load_model():
                # Replace with your actual model loading
                 model = tf.keras.models.load_model("model.h5")
                 return model
          
            
            @st.cache_resource
            def load_components():
                # Replace with your actual component loading
                 with open('tokenizer.pkl', 'rb') as f:
                     tokenizer = pickle.load(f)
                 with open('mlb.pkl', 'rb') as f:
                     mlb = pickle.load(f)
                 with open('metadata.pkl', 'rb') as f:
                     metadata = pickle.load(f)
                 return tokenizer, mlb, metadata
            
            
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
            
            # Example codes for the analysis page
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
            }
            
            # Example Gallery Section
            st.markdown("""
            <div class="custom-card">
                <h3>🚀 Quick Start Examples</h3>
                <p>Select from common pharmaceutical code patterns to analyze:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for examples
            cols = st.columns(2)
            example_titles = list(EXAMPLE_CODES.keys())
            for i, title in enumerate(example_titles):
                with cols[i % 2]:
                    if st.button(title, use_container_width=True, key=f"btn_{title}"):
                        st.session_state.example_code = EXAMPLE_CODES[title]
                        st.session_state.selected_example = title
            
            # Display selected example
            if 'example_code' in st.session_state:
                st.subheader(f"📋 Example: {st.session_state.get('selected_example', 'Selected Code')}")
                st.code(st.session_state.example_code, language='python')
                
                if st.button("🔍 Analyze This Example", type="primary", use_container_width=True):
                    st.session_state.analysis_code = st.session_state.example_code
        
            # Analysis Section
            st.markdown("---")
            st.subheader("🔍 Code Input")
            
            # Initialize session state
            if 'analysis_code' not in st.session_state:
                st.session_state.analysis_code = ""
            
            # Code input area
            code_input = st.text_area(
                "**Paste your Python code here:**",
                height=250,
                value=st.session_state.analysis_code,
                placeholder="Paste your pharmaceutical/biotech code here...\n\n# Example:\ndef analyze_compound(compound):\n    # Your code here\n    return result",
                help="Enter Python code to analyze for computational inefficiencies"
            )
            
            # Analysis controls
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                analyze_btn = st.button("🚀 Analyze Code", type="primary", use_container_width=True)
            
            if analyze_btn:
                if code_input.strip():
                    with st.spinner("🔬 Analyzing code patterns..."):
                        # Simulate analysis (replace with your actual analysis)
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Display results
                        st.subheader("📊 Analysis Results")
                        
                        # Mock results - replace with your actual model predictions
                        st.error("""
                        **🚨 Inefficiencies Detected:**
                        
                        • **Inefficient Sorting Algorithm**: Bubble sort detected - consider using built-in sorted() or numpy.argsort()
                        • **Linear Search Pattern**: O(n) complexity - consider using dictionary lookups or binary search
                        • **Manual Matrix Operations**: Nested loops detected - use NumPy vectorized operations
                        """)
                        
                        st.info("""
                        **💡 Optimization Suggestions:**
                        
                        • Replace bubble sort with optimized sorting algorithms (O(n log n))
                        • Use hash tables for patient record lookups (O(1) average case)
                        • Leverage NumPy for matrix operations (significant speedup)
                        • Consider parallel processing for large datasets
                        """)
                        
                        # Confidence scores (mock)
                        with st.expander("🔍 Detailed Analysis Metrics"):
                            st.write("""
                            | Pattern | Confidence | Severity |
                            |---------|------------|----------|
                            | Inefficient Sorting | 92% | High |
                            | Linear Search | 87% | Medium |
                            | Manual Matrix Ops | 95% | High |
                            """)
                else:
                    st.warning("⚠️ Please enter some code to analyze")
        
            # Clear button
            if st.button("🗑️ Clear Analysis", use_container_width=True):
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
            page = navigation()
            
            # Route to appropriate page
            if page == "ℹ️ About":
                render_about()
            elif page == "🚀 Free Demo":
                render_demo()
            elif page == "📊 Analysis":
                render_analysis()
        
        if __name__ == "__main__":
            main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
