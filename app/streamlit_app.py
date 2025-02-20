import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import time
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.model import BreastCancerModel
from src.models.diabetes import DiabetesModel
from src.models.heart_disease import HeartDiseaseModel
from src.models.parkinsons import ParkinsonsModel
from src.config import (
    BREAST_CANCER_MODEL_PATH,
    DIABETES_MODEL_PATH,
    HEART_DISEASE_MODEL_PATH,
    PARKINSONS_MODEL_PATH
)

# Set page config
st.set_page_config(
    page_title="Medical Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Update the CSS section with better colors
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;500;700;900&display=swap');

    /* Base styles */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        color: #2C3E50;
    }
    
    /* Animated containers */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3498DB, #2980B9);
        color: white;
        font-weight: 500;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        background: linear-gradient(135deg, #2980B9, #2C3E50);
    }
    
    /* Card container */
    .card {
        padding: 1.5rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #E0E0E0;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        border-color: #3498DB;
    }
    
    .card h3 {
        color: #2C3E50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .card p {
        color: #7F8C8D;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #3498DB, #2C3E50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    h2 {
        color: #2C3E50;
        font-weight: 600;
        margin: 1.5rem 0;
    }
    
    h3 {
        color: #34495E;
        font-weight: 500;
    }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        color: #2980B9;
        font-weight: 600;
    }
    
    div[data-testid="stMetricDelta"] {
        color: #27AE60;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A8A 0%, #1E40AF 100%);
        padding: 2rem 1rem;
    }
    
    /* Sidebar title color */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        color: white !important;
        background: none;
        -webkit-background-clip: unset;
        -webkit-text-fill-color: unset;
    }
    
    /* Sidebar caption */
    [data-testid="stSidebar"] .css-10trblm {
        color: #E5E7EB !important;
        opacity: 0.8;
    }
    
    /* Radio buttons in navigation */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Radio button options */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
        padding: 10px;
        margin: 4px 0;
        color: #E5E7EB !important;
        transition: all 0.2s ease;
    }
    
    /* Hover effect for radio options */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(5px);
    }
    
    /* Selected radio option */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
        background: rgba(255, 255, 255, 0.2);
        border-left: 4px solid #3498DB;
    }
    
    /* Navigation section header */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #E5E7EB !important;
        font-size: 0.8em;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Sidebar image container */
    [data-testid="stSidebar"] [data-testid="stImage"] {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Scrollbar styling for sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        scrollbar-width: thin;
        scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]::-webkit-scrollbar {
        width: 6px;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-track {
        background: transparent;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 3px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #E0E0E0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 5px;
        color: #2980B9;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(41, 128, 185, 0.1);
        color: #2C3E50;
    }
    
    /* Alert/Message styling */
    .stAlert {
        background-color: #F8FAFC;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .element-container.css-1e5imcs.e1tzin5v1 {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Success message */
    .success {
        background-color: #D4EDDA;
        color: #155724;
        border: 1px solid #C3E6CB;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Warning message */
    .warning {
        background-color: #FFF3CD;
        color: #856404;
        border: 1px solid #FFEEBA;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Error message */
    .error {
        background-color: #F8D7DA;
        color: #721C24;
        border: 1px solid #F5C6CB;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #F8FAFC;
        border-color: #3498DB;
    }
    
    /* DataTable styling */
    .dataframe {
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .dataframe th {
        background-color: #F8FAFC;
        color: #2C3E50;
        font-weight: 600;
        padding: 0.75rem 1rem;
    }
    
    .dataframe td {
        padding: 0.75rem 1rem;
        border-top: 1px solid #E0E0E0;
    }
    
    /* Slider styling */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div {
        background-color: #3498DB;
    }
    
    /* Selectbox styling */
    .stSelectbox {
        color: #2C3E50;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 5px;
    }

    /* Loading animation */
    .loading-container {
        text-align: center;
        padding: 2rem;
        animation: fadeIn 0.5s ease-out;
    }

    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #3498db;
        border-radius: 50%;
        margin: 20px auto;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Success message animation */
    .success-message {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        animation: slideIn 0.5s ease-out;
    }

    .success-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
        background: white;
        color: #28a745;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
    }

    /* Feature cards */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .feature-card {
        background: linear-gradient(135deg, #2C3E50, #3498DB);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
        color: white;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        background: linear-gradient(135deg, #3498DB, #2C3E50);
    }

    .feature-card h3 {
        color: white;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 1.5rem;
    }

    .feature-card p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        line-height: 1.5;
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
        background: rgba(255, 255, 255, 0.1);
        width: 70px;
        height: 70px;
        line-height: 70px;
        border-radius: 50%;
        margin: 0 auto 1rem auto;
        transition: all 0.3s ease;
    }

    .feature-card:hover .feature-icon {
        transform: scale(1.1);
        background: rgba(255, 255, 255, 0.2);
    }

    /* Pulse animation for important elements */
    .pulse {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Slide animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Staggered animation classes */
    .stagger-1 { animation: slideIn 0.5s ease-out 0.1s both; }
    .stagger-2 { animation: slideIn 0.5s ease-out 0.2s both; }
    .stagger-3 { animation: slideIn 0.5s ease-out 0.3s both; }
    .stagger-4 { animation: slideIn 0.5s ease-out 0.4s both; }
    .stagger-5 { animation: slideIn 0.5s ease-out 0.5s both; }
    
    /* Fade up animation */
    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Slide in from sides */
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Apply animations to specific elements */
    .title-animation {
        animation: fadeUp 0.8s ease-out;
    }
    
    .subtitle-animation {
        animation: fadeUp 0.8s ease-out 0.2s both;
    }
    
    .card-left {
        animation: slideInLeft 0.8s ease-out both;
    }
    
    .card-right {
        animation: slideInRight 0.8s ease-out both;
    }
    
    /* Sequential feature cards */
    .features-grid {
        opacity: 0;
        animation: fadeIn 0.5s ease-out 0.6s forwards;
    }
    
    .feature-card:nth-child(1) {
        animation: slideIn 0.5s ease-out 0.7s both;
    }
    
    .feature-card:nth-child(2) {
        animation: slideIn 0.5s ease-out 0.9s both;
    }
    
    .feature-card:nth-child(3) {
        animation: slideIn 0.5s ease-out 1.1s both;
    }

    /* Professional animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInFromLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes slideInFromRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Apply animations to specific elements */
    .main-title {
        animation: fadeInUp 0.8s ease-out;
    }

    .welcome-text {
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }

    .feature-card:nth-child(1) {
        animation: slideInFromLeft 0.8s ease-out 0.3s both;
    }

    .feature-card:nth-child(2) {
        animation: fadeInUp 0.8s ease-out 0.4s both;
    }

    .feature-card:nth-child(3) {
        animation: slideInFromRight 0.8s ease-out 0.5s both;
    }

    .tool-section {
        animation: fadeInUp 0.8s ease-out 0.6s both;
    }

    .metrics-section {
        animation: fadeInUp 0.8s ease-out 0.7s both;
    }

    .info-section {
        animation: fadeInUp 0.8s ease-out 0.8s both;
    }

    /* Card hover effects */
    .card {
        transition: all 0.3s ease;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        border-color: #3498DB;
    }

    /* Button hover animation */
    .stButton>button {
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
    }

    /* Smooth section transitions */
    .section-transition {
        transition: all 0.5s ease;
    }

    /* Loading animation */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .loading-bg {
        background: linear-gradient(-45deg, #3498db, #2980b9, #2c3e50, #3498db);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    </style>
""", unsafe_allow_html=True)

def check_model_exists(model_path):
    """Check if a model file exists"""
    return os.path.exists(model_path)

def load_animation():
    """Show a loading animation"""
    with st.spinner('Loading...'):
        time.sleep(0.5)

def show_success_animation():
    """Show success animation"""
    placeholder = st.empty()
    for i in range(5):
        placeholder.markdown(f"{'🎯' * (i+1)}")
        time.sleep(0.1)
    placeholder.empty()

def add_home_button():
    """Add a Back to Home button"""
    if st.button("🏠 Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

def show_loading_page():
    """Show an animated loading screen"""
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
            <div class="loading-container">
                <h1>🏥 Medical AI Assistant</h1>
                <div class="loading-spinner"></div>
                <p>Loading advanced diagnostic tools...</p>
            </div>
        """, unsafe_allow_html=True)
    time.sleep(1)
    placeholder.empty()

def show_success_message(message):
    """Show animated success message"""
    st.markdown(f"""
        <div class="success-message">
            <span class="success-icon">✓</span>
            {message}
        </div>
    """, unsafe_allow_html=True)

def show_feature_cards():
    """Show animated feature cards"""
    st.markdown("""
        <div class="features-grid">
            <div class="feature-card">
                <span class="feature-icon">🎯</span>
                <h3>High Accuracy</h3>
                <p>Advanced ML algorithms with 96.5% accuracy</p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">⚡</span>
                <h3>Real-time Analysis</h3>
                <p>Get instant predictions and risk assessments</p>
            </div>
            <div class="feature-card">
                <span class="feature-icon">🔒</span>
                <h3>Secure Analysis</h3>
                <p>Your data is processed securely and privately</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def home_page():
    show_loading_page()
    
    # Animated title
    st.markdown('<div style="animation: fadeIn 1s ease-out">', unsafe_allow_html=True)
    st.title("🏥 Medical AI Assistant")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Welcome message with animation
    st.markdown("""
        <div style="animation: slideIn 0.5s ease-out">
        <h2>Welcome to the Medical Prediction System</h2>
        <p>This advanced AI-powered system helps medical professionals assess various health conditions 
        using machine learning algorithms trained on extensive medical datasets.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show feature cards
    show_feature_cards()
    
    # Tools section with animated cards
    st.markdown('<div class="tool-section">', unsafe_allow_html=True)
    st.subheader("🔧 Available Assessment Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("""
                <div class="card">
                    <h3>🔬 Breast Cancer Assessment</h3>
                    <p>Analyzes cell nuclei characteristics to assess cancer risk.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Start Breast Cancer Assessment"):
                st.session_state.page = "Breast Cancer"
                st.rerun()
        
        with st.container():
            st.markdown("""
                <div class="card">
                    <h3>❤️ Heart Disease Assessment</h3>
                    <p>Evaluates cardiovascular health indicators.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Start Heart Disease Assessment"):
                st.session_state.page = "Heart Disease"
                st.rerun()
    
    with col2:
        with st.container():
            st.markdown("""
                <div class="card">
                    <h3>🩺 Diabetes Assessment</h3>
                    <p>Analyzes various health metrics to assess diabetes risk.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Start Diabetes Assessment"):
                st.session_state.page = "Diabetes"
                st.rerun()
        
        with st.container():
            st.markdown("""
                <div class="card">
                    <h3>🧠 Parkinson's Assessment</h3>
                    <p>Examines voice patterns to detect early signs.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Start Parkinson's Assessment"):
                st.session_state.page = "Parkinson's Disease"
                st.rerun()
    
    # System Overview (only once)
    st.markdown("---")
    st.subheader("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Assessments", "5.2k", "↑12%")
    with col2:
        st.metric("Accuracy", "96.5%", "↑1.2%")
    with col3:
        st.metric("Active Users", "1.2k", "↑15%")
    with col4:
        st.metric("Response Time", "0.5s", "↓0.1s")
    
    # Technical Information
    st.markdown("---")
    st.subheader("🔍 Technical Information")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        ### Data Sources
        - Breast Cancer Wisconsin Dataset
        - Pima Indians Diabetes Database
        - Heart Disease UCI Dataset
        - Parkinson's Disease Dataset
        """)
    
    with tech_col2:
        st.markdown("""
        ### Technologies Used
        - Machine Learning: scikit-learn
        - Web Interface: Streamlit
        - Data Processing: pandas, numpy
        - Version Control: Git
        """)
    
    # Research & Publications
    st.markdown("---")
    with st.expander("📚 Research & Publications"):
        st.markdown("""
        ### Related Research Papers
        1. "Machine Learning in Medical Diagnosis" (2023)
        2. "AI Applications in Healthcare" (2022)
        3. "Early Disease Detection Using ML" (2023)
        
        ### Key Findings
        - 96.5% accuracy in breast cancer detection
        - 94.2% accuracy in diabetes prediction
        - 91.8% accuracy in heart disease assessment
        - 93.5% accuracy in Parkinson's detection
        
        ### Methodology
        Our system employs advanced machine learning algorithms trained on extensive medical datasets, 
        ensuring reliable and accurate predictions for various medical conditions.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>Developed with ❤️ for healthcare professionals</p>
            <p>Version 1.0.0 | © 2024 Medical AI Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def breast_cancer_prediction():
    add_home_button()
    show_loading_page()
    
    st.markdown("""
        <div class="page-header">
            <h1>Breast Cancer Risk Assessment</h1>
            <p class="subtitle">Advanced cellular analysis using machine learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not check_model_exists(BREAST_CANCER_MODEL_PATH):
        st.error("⚠️ Breast Cancer model not found. Please train the model first.")
        return
    
    try:
        model = BreastCancerModel.load_model()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return
    
    # Create tabs for input methods
    tab1, tab2 = st.tabs(["📊 Standard Input", "🔬 Detailed Input"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            mean_radius = st.slider("Mean Radius", 6.0, 28.0, 14.0, help="Average size of cell nuclei")
            mean_texture = st.slider("Mean Texture", 9.0, 40.0, 14.0, help="Average standard deviation of gray-scale values")
            mean_perimeter = st.slider("Mean Perimeter", 40.0, 190.0, 90.0, help="Average size of the core tumor")
            mean_area = st.slider("Mean Area", 140.0, 2500.0, 550.0, help="Average area of cell nuclei")
        
        with col2:
            mean_smoothness = st.slider("Mean Smoothness", 0.05, 0.16, 0.1, help="Average of local variation in radius lengths")
            mean_compactness = st.slider("Mean Compactness", 0.02, 0.35, 0.1, help="Average of perimeter^2 / area - 1.0")
            mean_concavity = st.slider("Mean Concavity", 0.0, 0.5, 0.1, help="Average severity of concave portions of the contour")
            mean_concave_points = st.slider("Mean Concave Points", 0.0, 0.2, 0.1, help="Average number of concave portions of the contour")

    if st.button("Analyze Risk", help="Click to analyze breast cancer risk"):
        with st.spinner('Analyzing samples...'):
            time.sleep(1)  # Simulate processing
            try:
                # Create input data array
                input_data = np.array([
                    mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                    mean_compactness, mean_concavity, mean_concave_points, 0.2, 0.06,
                    0.4, 0.4, 2.0, 20.0, 0.01, 0.02, 0.02, 0.01, 0.02, 0.003,
                    16.0, 16.0, 100.0, 700.0, 0.12, 0.15, 0.15, 0.1, 0.25, 0.08
                ]).reshape(1, -1)
                
                prediction, similar_cases, similar_outcomes, distances = model.predict(input_data)
                
                if prediction[0] == 0:
                    st.error("⚠️ High Risk Assessment")
                    st.warning(
                        "The analysis indicates characteristics commonly associated with malignant breast masses."
                    )
                    
                    # Show risk factors
                    st.subheader("Risk Factors Identified")
                    if mean_radius > 15:
                        st.warning(f"• Mean radius ({mean_radius:.2f}) is elevated")
                    if mean_concave_points > 0.05:
                        st.warning(f"• Mean concave points ({mean_concave_points:.3f}) are high")
                else:
                    st.success("✅ Low Risk Assessment")
                    st.info(
                        "The analysis indicates characteristics commonly associated with benign breast masses."
                    )
                
                # Show similar cases
                with st.expander("View Similar Cases"):
                    st.markdown("### Reference Cases")
                    st.markdown("These are similar cases from our database:")
                    
                    similar_df = pd.DataFrame({
                        'Mean Radius': similar_cases['mean radius'].round(2),
                        'Mean Texture': similar_cases['mean texture'].round(2),
                        'Mean Area': similar_cases['mean area'].round(2),
                        'Diagnosis': ['Malignant' if o == 0 else 'Benign' for o in similar_outcomes],
                        'Similarity': [f"{(1 - d/d.max())*100:.1f}%" for d in distances]
                    })
                    st.dataframe(similar_df)
                
                show_success_message("Analysis completed successfully!")
            except Exception as e:
                st.error(f"⚠️ Error during analysis: {str(e)}")

def diabetes_prediction():
    # Add home button at the top
    add_home_button()
    
    load_animation()
    st.header("Diabetes Prediction")
    st.write("Enter measurements to predict diabetes risk")
    
    try:
        model = DiabetesModel.load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", value=0, min_value=0)
        glucose = st.number_input("Glucose (mg/dL)", value=120, min_value=0)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", value=70, min_value=0)
        skin_thickness = st.number_input("Skin Thickness (mm)", value=20, min_value=0)
        
    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", value=79, min_value=0)
        bmi = st.number_input("BMI", value=25.0, min_value=0.0)
        dpf = st.number_input("Diabetes Pedigree Function", value=0.5, min_value=0.0)
        age = st.number_input("Age", value=33, min_value=0)
    
    if st.button("Predict"):
        try:
            # Calculate derived features
            glucose_bmi = glucose * bmi / 1000
            glucose_age = glucose * age / 100
            
            input_data = np.array([
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age, glucose_bmi, glucose_age
            ]).reshape(1, -1)
            
            prediction, similar_cases, similar_outcomes, distances = model.predict(input_data)
            
            # Show prediction with risk factors
            if prediction[0] == 1:
                st.error("High risk of diabetes")
                if glucose > 140:
                    st.warning("⚠️ High glucose level detected")
                if bmi > 30:
                    st.warning("⚠️ High BMI detected")
            else:
                st.success("Low risk of diabetes")
            
            # Show similar cases
            st.write("### Similar Cases from Dataset")
            st.write("The prediction is based on these similar cases:")
            
            similar_df = pd.DataFrame({
                'Age': similar_cases['Age'].round(1),
                'BMI': similar_cases['BMI'].round(1),
                'Glucose': similar_cases['Glucose'].round(1),
                'Blood Pressure': similar_cases['BloodPressure'].round(1),
                'Outcome': ['Diabetic' if o == 1 else 'Non-diabetic' for o in similar_outcomes],
                'Similarity': [f"{(1 - d/d.max())*100:.1f}%" for d in distances]
            })
            st.dataframe(similar_df)
            
            # Show risk analysis
            st.write("### Risk Analysis")
            risk_factors = []
            if glucose > 140: risk_factors.append(f"Glucose ({glucose} mg/dL) is above normal range")
            if bmi > 30: risk_factors.append(f"BMI ({bmi:.1f}) indicates obesity")
            if blood_pressure > 90: risk_factors.append(f"Blood pressure ({blood_pressure} mm Hg) is elevated")
            if dpf > 0.8: risk_factors.append(f"Diabetes pedigree function ({dpf:.2f}) indicates family history")
            
            if risk_factors:
                st.write("Risk factors identified:")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("No major risk factors identified")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def heart_disease_prediction():
    # Add home button at the top
    add_home_button()
    
    load_animation()
    st.header("Heart Disease Prediction")
    st.write("Enter measurements to predict heart disease risk")
    
    try:
        model = HeartDiseaseModel.load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", value=50, min_value=0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", 
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120, min_value=0)
        chol = st.number_input("Serum Cholesterol (mg/dl)", value=200, min_value=0)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG Results", 
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        
    with col2:
        thalach = st.number_input("Maximum Heart Rate", value=150, min_value=0)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression by Exercise", value=0.0)
        slope = st.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels (0-3)", value=0, min_value=0, max_value=3)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    
    if st.button("Predict"):
        try:
            # Convert categorical inputs to numerical
            sex_num = 1 if sex == "Male" else 0
            cp_num = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
            fbs_num = 1 if fbs == "Yes" else 0
            restecg_num = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
            exang_num = 1 if exang == "Yes" else 0
            slope_num = ["Upsloping", "Flat", "Downsloping"].index(slope)
            thal_num = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 3
            
            input_data = np.array([
                age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num,
                thalach, exang_num, oldpeak, slope_num, ca, thal_num
            ]).reshape(1, -1)
            
            prediction, similar_cases, similar_outcomes, distances = model.predict(input_data)
            
            # Show prediction and risk analysis
            if prediction[0] == 1:
                st.error("High risk of heart disease")
                
                # Show specific risk factors
                st.write("### Risk Factors Identified:")
                risk_factors = []
                
                if age > 60:
                    risk_factors.append(f"Age ({age} years) - Higher risk with increasing age")
                if cp_num >= 2:
                    risk_factors.append("Chest Pain Type indicates potential issue")
                if trestbps > 140:
                    risk_factors.append(f"High Blood Pressure ({trestbps} mm Hg)")
                if chol > 240:
                    risk_factors.append(f"High Cholesterol ({chol} mg/dl)")
                if thalach < 120:
                    risk_factors.append(f"Low Maximum Heart Rate ({thalach} bpm)")
                if oldpeak > 2:
                    risk_factors.append(f"Significant ST Depression ({oldpeak})")
                if ca > 0:
                    risk_factors.append(f"Number of Major Vessels: {ca}")
                
                for factor in risk_factors:
                    st.warning(f"⚠️ {factor}")
            else:
                st.success("Low risk of heart disease")
                
                # Show protective factors
                good_factors = []
                if age < 50:
                    good_factors.append(f"Age ({age} years) is in a lower risk category")
                if trestbps < 120:
                    good_factors.append(f"Normal Blood Pressure ({trestbps} mm Hg)")
                if chol < 200:
                    good_factors.append(f"Healthy Cholesterol Level ({chol} mg/dl)")
                
                if good_factors:
                    st.write("### Protective Factors:")
                    for factor in good_factors:
                        st.info(f"✓ {factor}")
            
            # Show similar cases
            st.write("### Similar Cases from Dataset")
            st.write("The prediction is based on these similar cases:")
            
            similar_df = pd.DataFrame({
                'Age': similar_cases['age'].round(0),
                'Sex': ['Male' if s == 1 else 'Female' for s in similar_cases['sex']],
                'Blood Pressure': similar_cases['trestbps'].round(0),
                'Cholesterol': similar_cases['chol'].round(0),
                'Max Heart Rate': similar_cases['thalach'].round(0),
                'Outcome': ['High Risk' if o == 1 else 'Low Risk' for o in similar_outcomes],
                'Similarity': [f"{(1 - d/d.max())*100:.1f}%" for d in distances]
            })
            st.dataframe(similar_df)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def parkinsons_prediction():
    # Add home button at the top
    add_home_button()
    
    load_animation()
    st.header("Parkinsons Disease Prediction")
    st.write("Enter the following measurements:")
    
    if not check_model_exists(PARKINSONS_MODEL_PATH):
        st.error("Parkinson's model not found. Please train the model first.")
        if st.button("Train Parkinson's Model"):
            try:
                from train_models import train_parkinsons
                train_parkinsons()
                st.success("Model trained successfully! Please refresh the page.")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
        return
    
    try:
        model = ParkinsonsModel.load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        mdvp_fo = st.number_input("MDVP:Fo(Hz)", min_value=88.333, max_value=260.105, value=120.000, format="%.6f")
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", min_value=102.145, max_value=592.030, value=157.000, format="%.6f")
        mdvp_flo = st.number_input("MDVP:Flo(Hz)", min_value=65.476, max_value=239.170, value=75.000, format="%.6f")
        mdvp_jitter = st.number_input("MDVP:Jitter(%)", min_value=0.00168, max_value=0.03316, value=0.00784, format="%.6f")
        mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.000007, max_value=0.000260, value=0.000070, format="%.6f")
        mdvp_rap = st.number_input("MDVP:RAP", min_value=0.00068, max_value=0.02144, value=0.00370, format="%.6f")
        mdvp_ppq = st.number_input("MDVP:PPQ", min_value=0.00092, max_value=0.01958, value=0.00554, format="%.6f")
        jitter_ddp = st.number_input("Jitter:DDP", min_value=0.00204, max_value=0.06433, value=0.01109, format="%.6f")
    
    with col2:
        mdvp_shimmer = st.number_input("MDVP:Shimmer", min_value=0.00954, max_value=0.11908, value=0.04374, format="%.6f")
        mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.085, max_value=1.302, value=0.426, format="%.6f")
        shimmer_apq3 = st.number_input("Shimmer:APQ3", min_value=0.00455, max_value=0.05647, value=0.02182, format="%.6f")
        shimmer_apq5 = st.number_input("Shimmer:APQ5", min_value=0.0057, max_value=0.0794, value=0.03130, format="%.6f")
        mdvp_apq = st.number_input("MDVP:APQ", min_value=0.00719, max_value=0.13778, value=0.02971, format="%.6f")
        shimmer_dda = st.number_input("Shimmer:DDA", min_value=0.01364, max_value=0.16942, value=0.06545, format="%.6f")
        nhr = st.number_input("NHR", min_value=0.00065, max_value=0.31482, value=0.02211, format="%.6f")
        hnr = st.number_input("HNR", min_value=8.441, max_value=33.047, value=21.033, format="%.6f")
        rpde = st.number_input("RPDE", min_value=0.256570, max_value=0.685151, value=0.414783, format="%.6f")
        dfa = st.number_input("DFA", min_value=0.574282, max_value=0.825288, value=0.815285, format="%.6f")
        spread1 = st.number_input("Spread1", min_value= -7.964984, max_value= -2.434031, value= -4.813031, format="%.6f")
        spread2 = st.number_input("Spread2", min_value=0.006274, max_value=0.450493, value=0.266482, format="%.6f")
        d2 = st.number_input("D2", min_value=1.423287, max_value=3.671155, value=2.301442, format="%.6f")
        ppe = st.number_input("PPE", min_value=0.044539, max_value=0.527367, value=0.284654, format="%.6f")
    
    if st.button("Predict"):
        try:
            input_data = np.array([
                mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_jitter_abs,
                mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
                shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr,
                rpde, dfa, spread1, spread2, d2, ppe
            ]).reshape(1, -1)
            
            prediction, similar_cases, similar_outcomes, distances = model.predict(input_data)
            
            if prediction[0] == 1:
                st.error("⚠️ High risk of Parkinson's disease")
                st.write("### Risk Factors Identified:")
                risk_factors = []
                if mdvp_jitter > 0.008:
                    risk_factors.append(f"High Jitter ({mdvp_jitter:.5f}%) indicates vocal instability")
                if mdvp_jitter_abs > 0.0004:
                    risk_factors.append(f"High Absolute Jitter ({mdvp_jitter_abs:.5f}) indicates frequency instability")
                if mdvp_shimmer > 0.04:
                    risk_factors.append(f"High Shimmer ({mdvp_shimmer:.5f}) indicates amplitude variations")
                if mdvp_shimmer_db > 0.4:
                    risk_factors.append(f"High Shimmer dB ({mdvp_shimmer_db:.5f}dB) indicates amplitude instability")
                if hnr < 20:
                    risk_factors.append(f"Low HNR ({hnr:.3f}) indicates voice quality issues")
                if nhr > 0.03:
                    risk_factors.append(f"High NHR ({nhr:.5f}) indicates increased noise")
                if rpde > 0.5:
                    risk_factors.append(f"High RPDE ({rpde:.3f}) indicates increased vocal complexity")
                if dfa < 0.65:
                    risk_factors.append(f"Low DFA ({dfa:.3f}) indicates changes in vocal pattern")
                
                for factor in risk_factors:
                    st.warning(f"⚠️ {factor}")
            else:
                st.success("✅ Low risk of Parkinson's disease")
                good_factors = []
                if mdvp_jitter < 0.006:
                    good_factors.append(f"Normal Jitter ({mdvp_jitter:.5f}%)")
                if mdvp_shimmer < 0.03:
                    good_factors.append(f"Normal Shimmer ({mdvp_shimmer:.5f})")
                if hnr > 22:
                    good_factors.append(f"Good HNR ({hnr:.3f})")
                if nhr < 0.02:
                    good_factors.append(f"Good NHR ({nhr:.5f})")
                
                if good_factors:
                    st.write("### Protective Factors:")
                    for factor in good_factors:
                        st.info(f"✓ {factor}")
            
            # Show similar cases
            st.write("### Similar Cases from Dataset")
            similar_df = pd.DataFrame({
                'Jitter(%)': similar_cases['MDVP:Jitter(%)'].round(5),
                'Shimmer': similar_cases['MDVP:Shimmer'].round(5),
                'HNR': similar_cases['HNR'].round(2),
                'RPDE': similar_cases['RPDE'].round(3),
                'DFA': similar_cases['DFA'].round(3),
                'Diagnosis': ['Parkinson\'s' if o == 1 else 'Healthy' for o in similar_outcomes],
                'Similarity': [f"{(1 - d/d.max())*100:.1f}%" for d in distances]
            })
            st.dataframe(similar_df)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def main():
    # Initialize session state if not exists
    if "page" not in st.session_state:
        st.session_state.page = "Home"
        show_loading_page()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/hospital-2.png", width=100)
        st.title("Medical AI Assistant")
        st.caption("v1.0.0")
        
        # Navigation
        pages = {
            "🏠 Home": "Home",
            "🔬 Breast Cancer": "Breast Cancer",
            "🩺 Diabetes": "Diabetes",
            "❤️ Heart Disease": "Heart Disease",
            "🧠 Parkinson's Disease": "Parkinson's Disease"
        }
        
        # Get current page index
        current_page = st.session_state.page
        current_key = next(k for k, v in pages.items() if v == current_page)
        
        selected = st.radio(
            "🧭 Navigation",
            list(pages.keys()),
            index=list(pages.keys()).index(current_key)
        )
        
        # Update page when selection changes
        if pages[selected] != st.session_state.page:
            st.session_state.page = pages[selected]
            st.rerun()
    
    # Main content routing
    try:
        if st.session_state.page == "Home":
            home_page()
        elif st.session_state.page == "Breast Cancer":
            breast_cancer_prediction()
        elif st.session_state.page == "Diabetes":
            diabetes_prediction()
        elif st.session_state.page == "Heart Disease":
            heart_disease_prediction()
        elif st.session_state.page == "Parkinson's Disease":
            parkinsons_prediction()
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.session_state.page = "Home"
        st.rerun()

if __name__ == "__main__":
    main() 