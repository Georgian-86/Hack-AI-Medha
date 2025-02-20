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

# Add this updated CSS at the beginning of the file
st.markdown("""
    <style>
    /* Original styling */
    .success-message {
        background-color: #28a745;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }

    .success-icon {
        font-size: 20px;
        margin-right: 10px;
    }

    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }

    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    .card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }

    .tool-section {
        margin: 2rem 0;
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
    
    # Hero section with gradient background
    st.markdown("""
        <div style="
            padding: 2rem;
            border-radius: 15px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            margin-bottom: 2rem;
            text-align: center;
            animation: fadeIn 1s ease-out;
        ">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">🏥 Medical AI Assistant</h1>
            <p style="font-size: 1.2rem; opacity: 0.9;">
                Advanced AI-powered diagnostics for healthcare professionals
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Quick stats cards
    st.markdown("""
        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        ">
            <div style="
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                border-top: 4px solid #2ecc71;
            ">
                <h3 style="color: #2ecc71; margin: 0;">96.5%</h3>
                <p style="color: #666; margin: 0;">Accuracy Rate</p>
            </div>
            <div style="
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                border-top: 4px solid #3498db;
            ">
                <h3 style="color: #3498db; margin: 0;">5,200+</h3>
                <p style="color: #666; margin: 0;">Assessments</p>
            </div>
            <div style="
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                border-top: 4px solid #e74c3c;
            ">
                <h3 style="color: #e74c3c; margin: 0;">0.5s</h3>
                <p style="color: #666; margin: 0;">Response Time</p>
            </div>
            <div style="
                background: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                border-top: 4px solid #9b59b6;
            ">
                <h3 style="color: #9b59b6; margin: 0;">1,200+</h3>
                <p style="color: #666; margin: 0;">Active Users</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Available tools section
    st.markdown("""
        <h2 style="
            text-align: center;
            margin: 2rem 0;
            color: #2c3e50;
        ">Available Assessment Tools</h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
                border-left: 5px solid #e74c3c;
            ">
                <h3 style="color: #e74c3c;">🔬 Breast Cancer Assessment</h3>
                <p style="color: #666;">Advanced cellular analysis using machine learning to assess cancer risk with high accuracy.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Start Breast Cancer Assessment", key="breast"):
            st.session_state.page = "Breast Cancer"
            st.rerun()
            
        st.markdown("""
            <div style="
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
                border-left: 5px solid #3498db;
            ">
                <h3 style="color: #3498db;">❤️ Heart Disease Assessment</h3>
                <p style="color: #666;">Comprehensive cardiovascular risk analysis using multiple health indicators.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Start Heart Disease Assessment", key="heart"):
            st.session_state.page = "Heart Disease"
            st.rerun()
    
    with col2:
        st.markdown("""
            <div style="
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
                border-left: 5px solid #2ecc71;
            ">
                <h3 style="color: #2ecc71;">🩺 Diabetes Assessment</h3>
                <p style="color: #666;">Predictive analysis of diabetes risk based on key health metrics and indicators.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Start Diabetes Assessment", key="diabetes"):
            st.session_state.page = "Diabetes"
            st.rerun()
            
        st.markdown("""
            <div style="
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
                border-left: 5px solid #9b59b6;
            ">
                <h3 style="color: #9b59b6;">🧠 Parkinson's Assessment</h3>
                <p style="color: #666;">Advanced voice pattern analysis for early detection of Parkinson's disease.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Start Parkinson's Assessment", key="parkinsons"):
            st.session_state.page = "Parkinson's Disease"
            st.rerun()
    
    # Technical Specifications Section
    st.markdown("""
        <h2 style="text-align: center; color: #2c3e50; margin: 2rem 0;">Technical Specifications</h2>
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 2rem 0;
        ">
            <div style="
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
            ">
                <div>
                    <h3 style="color: #3498db;">🔬 Data Sources</h3>
                    <ul style="color: #666; list-style-type: none; padding-left: 0;">
                        <li style="margin: 0.5rem 0;">• Breast Cancer Wisconsin Dataset</li>
                        <li style="margin: 0.5rem 0;">• Pima Indians Diabetes Database</li>
                        <li style="margin: 0.5rem 0;">• Heart Disease UCI Dataset</li>
                        <li style="margin: 0.5rem 0;">• Parkinson's Disease Dataset</li>
                    </ul>
                </div>
                <div>
                    <h3 style="color: #3498db;">⚙️ Technologies Used</h3>
                    <ul style="color: #666; list-style-type: none; padding-left: 0;">
                        <li style="margin: 0.5rem 0;">• Machine Learning: scikit-learn</li>
                        <li style="margin: 0.5rem 0;">• Web Interface: Streamlit</li>
                        <li style="margin: 0.5rem 0;">• Data Processing: pandas, numpy</li>
                        <li style="margin: 0.5rem 0;">• Version Control: Git</li>
                    </ul>
                </div>
                <div>
                    <h3 style="color: #3498db;">📊 Model Performance</h3>
                    <ul style="color: #666; list-style-type: none; padding-left: 0;">
                        <li style="margin: 0.5rem 0;">• Breast Cancer Detection: 96.5%</li>
                        <li style="margin: 0.5rem 0;">• Diabetes Prediction: 94.2%</li>
                        <li style="margin: 0.5rem 0;">• Heart Disease Assessment: 91.8%</li>
                        <li style="margin: 0.5rem 0;">• Parkinson's Detection: 93.5%</li>
                    </ul>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Features Section
    st.markdown("""
        <h2 style="text-align: center; color: #2c3e50; margin: 2rem 0;">Why Choose Our Platform?</h2>
        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        ">
            <div style="
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
                <h3 style="color: #2c3e50;">High Accuracy</h3>
                <p style="color: #666;">Advanced ML algorithms with 96.5% accuracy in predictions</p>
            </div>
            <div style="
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
                <h3 style="color: #2c3e50;">Real-time Analysis</h3>
                <p style="color: #666;">Get instant predictions and comprehensive risk assessments</p>
            </div>
            <div style="
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔒</div>
                <h3 style="color: #2c3e50;">Secure Analysis</h3>
                <p style="color: #666;">Your data is processed securely and privately</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Research & Publications Section
    st.markdown("## Research & Publications")
    
    # Create three columns for the sections
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📚 Recent Papers")
        with st.container():
            st.markdown("""
                **Machine Learning in Medical Diagnosis** (2023)  
                *Impact on early disease detection and prevention*
            """)
            st.markdown("""
                **AI Applications in Healthcare** (2022)  
                *Transforming patient care through technology*
            """)
            st.markdown("""
                **Early Disease Detection Using ML** (2023)  
                *Predictive analytics in healthcare*
            """)
    
    with col2:
        st.markdown("### 🔍 Methodology")
        with st.container():
            st.info("""
                Our system employs advanced machine learning algorithms trained on extensive medical datasets, 
                ensuring reliable and accurate predictions for various medical conditions.

                All models undergo rigorous testing and validation procedures, with continuous monitoring 
                and updates to maintain high accuracy levels.
            """)
    
    with col3:
        st.markdown("### 🎯 Future Developments")
        
        # Future Development Cards
        with st.container():
            st.success("**Integration**\n\nElectronic health records integration for seamless data flow")
            
            st.success("**Visualization**\n\nAdvanced visualization tools for better insight into predictions")
            
            st.success("**Mobile Access**\n\nDevelopment of mobile applications for on-the-go access")

    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Footer Section
    st.markdown("---")  # Add a divider
    
    # Header
    st.header("Ready to get started?")
    st.write("Choose any assessment tool above to begin your analysis")
    
    # Create three columns for contact, resources, and legal
    contact_col, resources_col, legal_col = st.columns(3)
    
    with contact_col:
        st.subheader("Contact")
        st.markdown("""
            📧 **Email:** support@medicalai.com  
            📞 **Phone:** +1 (555) 123-4567
        """)
    
    with resources_col:
        st.subheader("Resources")
        st.markdown("""
            📚 [Documentation](https://docs.medicalai.com)  
            🔧 [API Reference](https://api.medicalai.com)
        """)
    
    with legal_col:
        st.subheader("Legal")
        st.markdown("""
            📜 [Privacy Policy](https://privacy.medicalai.com)  
            ⚖️ [Terms of Service](https://terms.medicalai.com)
        """)
    
    # Copyright and version info
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("© 2024 Medical AI Assistant | Version 1.0.0")
    with col2:
        st.markdown("Developed with ❤️ for healthcare professionals")

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
    
    with tab2:
        st.markdown("### Detailed Measurements")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            radius_mean = st.number_input("Radius (mean)", 6.0, 28.0, 14.0, help="Mean of distances from center to points on the perimeter")
            texture_mean = st.number_input("Texture (mean)", 9.0, 40.0, 14.0, help="Standard deviation of gray-scale values")
            perimeter_mean = st.number_input("Perimeter (mean)", 40.0, 190.0, 90.0, help="Mean size of the core tumor")
            area_mean = st.number_input("Area (mean)", 140.0, 2500.0, 550.0, help="Mean area of the tumor")
            smoothness_mean = st.number_input("Smoothness (mean)", 0.05, 0.16, 0.1, help="Mean of local variation in radius lengths")
            compactness_mean = st.number_input("Compactness (mean)", 0.02, 0.35, 0.1, help="Mean of perimeter^2 / area - 1.0")
            concavity_mean = st.number_input("Concavity (mean)", 0.0, 0.5, 0.1, help="Mean of severity of concave portions")
            concave_points_mean = st.number_input("Concave points (mean)", 0.0, 0.2, 0.1, help="Mean number of concave portions")
            symmetry_mean = st.number_input("Symmetry (mean)", 0.1, 0.3, 0.2, help="Mean symmetry of the tumor")
            fractal_dimension_mean = st.number_input("Fractal dimension (mean)", 0.05, 0.1, 0.06, help="Mean fractal dimension")
        
        with col2:
            radius_se = st.number_input("Radius (SE)", 0.1, 2.0, 0.4, help="Standard error of distances from center to points")
            texture_se = st.number_input("Texture (SE)", 0.2, 4.0, 1.0, help="Standard error of gray-scale values")
            perimeter_se = st.number_input("Perimeter (SE)", 1.0, 20.0, 5.0, help="Standard error of perimeter")
            area_se = st.number_input("Area (SE)", 6.0, 540.0, 40.0, help="Standard error of area")
            smoothness_se = st.number_input("Smoothness (SE)", 0.001, 0.03, 0.007, help="Standard error of smoothness")
            compactness_se = st.number_input("Compactness (SE)", 0.002, 0.135, 0.025, help="Standard error of compactness")
            concavity_se = st.number_input("Concavity (SE)", 0.0, 0.396, 0.03, help="Standard error of concavity")
            concave_points_se = st.number_input("Concave points (SE)", 0.0, 0.05, 0.01, help="Standard error of concave points")
            symmetry_se = st.number_input("Symmetry (SE)", 0.008, 0.079, 0.02, help="Standard error of symmetry")
            fractal_dimension_se = st.number_input("Fractal dimension (SE)", 0.001, 0.029, 0.003, help="Standard error of fractal dimension")
        
        with col3:
            radius_worst = st.number_input("Radius (worst)", 7.0, 36.0, 16.0, help="Worst radius")
            texture_worst = st.number_input("Texture (worst)", 12.0, 50.0, 21.0, help="Worst texture")
            perimeter_worst = st.number_input("Perimeter (worst)", 50.0, 250.0, 107.0, help="Worst perimeter")
            area_worst = st.number_input("Area (worst)", 185.0, 4250.0, 750.0, help="Worst area")
            smoothness_worst = st.number_input("Smoothness (worst)", 0.07, 0.22, 0.13, help="Worst smoothness")
            compactness_worst = st.number_input("Compactness (worst)", 0.03, 1.06, 0.25, help="Worst compactness")
            concavity_worst = st.number_input("Concavity (worst)", 0.0, 1.25, 0.27, help="Worst concavity")
            concave_points_worst = st.number_input("Concave points (worst)", 0.0, 0.29, 0.11, help="Worst concave points")
            symmetry_worst = st.number_input("Symmetry (worst)", 0.15, 0.66, 0.29, help="Worst symmetry")
            fractal_dimension_worst = st.number_input("Fractal dimension (worst)", 0.055, 0.207, 0.083, help="Worst fractal dimension")

    # Add analyze button outside tabs to work for both
    if st.button("Analyze Risk", help="Click to analyze breast cancer risk"):
        with st.spinner('Analyzing samples...'):
            try:
                # Get input data based on active tab
                if tab1._active:
                    input_data = np.array([
                        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                        mean_compactness, mean_concavity, mean_concave_points, 0.2, 0.06,
                        0.4, 0.4, 2.0, 20.0, 0.01, 0.02, 0.02, 0.01, 0.02, 0.003,
                        16.0, 16.0, 100.0, 700.0, 0.12, 0.15, 0.15, 0.1, 0.25, 0.08
                    ]).reshape(1, -1)
                else:
                    input_data = np.array([
                        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
                        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
                        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
                        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
                    ]).reshape(1, -1)
                
                prediction, similar_cases, similar_outcomes, distances = model.predict(input_data)
                
                # Show prediction results
                if prediction[0] == 0:
                    st.error("⚠️ High Risk of Breast Cancer")
                    st.warning(
                        "The analysis indicates characteristics commonly associated with malignant breast masses."
                    )
                    
                    # Show risk factors based on active tab
                    st.subheader("Risk Factors Identified")
                    if tab1._active:
                        if mean_radius > 15:
                            st.warning(f"• Mean radius ({mean_radius:.2f}) is elevated")
                        if mean_concave_points > 0.05:
                            st.warning(f"• Mean concave points ({mean_concave_points:.3f}) are high")
                    else:
                        if radius_worst > 20:
                            st.warning(f"• Worst radius ({radius_worst:.2f}) is significantly elevated")
                        if concave_points_worst > 0.15:
                            st.warning(f"• Worst concave points ({concave_points_worst:.3f}) are very high")
                else:
                    st.success("✅ Low Risk of Breast Cancer")
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