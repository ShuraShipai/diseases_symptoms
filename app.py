import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Set page configuration
st.set_page_config(
    page_title="Medical Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0072B2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0072B2;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E6F3FF;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        text-align: center;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<h1 class='main-header'>Medical Disease Prediction System</h1>", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('dataset.csv')
    return data

# Preprocess the data
def preprocess_data(data):
    # Clean symptom data
    symptom_columns = [col for col in data.columns if col.startswith('Symptom_')]
    for col in symptom_columns:
        data[col] = data[col].str.strip() if data[col].dtype == 'object' else data[col]
    
    # Group symptoms for each disease instance
    data['Symptoms'] = data.apply(
        lambda row: [row[col] for col in symptom_columns if pd.notna(row[col]) and row[col] != ''], 
        axis=1
    )
    
    # Get all unique symptoms
    all_symptoms = set()
    for symptoms in data['Symptoms']:
        all_symptoms.update(symptoms)
    
    return data, list(all_symptoms), symptom_columns

# Feature Engineering
def create_feature_matrix(data, all_symptoms):
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    
    # Transform symptoms to binary features
    feature_matrix = pd.DataFrame(
        mlb.fit_transform(data['Symptoms']),
        columns=mlb.classes_
    )
    
    # Add the disease labels
    feature_matrix['Disease'] = data['Disease']
    
    return feature_matrix, mlb

# Train the model
def train_model(feature_matrix, test_size=0.2, random_state=42):
    # Split features and target
    X = feature_matrix.drop('Disease', axis=1)
    y = feature_matrix['Disease']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Symptom': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, X_train, X_test, y_train, y_test, feature_importance, accuracy

# Predict disease based on symptoms
def predict_disease(symptoms, model, mlb):
    # Convert symptoms to binary feature vector
    symptom_vector = mlb.transform([symptoms])
    
    # Get disease prediction
    disease = model.predict(symptom_vector)[0]
    
    # Get prediction probabilities
    disease_proba = model.predict_proba(symptom_vector)[0]
    disease_indices = model.classes_
    
    # Get top 5 predictions with probabilities
    top_indices = disease_proba.argsort()[-5:][::-1]
    top_diseases = [(disease_indices[i], disease_proba[i]) for i in top_indices]
    
    return disease, top_diseases

# Main application
def main():
    # Loading data
    with st.spinner('Loading and preprocessing data...'):
        data = load_data()
        data, all_symptoms, symptom_columns = preprocess_data(data)
    
    # Sidebar
    st.sidebar.markdown("<h2 class='sub-header'>Model Configuration</h2>", unsafe_allow_html=True)
    
    # Model settings
    test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.2, 0.05)
    n_estimators = st.sidebar.slider('Number of Trees', 50, 500, 100, 50)
    
    # Display dataset info
    st.sidebar.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
    st.sidebar.write(f"Total records: {len(data)}")
    st.sidebar.write(f"Unique diseases: {data['Disease'].nunique()}")
    st.sidebar.write(f"Unique symptoms: {len(all_symptoms)}")
    
    # Main content - tabs
    tab1, tab2, tab3 = st.tabs(["Disease Prediction", "Model Training", "Dataset Exploration"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Symptom-Based Disease Prediction</h2>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Select the symptoms the patient is experiencing to get a prediction of possible diseases.</div>", unsafe_allow_html=True)
        
        # Symptom selection
        selected_symptoms = st.multiselect(
            "Select Symptoms:",
            options=sorted(all_symptoms),
            help="You can select multiple symptoms"
        )
        
        if st.button("Predict Disease"):
            if len(selected_symptoms) == 0:
                st.warning("Please select at least one symptom")
            else:
                with st.spinner('Predicting...'):
                    # Train the model if not already trained
                    feature_matrix, mlb = create_feature_matrix(data, all_symptoms)
                    model, _, _, _, _, _, _ = train_model(feature_matrix, test_size=test_size)
                    
                    # Make prediction
                    disease, top_diseases = predict_disease(selected_symptoms, model, mlb)
                    
                    # Display prediction results
                    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                    st.markdown(f"<h2>Top Prediction: {disease}</h2>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display top 5 predictions with confidence
                    st.markdown("<h3 class='sub-header'>Alternative Possibilities</h3>", unsafe_allow_html=True)
                    
                    # Create DataFrame for display
                    results_df = pd.DataFrame(
                        top_diseases, 
                        columns=['Disease', 'Confidence']
                    )
                    
                    # Plot confidence bars
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = sns.barplot(y='Disease', x='Confidence', data=results_df, palette='Blues_d', ax=ax)
                    
                    # Add percentage labels
                    for i, p in enumerate(bars.patches):
                        width = p.get_width()
                        ax.text(width + 0.01, p.get_y() + p.get_height()/2, 
                                f'{width:.1%}', ha='left', va='center')
                    
                    plt.title('Disease Prediction Confidence')
                    plt.xlim(0, 1)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Display note about accuracy
                    st.info("Note: This prediction is based on the symptoms provided. Always consult with a qualified healthcare professional for proper diagnosis.")
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Model Training and Evaluation</h2>", unsafe_allow_html=True)
        
        if st.button("Train Model"):
            with st.spinner('Training model...'):
                # Create feature matrix
                feature_matrix, mlb = create_feature_matrix(data, all_symptoms)
                
                # Train the model
                model, X_train, X_test, y_train, y_test, feature_importance, accuracy = train_model(
                    feature_matrix, 
                    test_size=test_size,
                    random_state=42
                )
                
                # Display results
                st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
                
                # Display feature importance
                st.markdown("<h3 class='sub-header'>Most Important Symptoms</h3>", unsafe_allow_html=True)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(y='Symptom', x='Importance', data=feature_importance.head(15), palette='Blues_d', ax=ax)
                plt.title('Top 15 Most Important Symptoms')
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Model performance metrics
                st.markdown("<h3 class='sub-header'>Model Performance</h3>", unsafe_allow_html=True)
                
                # Make predictions on test set
                y_pred = model.predict(X_test)
                
                # Create metrics columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                
                with col2:
                    # Calculate precision (macro avg)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    precision = report['macro avg']['precision']
                    st.metric("Precision", f"{precision:.2%}")
                
                with col3:
                    # Calculate recall (macro avg)
                    recall = report['macro avg']['recall']
                    st.metric("Recall", f"{recall:.2%}")
    
    with tab3:
        st.markdown("<h2 class='sub-header'>Dataset Exploration</h2>", unsafe_allow_html=True)
        
        # Disease distribution
        st.markdown("<h3 class='sub-header'>Disease Distribution</h3>", unsafe_allow_html=True)
        
        # Count diseases
        disease_counts = data['Disease'].value_counts()
        
        # Plot disease distribution
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(y=disease_counts.head(15).index, x=disease_counts.head(15).values, palette='Blues_d', ax=ax)
        plt.title('Top 15 Diseases by Frequency')
        plt.xlabel('Count')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Symptom distribution
        st.markdown("<h3 class='sub-header'>Symptom Distribution</h3>", unsafe_allow_html=True)
        
        # Count symptom occurrences
        symptom_counts = {}
        for symptoms_list in data['Symptoms']:
            for symptom in symptoms_list:
                symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
        
        # Convert to DataFrame for plotting
        symptom_df = pd.DataFrame({
            'Symptom': list(symptom_counts.keys()),
            'Count': list(symptom_counts.values())
        }).sort_values('Count', ascending=False)
        
        # Plot top symptoms
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(y='Symptom', x='Count', data=symptom_df.head(15), palette='Blues_d', ax=ax)
        plt.title('Top 15 Most Common Symptoms')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Raw data preview
        st.markdown("<h3 class='sub-header'>Raw Data Preview</h3>", unsafe_allow_html=True)
        
        sample_size = st.slider("Sample size", 5, 100, 10)
        st.dataframe(data.sample(sample_size))

    # Footer
    st.markdown("<div class='footer'>This application is for educational purposes only. Always consult with healthcare professionals for medical advice.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()