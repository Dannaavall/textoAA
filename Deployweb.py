import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from bs4 import BeautifulSoup
import os

# Page configuration
st.set_page_config(
    page_title="ML Final Project - Inference App",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cnn_model' not in st.session_state:
    st.session_state.cnn_model = None
if 'text_model' not in st.session_state:
    st.session_state.text_model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

# Constants from notebooks
IMAGE_SIZE = (128, 128)
FRUIT_CLASSES = ['Apple', 'Banana', 'Mango', 'Strawberry']
TEXT_MAXLEN = 100
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128

# Text preprocessing function (exact same as notebook)
def clean_text(text):
    """Clean text exactly as in Part 2 notebook"""
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)               # Remove non-alphabetic characters
    text = text.lower()                                    # Lowercase
    text = text.strip()                                    # Remove extra spaces
    return text

# Image preprocessing function (exact same as notebook)
def preprocess_image(image):
    """Preprocess image exactly as in Part 1 notebook"""
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image) / 255.0  # Rescale to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Load CNN model
@st.cache_resource
def load_cnn_model(model_path):
    """Load the trained CNN model"""
    try:
        # Try loading with compile=False first (more compatible)
        try:
            model = keras.models.load_model(model_path, compile=False)
        except Exception:
            # Fallback to default loading
            model = keras.models.load_model(model_path)
        
        if model is None:
            raise Exception("Model loaded but is None")
        
        return model
    except Exception as e:
        error_msg = str(e)
        if "file signature not found" in error_msg.lower():
            st.error("""
            **Error: File signature not found**
            
            The model file appears to be corrupted or incomplete.
            Please re-download the model from your Colab notebook.
            """)
        else:
            st.error(f"Error loading CNN model: {error_msg}")
        return None

# Load text model
@st.cache_resource
def load_text_model(model_path):
    """Load the trained text model (LSTM)"""
    try:
        # Try loading with different methods for compatibility
        try:
            model = keras.models.load_model(model_path, compile=False)
        except Exception as e1:
            # If that fails, try with compile=True
            try:
                model = keras.models.load_model(model_path, compile=True)
            except Exception as e2:
                # If both fail, try loading with custom_objects
                try:
                    model = keras.models.load_model(
                        model_path,
                        custom_objects={'Embedding': keras.layers.Embedding,
                                      'LSTM': keras.layers.LSTM,
                                      'Dense': keras.layers.Dense}
                    )
                except Exception as e3:
                    raise Exception(f"Multiple load attempts failed. Last error: {str(e3)}")
        
        # Verify model structure
        if model is None:
            raise Exception("Model loaded but is None")
        
        return model
    except Exception as e:
        error_msg = str(e)
        if "file signature not found" in error_msg.lower():
            st.error("""
            **Error: File signature not found**
            
            This usually means:
            1. The model file is corrupted or incomplete
            2. The file wasn't fully downloaded
            3. TensorFlow version mismatch
            
            **Solutions:**
            - Re-download the model from Colab
            - Check file size (should be > 10 MB)
            - Ensure TensorFlow >= 2.13.0
            - Try saving the model again with `save_format='h5'` explicitly
            """)
        else:
            st.error(f"Error loading text model: {error_msg}")
        return None

# Load tokenizer
@st.cache_resource
def load_tokenizer(tokenizer_path):
    """Load the tokenizer"""
    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None

# Main app
def main():
    st.markdown('<h1 class="main-header">🤖 Machine Learning Final Project - Inference App</h1>', unsafe_allow_html=True)
    
    # Sidebar for model loading
    with st.sidebar:
        st.header("📁 Model Loading")
        
        st.subheader("Image Classification Model")
        cnn_model_file = st.file_uploader("Upload CNN model (.h5 or .keras)", type=['h5', 'keras'], key='cnn_upload')
        if cnn_model_file:
            with open("temp_cnn_model.h5", "wb") as f:
                f.write(cnn_model_file.getbuffer())
            st.session_state.cnn_model = load_cnn_model("temp_cnn_model.h5")
            if st.session_state.cnn_model:
                st.success("✅ CNN model loaded successfully!")
        
        st.subheader("Text Classification Model")
        text_model_file = st.file_uploader("Upload Text model (.h5 or .keras)", type=['h5', 'keras'], key='text_upload')
        if text_model_file:
            # Show file info
            file_size = len(text_model_file.getbuffer()) / (1024 * 1024)  # Size in MB
            st.info(f"📦 File size: {file_size:.2f} MB")
            
            if file_size < 0.1:  # Less than 100 KB is suspicious
                st.warning("⚠️ File seems too small. It might be corrupted.")
            
            with open("temp_text_model.h5", "wb") as f:
                f.write(text_model_file.getbuffer())
            
            with st.spinner("Loading model..."):
                st.session_state.text_model = load_text_model("temp_text_model.h5")
            
            if st.session_state.text_model:
                st.success("✅ Text model loaded successfully!")
                # Show model summary info
                try:
                    total_params = st.session_state.text_model.count_params()
                    st.caption(f"📊 Model parameters: {total_params:,}")
                except:
                    pass
        
        st.subheader("Tokenizer")
        tokenizer_file = st.file_uploader("Upload tokenizer (.pkl)", type=['pkl'], key='tokenizer_upload')
        if tokenizer_file:
            with open("temp_tokenizer.pkl", "wb") as f:
                f.write(tokenizer_file.getbuffer())
            st.session_state.tokenizer = load_tokenizer("temp_tokenizer.pkl")
            if st.session_state.tokenizer:
                st.success("✅ Tokenizer loaded successfully!")
        
        st.markdown("---")
        
        # Diagnostic section
        with st.expander("🔧 Troubleshooting"):
            st.write("""
            **Common Issues:**
            
            1. **"File signature not found"**
               - File is corrupted or incomplete
               - Solution: Re-download from Colab, ensure full download
            
            2. **Model won't load**
               - Check TensorFlow version (>= 2.13.0)
               - Verify file size (model should be > 10 MB)
               - Try saving with `save_format='h5'` explicitly
            
            3. **Tokenizer errors**
               - Ensure tokenizer was saved with pickle
               - Check file extension is .pkl
            """)
        
        st.info("💡 **Note:** Models should be saved from your training notebooks. See README for details.")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Classification", "📝 Text Classification", "📊 Model Information"])
    
    # Tab 1: Image Classification
    with tab1:
        st.markdown('<h2 class="section-header">Image Classification - Fruit Recognition</h2>', unsafe_allow_html=True)
        
        if st.session_state.cnn_model is None:
            st.warning("⚠️ Please load the CNN model from the sidebar first.")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Upload Image")
                uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key='image_upload')
                
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    prediction = st.session_state.cnn_model.predict(processed_image, verbose=0)
                    probabilities = prediction[0]
                    predicted_class_idx = np.argmax(probabilities)
                    predicted_class = FRUIT_CLASSES[predicted_class_idx]
                    confidence = probabilities[predicted_class_idx]
                    
                    # Display results
                    st.success(f"🎯 **Predicted Class:** {predicted_class}")
                    st.info(f"📊 **Confidence:** {confidence*100:.2f}%")
                    
                    # Top 3 predictions
                    st.subheader("Top 3 Predictions")
                    top_3_indices = np.argsort(probabilities)[-3:][::-1]
                    for i, idx in enumerate(top_3_indices, 1):
                        st.write(f"{i}. **{FRUIT_CLASSES[idx]}**: {probabilities[idx]*100:.2f}%")
                    
                    # Probability bar chart
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.barh(FRUIT_CLASSES, probabilities * 100)
                    ax.set_xlabel('Probability (%)')
                    ax.set_title('Class Probabilities')
                    ax.set_xlim(0, 100)
                    st.pyplot(fig)
            
            with col2:
                st.subheader("Model Information")
                st.write("**Model Type:** CNN (Convolutional Neural Network)")
                st.write("**Dataset:** Fruits Dataset")
                st.write("**Classes:** Apple, Banana, Mango, Strawberry")
                st.write("**Image Size:** 128x128 pixels")
                st.write("**Preprocessing:** Rescale to [0, 1] (divide by 255)")
                
                st.subheader("How it works:")
                st.write("""
                1. Upload an image of a fruit
                2. Image is resized to 128x128 pixels
                3. Pixel values are normalized to [0, 1]
                4. CNN model predicts the fruit class
                5. Results show predicted class and probabilities
                """)
    
    # Tab 2: Text Classification
    with tab2:
        st.markdown('<h2 class="section-header">Text Classification - Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.text_model is None or st.session_state.tokenizer is None:
            st.warning("⚠️ Please load the text model and tokenizer from the sidebar first.")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Enter Text")
                text_input = st.text_area("Enter your review or text:", height=200, key='text_input')
                
                if st.button("Analyze Sentiment", key='analyze_btn'):
                    if text_input.strip():
                        # Preprocess text (exact same as notebook)
                        cleaned_text = clean_text(text_input)
                        
                        # Tokenize and pad
                        sequence = st.session_state.tokenizer.texts_to_sequences([cleaned_text])
                        padded_sequence = pad_sequences(sequence, maxlen=TEXT_MAXLEN, padding='post', truncating='post')
                        
                        # Predict
                        prediction = st.session_state.text_model.predict(padded_sequence, verbose=0)
                        probability = prediction[0][0]
                        sentiment = "Positive" if probability > 0.5 else "Negative"
                        confidence = probability if sentiment == "Positive" else (1 - probability)
                        
                        # Display results
                        if sentiment == "Positive":
                            st.success(f"😊 **Sentiment:** {sentiment}")
                        else:
                            st.error(f"😞 **Sentiment:** {sentiment}")
                        
                        st.info(f"📊 **Confidence:** {confidence*100:.2f}%")
                        st.write(f"**Raw Score:** {probability:.4f}")
                        
                        # Probability gauge
                        fig, ax = plt.subplots(figsize=(8, 2))
                        ax.barh(['Sentiment'], [probability * 100], color='green' if sentiment == 'Positive' else 'red')
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('Probability (%)')
                        ax.set_title('Sentiment Probability')
                        st.pyplot(fig)
                    else:
                        st.warning("Please enter some text to analyze.")
            
            with col2:
                st.subheader("Model Information")
                st.write("**Model Type:** LSTM (Long Short-Term Memory)")
                st.write("**Dataset:** IMDB Movie Reviews")
                st.write("**Classes:** Positive, Negative")
                st.write("**Max Sequence Length:** 100 tokens")
                st.write("**Vocabulary Size:** 10,000")
                st.write("**Embedding Dimension:** 128")
                
                st.subheader("Preprocessing Steps:")
                st.write("""
                1. Remove HTML tags
                2. Remove non-alphabetic characters
                3. Convert to lowercase
                4. Tokenize and convert to sequences
                5. Pad/truncate to 100 tokens
                6. Model predicts sentiment
                """)
                
                st.subheader("Example Texts:")
                example_texts = [
                    ("Positive", "This movie was absolutely fantastic! I loved every minute of it."),
                    ("Negative", "Terrible movie. Waste of time and money. Very disappointing.")
                ]
                for sentiment, text in example_texts:
                    with st.expander(f"{sentiment} Example"):
                        st.write(text)
    
    # Tab 3: Model Information
    with tab3:
        st.markdown('<h2 class="section-header">Model Information & Performance</h2>', unsafe_allow_html=True)
        
        # Image Classification Section
        st.header("🖼️ Image Classification Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.write("""
            **Dataset:** Fruits Dataset
            - **Classes:** 4 (Apple, Banana, Mango, Strawberry)
            - **Training Images:** 64
            - **Validation Images:** 16
            - **Image Size:** 128x128 pixels
            - **Format:** RGB
            """)
            
            st.subheader("Model Architecture")
            st.write("""
            **CNN Simple (Best Model)**
            - Convolutional layers for feature extraction
            - Pooling layers for dimensionality reduction
            - Dense layers for classification
            - Optimizer: Adam
            - Loss: Categorical Crossentropy
            """)
        
        with col2:
            st.subheader("Performance Metrics")
            st.metric("Validation Accuracy", "62.50%")
            st.metric("Best Model", "CNN Simple + Adam")
            st.metric("Epochs Trained", "5")
            
            # Placeholder for confusion matrix
            st.subheader("Confusion Matrix")
            # Example confusion matrix (you should replace with actual values)
            cm_data = np.array([[0, 0, 0, 0],
                               [1, 3, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Greens',
                       xticklabels=FRUIT_CLASSES, yticklabels=FRUIT_CLASSES,
                       ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix - Image Classification')
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Text Classification Section
        st.header("📝 Text Classification Model")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Dataset Information")
            st.write("""
            **Dataset:** IMDB Movie Reviews
            - **Total Reviews:** 50,000
            - **Training:** 40,000
            - **Testing:** 10,000
            - **Classes:** 2 (Positive, Negative)
            - **Balanced:** Yes (25,000 each)
            """)
            
            st.subheader("Model Architecture")
            st.write("""
            **LSTM (Best Model)**
            - Embedding layer (vocab_size=10,000, dim=128)
            - LSTM layer (64 units, dropout=0.3)
            - Dense output layer (sigmoid)
            - Optimizer: Adam
            - Loss: Binary Crossentropy
            """)
        
        with col4:
            st.subheader("Performance Metrics")
            st.metric("Validation Accuracy", "82.53%")
            st.metric("F1 Score", "0.83")
            st.metric("Best Model", "LSTM")
            st.metric("Training Time", "472.26 seconds")
            
            # Placeholder for confusion matrix
            st.subheader("Confusion Matrix")
            cm_text = np.array([[4318, 643],
                               [1105, 3934]])
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_text, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix - Text Classification')
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Training Graphs Section
        st.header("📈 Training History")
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Image Classification - Accuracy")
            # Example training history (replace with actual data)
            epochs = np.arange(1, 6)
            train_acc = [0.43, 0.54, 0.61, 0.63, 0.63]
            val_acc = [0.50, 0.38, 0.38, 0.63, 0.38]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, train_acc, 'o-', label='Training', linewidth=2)
            ax.plot(epochs, val_acc, 's-', label='Validation', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('CNN Training History - Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col6:
            st.subheader("Text Classification - Accuracy")
            # Example training history (replace with actual data)
            epochs_text = np.arange(1, 7)
            train_acc_text = [0.64, 0.77, 0.81, 0.84, 0.86, 0.87]
            val_acc_text = [0.78, 0.77, 0.82, 0.82, 0.83, 0.83]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs_text, train_acc_text, 'o-', label='Training', linewidth=2)
            ax.plot(epochs_text, val_acc_text, 's-', label='Validation', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('LSTM Training History - Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Model Limitations
        st.header("⚠️ Model Limitations")
        
        col7, col8 = st.columns(2)
        
        with col7:
            st.subheader("Image Classification")
            st.write("""
            1. **Small Dataset:** Only 80 images total, which limits generalization
            2. **Limited Classes:** Only 4 fruit types, may not work well for other fruits
            3. **Image Quality:** Performance depends on image quality and lighting
            4. **Background:** May be affected by complex backgrounds
            5. **Overfitting:** Model shows signs of overfitting (validation accuracy fluctuates)
            """)
        
        with col8:
            st.subheader("Text Classification")
            st.write("""
            1. **Sequence Length:** Limited to 100 tokens, longer texts are truncated
            2. **Vocabulary:** Only 10,000 most common words, rare words become <OOV>
            3. **Domain Specific:** Trained on movie reviews, may not generalize to other domains
            4. **Context:** May struggle with sarcasm, irony, or complex language
            5. **Binary Only:** Only classifies as positive/negative, no neutral sentiment
            """)

if __name__ == "__main__":
    main()

