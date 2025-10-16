"""
Chest X-Ray Disease Classifier - Streamlit Frontend
Multi-label disease detection with Grad-CAM visualization
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
import os
from pathlib import Path

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Chest X-Ray Disease Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disease-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .disclaimer {
        background-color: #fff9c4;
        border-left: 5px solid #fbc02d;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DISEASE CONFIGURATION ====================
DISEASES = [
    'Cardiomegaly', 'Edema', 'Emphysema', 'Hernia', 
    'Pneumonia', 'Fibrosis', 'Pneumothorax'
]

DISEASE_INFO = {
    'Cardiomegaly': '❤️ Enlarged heart - May indicate heart disease or high blood pressure',
    'Edema': '💧 Fluid accumulation in lungs - Often associated with heart failure',
    'Emphysema': '🫁 Lung tissue damage - Usually caused by smoking',
    'Hernia': '⚕️ Organ displacement - Requires surgical evaluation',
    'Pneumonia': '🦠 Lung infection - Bacterial or viral inflammation',
    'Fibrosis': '🔬 Lung scarring - Progressive lung tissue damage',
    'Pneumothorax': '💨 Collapsed lung - Air in pleural space'
}

# ==================== MODEL DEFINITION ====================
class DenseNetClassifier(nn.Module):
    def __init__(self, model_name='densenet121', num_classes=7):
        super(DenseNetClassifier, self).__init__()
        
        if model_name == 'densenet121':
            self.model = models.densenet121(pretrained=False)
            num_ftrs = self.model.classifier.in_features
        elif model_name == 'densenet169':
            self.model = models.densenet169(pretrained=False)
            num_ftrs = self.model.classifier.in_features
        elif model_name == 'densenet201':
            self.model = models.densenet201(pretrained=False)
            num_ftrs = self.model.classifier.in_features
        else:
            raise ValueError(f'Unknown model: {model_name}')
        
        # Replace classifier
        self.model.classifier = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

# ==================== GRAD-CAM IMPLEMENTATION ====================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(model_output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = model_output[0, target_class]
        class_loss.backward()
        
        # Generate CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # Average the channels
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # Apply ReLU
        heatmap = torch.clamp(heatmap, min=0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap.cpu().numpy()

# ==================== HELPER FUNCTIONS ====================
@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        model = DenseNetClassifier(model_name='densenet121', num_classes=7)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image, image_size=224):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transform
    img_tensor = transform(image).unsqueeze(0)
    
    return img_tensor

def apply_heatmap_overlay(image, heatmap, alpha=0.4):
    """Apply heatmap overlay on original image"""
    # Convert PIL to numpy array first
    img_array = np.array(image)
    
    # Get image dimensions (height, width)
    img_height, img_width = img_array.shape[:2]
    
    # Resize heatmap to match image dimensions (width, height for cv2.resize)
    heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
    
    # Convert to colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure both arrays have the same shape
    if img_array.shape != heatmap_colored.shape:
        # If grayscale, convert to RGB
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        # If different dimensions, resize heatmap again
        if img_array.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (img_array.shape[1], img_array.shape[0]))
    
    # Ensure both are uint8
    img_array = img_array.astype(np.uint8)
    heatmap_colored = heatmap_colored.astype(np.uint8)
    
    # Overlay
    overlayed = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(overlayed)

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability >= 0.7:
        return "High", "high-risk", "🔴"
    elif probability >= 0.4:
        return "Medium", "medium-risk", "🟡"
    else:
        return "Low", "low-risk", "🟢"

def predict_diseases(model, image_tensor, device):
    """Make predictions for all diseases"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
    
    return probabilities

def generate_gradcam_for_disease(model, image_tensor, device, disease_idx):
    """Generate Grad-CAM for specific disease"""
    try:
        # Get the last convolutional layer
        if hasattr(model.model, 'features'):
            target_layer = model.model.features[-1]
        else:
            # Fallback - find last Conv2d layer
            for module in reversed(list(model.model.modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
            else:
                target_layer = list(model.model.children())[-2]
        
        # Create Grad-CAM object
        grad_cam = GradCAM(model, target_layer)
        
        # Generate heatmap
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad = True
        
        heatmap = grad_cam.generate_cam(image_tensor, target_class=disease_idx)
        
        return heatmap
    except Exception as e:
        print(f"Grad-CAM generation error: {str(e)}")
        # Return empty heatmap on error
        return np.zeros((7, 7))

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<div class="main-header">🏥 Chest X-Ray Disease Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Multi-Label Disease Detection with Explainable AI</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model path
        model_path = st.text_input(
            "Model Path",
            value="best_model.pth",
            help="Path to the trained model file"
        )
        
        # Visualization options
        st.subheader("📊 Visualization Options")
        show_gradcam = st.checkbox("Show Grad-CAM Heatmaps", value=True)
        heatmap_alpha = st.slider("Heatmap Opacity", 0.0, 1.0, 0.4, 0.1)
        
        # Threshold
        st.subheader("🎯 Detection Threshold")
        threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.3, 0.05)
        
        st.markdown("---")
        st.info("**Model:** DenseNet-121\n\n**Diseases:** 7 thoracic conditions\n\n**Input:** 224x224 RGB X-Ray")
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please ensure 'best_model.pth' is in the same directory as this app.")
        return
    
    # Load model
    with st.spinner("🔄 Loading model..."):
        model, device = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the model file.")
        return
    
    st.success(f"✅ Model loaded successfully! Running on: **{device}**")
    
    # File uploader
    st.markdown("---")
    st.header("📤 Upload Chest X-Ray Image")
    
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a frontal chest X-ray for analysis"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Original X-Ray")
            st.image(image, use_container_width=True)
        
        # Analyze button
        if st.button("🔬 Analyze X-Ray", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing X-ray image..."):
                # Preprocess
                img_tensor = preprocess_image(image)
                
                # Predict
                probabilities = predict_diseases(model, img_tensor, device)
                
                # Store results in session state
                st.session_state['probabilities'] = probabilities
                st.session_state['image'] = image
                st.session_state['img_tensor'] = img_tensor
        
        # Display results if available
        if 'probabilities' in st.session_state:
            probabilities = st.session_state['probabilities']
            image = st.session_state['image']
            img_tensor = st.session_state['img_tensor']
            
            st.markdown("---")
            st.header("📊 Analysis Results")
            
            # Top findings
            top_indices = np.argsort(probabilities)[::-1][:3]
            
            st.subheader("🎯 Top Findings")
            top_cols = st.columns(3)
            
            for idx, disease_idx in enumerate(top_indices):
                disease = DISEASES[disease_idx]
                prob = probabilities[disease_idx]
                risk_level, risk_class, emoji = get_risk_level(prob)
                
                with top_cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{emoji} {disease}</h3>
                        <h1>{prob*100:.1f}%</h1>
                        <p>{risk_level} Risk</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed results
            st.markdown("---")
            st.subheader("🏥 Detailed Disease Analysis")
            
            # Sort by probability
            sorted_indices = np.argsort(probabilities)[::-1]
            
            for disease_idx in sorted_indices:
                disease = DISEASES[disease_idx]
                prob = probabilities[disease_idx]
                risk_level, risk_class, emoji = get_risk_level(prob)
                
                with st.expander(f"{emoji} **{disease}** - {prob*100:.1f}% ({risk_level} Risk)", expanded=(prob >= threshold)):
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.markdown(f"**Probability:** {prob*100:.2f}%")
                        st.progress(float(prob))
                        st.info(DISEASE_INFO[disease])
                        
                        if prob >= threshold:
                            st.warning(f"⚠️ Detection threshold exceeded ({threshold*100:.0f}%)")
                    
                    with col_b:
                        if show_gradcam and prob >= 0.2:  # Only show for significant predictions
                            with st.spinner(f"Generating heatmap for {disease}..."):
                                try:
                                    # Reload model in eval mode to ensure clean state
                                    model.eval()
                                    
                                    heatmap = generate_gradcam_for_disease(
                                        model, img_tensor.clone(), device, disease_idx
                                    )
                                    
                                    # Validate heatmap
                                    if heatmap is not None and heatmap.size > 0:
                                        # Apply overlay
                                        overlayed_img = apply_heatmap_overlay(
                                            image, heatmap, alpha=heatmap_alpha
                                        )
                                        
                                        st.image(overlayed_img, caption=f"Focus areas for {disease}", use_container_width=True)
                                    else:
                                        st.warning("Heatmap generation skipped (invalid output)")
                                        
                                except Exception as e:
                                    st.error(f"Could not generate heatmap: {str(e)}")
                                    st.info("💡 Tip: Try reloading the page or uploading a different image")
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Probability Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = [get_risk_level(p)[1] for p in probabilities]
            color_map = {
                'high-risk': '#f44336',
                'medium-risk': '#ff9800',
                'low-risk': '#4caf50'
            }
            bar_colors = [color_map[c] for c in colors]
            
            bars = ax.barh(DISEASES, probabilities * 100, color=bar_colors)
            ax.axvline(x=threshold*100, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold*100:.0f}%)')
            ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Disease', fontsize=12, fontweight='bold')
            ax.set_title('Disease Detection Probabilities', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)
            
            # Add percentage labels
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                       f'{prob*100:.1f}%', 
                       ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Summary statistics
            st.markdown("---")
            st.subheader("📋 Summary Statistics")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Max Probability", f"{probabilities.max()*100:.1f}%")
            
            with stat_col2:
                st.metric("Avg Probability", f"{probabilities.mean()*100:.1f}%")
            
            with stat_col3:
                detected = sum(probabilities >= threshold)
                st.metric("Conditions Detected", f"{detected}/7")
            
            with stat_col4:
                st.metric("Most Likely", DISEASES[probabilities.argmax()])
            
            # Disclaimer
            st.markdown("---")
            st.markdown("""
            <div class="disclaimer">
                <h3>⚠️ Medical Disclaimer</h3>
                <p>
                    This tool is for <strong>educational and research purposes only</strong>. 
                    It should NOT be used as a substitute for professional medical diagnosis or treatment. 
                    Always consult with qualified healthcare professionals for medical advice.
                </p>
                <p>
                    <strong>Model Information:</strong> DenseNet-121 trained on NIH Chest X-ray dataset. 
                    The model's predictions are probabilistic and may not always be accurate.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Landing page instructions
        st.info("""
        ### 🚀 How to Use:
        1. **Upload** a chest X-ray image using the file uploader above
        2. **Click** the "Analyze X-Ray" button
        3. **Review** the disease predictions and probability scores
        4. **Examine** the Grad-CAM heatmaps to see which regions influenced the predictions
        5. **Adjust** the threshold and visualization settings in the sidebar
        
        ### 📝 Supported Diseases:
        """)
        
        cols = st.columns(2)
        for idx, disease in enumerate(DISEASES):
            with cols[idx % 2]:
                st.markdown(f"- **{disease}**: {DISEASE_INFO[disease]}")
        
        st.markdown("---")
        st.success("✨ Upload an X-ray image to get started!")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
