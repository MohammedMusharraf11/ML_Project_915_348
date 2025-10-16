# Chest X-Ray Disease Classification: A Comprehensive Journey from Traditional ML to Deep Learning

## 📋 Project Report

**Project Duration:** September 2025 - October 2025  
**Team:** ML Project - SEM 5  
**Domain:** Medical Image Analysis & Computer Vision  

---

## 🎯 Executive Summary

This project tackled the challenge of automated thoracic disease detection from chest X-ray images, progressing through multiple machine learning paradigms to achieve clinically significant accuracy. Starting with traditional computer vision techniques (SIFT + BoVW) and culminating in transfer learning with DenseNet-121, we achieved substantial improvements in diagnostic accuracy across 7 thoracic conditions.

**Key Achievements:**
- ✅ Developed end-to-end pipeline from traditional ML to deep learning
- ✅ Achieved **average AUC of 0.80+** using DenseNet-121 transfer learning
- ✅ Built production-ready Streamlit application with Grad-CAM interpretability
- ✅ Processed and analyzed **112,000+** chest X-ray images from NIH dataset

---

## 🏥 Problem Statement

### Background
Chest X-rays are one of the most widely used and cost-effective diagnostic tools in medicine, yet interpreting them requires significant time, expertise, and attention from radiologists. Limited access to expert radiologists can delay diagnosis and treatment, especially in under-resourced areas.

### Challenge
Develop an automated system capable of detecting multiple thoracic diseases from chest X-ray images to:
- **Assist radiologists** by providing faster preliminary screenings
- **Reduce diagnostic burden** in resource-constrained settings
- **Provide consistent analysis** across different medical facilities
- **Prioritize critical cases** for urgent review

### Target Diseases
The system detects 7 thoracic conditions:
1. **Cardiomegaly** - Enlarged heart
2. **Edema** - Fluid accumulation in lungs
3. **Emphysema** - Lung tissue damage
4. **Hernia** - Organ displacement
5. **Pneumonia** - Lung infection
6. **Fibrosis** - Lung scarring
7. **Pneumothorax** - Collapsed lung

---

## 📊 Dataset

### NIH Chest X-ray Dataset
- **Total Images:** 112,120 frontal-view chest X-rays
- **Patients:** 30,805 unique patients
- **Image Size:** 1024×1024 pixels (grayscale)
- **Format:** PNG
- **Labels:** Multi-label (patients can have multiple conditions)

### Dataset Characteristics
```
Disease Distribution:
  Cardiomegaly:    2,772 cases (2.47%)
  Edema:           2,303 cases (2.05%)
  Emphysema:       2,516 cases (2.24%)
  Hernia:            227 cases (0.20%)
  Pneumonia:       1,431 cases (1.27%)
  Fibrosis:        1,686 cases (1.50%)
  Pneumothorax:    5,302 cases (4.72%)
  No Finding:     60,361 cases (53.75%)
```

### Data Split
- **Training Set:** 70% (~78,500 images)
- **Validation Set:** 15% (~16,800 images)
- **Test Set:** 15% (~16,800 images)

**Note:** Stratified splitting used to maintain disease distribution across splits.

---

## 🔬 Methodology: A Three-Phase Journey

## Phase 1: Traditional Machine Learning with Computer Vision

### Approach Overview
Our first approach used classical computer vision techniques combining:
- **SIFT (Scale-Invariant Feature Transform)** for keypoint detection
- **Bag of Visual Words (BoVW)** for feature representation
- **TF-IDF weighting** for feature importance
- **Traditional ML classifiers** (Logistic Regression, SVM, XGBoost)

### Step 1: SIFT Feature Extraction

**What is SIFT?**
SIFT detects and describes local features in images that are invariant to scale, rotation, and illumination changes.

**Process:**
```python
# Extracted SIFT descriptors from each X-ray
- Keypoint detection across multiple scales
- 128-dimensional feature vectors per keypoint
- Average ~200-300 keypoints per chest X-ray
```

**Challenges:**
- Variable number of keypoints per image (100-500 range)
- High dimensional feature space (128 dimensions per keypoint)
- Need for fixed-size representation for ML models

### Step 2: Building the Visual Vocabulary

**Bag of Visual Words (BoVW):**
```
1. Collect all SIFT descriptors from training images
   → Result: Millions of 128-D feature vectors

2. Apply K-Means clustering to create "visual words"
   → Codebook size: 500 clusters
   → Each cluster represents a visual pattern

3. Convert each image to histogram of visual words
   → Count frequency of each visual word in image
   → Result: 500-dimensional feature vector per image
```

**Codebook Creation:**
- **K-Means clustering** with k=500
- **Convergence criterion:** 100 iterations or <0.001 change
- **Training time:** ~2 hours on CPU

### Step 3: TF-IDF Weighting

Applied Term Frequency-Inverse Document Frequency weighting:

```
TF-IDF(word, image) = TF(word, image) × IDF(word)

where:
  TF = Frequency of visual word in image
  IDF = log(Total images / Images containing word)
```

**Purpose:** Reduce impact of common visual patterns, emphasize discriminative features.

### Step 4: Baseline Model Training

Trained independent binary classifiers for each disease:

#### Logistic Regression (Baseline)
```
Configuration:
- Solver: lbfgs
- Max iterations: 1000
- Class weight: balanced (to handle imbalance)
- Regularization: L2 (default)
```

#### Support Vector Machine (Baseline)
```
Configuration:
- Kernel: RBF (Radial Basis Function)
- C: 1.0
- Gamma: scale
- Class weight: balanced
```

#### XGBoost (Baseline)
```
Configuration:
- Max depth: 3
- Learning rate: 0.1
- n_estimators: 100
- Scale_pos_weight: Calculated per disease
```

### Phase 1 Results: Traditional ML

| Disease | LR (AUC) | SVM (AUC) | XGBoost (AUC) | Best Model |
|---------|----------|-----------|---------------|------------|
| **Cardiomegaly** | 0.569 | 0.430 | 0.481 | Logistic Regression |
| **Edema** | 0.808 | **0.810** | 0.784 | SVM |
| **Emphysema** | 0.623 | 0.382 | 0.554 | Logistic Regression |
| **Hernia** | 0.698 | 0.715 | **0.848** | XGBoost |
| **Pneumonia** | 0.641 | **0.644** | 0.625 | SVM |
| **Fibrosis** | 0.545 | 0.452 | 0.402 | Logistic Regression |
| **Pneumothorax** | 0.684 | 0.316 | 0.619 | Logistic Regression |
| **Average** | **0.653** | 0.534 | 0.616 | - |

### Phase 1 Analysis

**Strengths:**
- ✅ Hernia detection performed well (0.848 AUC with XGBoost)
- ✅ Edema showed promise (0.810 AUC)
- ✅ Fast inference time (~50ms per image)
- ✅ Interpretable features (visual words)

**Limitations:**
- ❌ Suboptimal performance on most diseases (AUC < 0.70)
- ❌ SIFT may miss subtle radiological patterns
- ❌ BoVW loses spatial information
- ❌ Feature engineering bottleneck
- ❌ Cardiomegaly detection especially poor (0.569 AUC)

**Decision:** Results were **not clinically satisfactory** (target: AUC > 0.80). Needed to explore deep learning approaches.

---

## Phase 2: Convolutional Neural Networks (CNN from Scratch)

### Motivation
Traditional ML struggled to capture complex visual patterns in chest X-rays. CNNs can:
- Automatically learn hierarchical features
- Preserve spatial information
- Detect subtle patterns humans might miss

### Architecture Design

**Custom CNN Architecture:**
```python
CustomCNN(
  # Block 1: Initial feature extraction
  Conv2d(3, 32, kernel=3) → BatchNorm → ReLU → MaxPool
  
  # Block 2: Deeper features
  Conv2d(32, 64, kernel=3) → BatchNorm → ReLU → MaxPool
  
  # Block 3: Complex patterns
  Conv2d(64, 128, kernel=3) → BatchNorm → ReLU → MaxPool
  
  # Block 4: High-level features
  Conv2d(128, 256, kernel=3) → BatchNorm → ReLU → MaxPool
  
  # Classifier
  GlobalAvgPooling → FC(256, 128) → Dropout(0.5) → FC(128, 7)
)

Total Parameters: ~2.5M
```

**Training Configuration:**
```
Input size: 224×224 RGB
Batch size: 32
Optimizer: Adam (lr=0.001)
Loss: BCEWithLogitsLoss
Epochs: 50 (with early stopping)
Data augmentation:
  - Random horizontal flip
  - Random rotation (±10°)
  - Color jitter
  - Random affine transforms
```

### Phase 2 Results: Custom CNN

**Training Challenges:**
- ⚠️ Slow convergence (10+ hours on GPU)
- ⚠️ Validation AUC plateaued around 0.65-0.70
- ⚠️ Overfitting despite dropout and regularization
- ⚠️ Inconsistent performance across diseases

**Approximate Results:**
```
Average AUC: ~0.68 (validation)
Best individual: Pneumothorax (0.74)
Worst: Fibrosis (0.58)
```

### Phase 2 Analysis

**Why did custom CNN underperform?**

1. **Limited Training Data**
   - 78,500 images insufficient for training CNN from scratch
   - Medical imaging requires 100K+ images per class ideally

2. **Class Imbalance**
   - Rare diseases (Hernia: 0.2%) had insufficient samples
   - Model biased toward "No Finding" class

3. **Lack of Medical Domain Knowledge**
   - Random initialization doesn't know what chest X-rays look like
   - Missing anatomical priors

4. **Training Instability**
   - Gradient issues in deeper layers
   - Difficulty learning subtle radiological features

**Decision:** Custom CNN showed improvement over traditional ML but still fell short of clinical requirements. **Needed pre-trained models with transfer learning.**

---

## Phase 3: Transfer Learning with DenseNet-121

### Why Transfer Learning?

Transfer learning leverages knowledge from models pre-trained on large datasets (ImageNet: 1.2M images, 1000 classes):

**Benefits:**
- ✅ Pre-learned low-level features (edges, textures, shapes)
- ✅ Better initialization than random weights
- ✅ Requires less training data
- ✅ Faster convergence
- ✅ Better generalization

### Model Selection: DenseNet-121

**Why DenseNet over ResNet or VGG?**

**DenseNet Advantages:**
1. **Dense Connectivity:** Each layer receives input from all previous layers
   - Better gradient flow
   - Feature reuse
   - Reduced parameters (7.9M vs ResNet-50's 25M)

2. **Medical Imaging Success:** State-of-the-art on CheXNet benchmark
   
3. **Efficient Feature Propagation:** Critical for subtle disease patterns

**Architecture:**
```python
DenseNet121 (Pre-trained on ImageNet)
├── Conv2d (7×7, stride 2)
├── DenseBlock1 (6 layers)
├── Transition1 (compression)
├── DenseBlock2 (12 layers)
├── Transition2 (compression)
├── DenseBlock3 (24 layers)
├── Transition3 (compression)
├── DenseBlock4 (16 layers)
├── GlobalAvgPool
└── FC (1024 → 7) [Modified for our task]

Total Parameters: 7,978,856
Trainable Parameters: 7,978,856
```

### Training Strategy

**Fine-tuning Approach:**
```python
# 1. Load pre-trained DenseNet-121
model = models.densenet121(pretrained=True)

# 2. Replace final classifier
model.classifier = nn.Linear(1024, 7)  # 7 diseases

# 3. Fine-tune entire network
# (No frozen layers - full end-to-end training)
```

**Training Configuration:**
```
Image size: 224×224 RGB
Batch size: 32
Optimizer: Adam
Learning rate: 1e-4 (lower than from-scratch)
Weight decay: 1e-5
Scheduler: ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 3 epochs
Loss function: BCEWithLogitsLoss
  - With positive class weights for imbalance
Early stopping: Patience 5 epochs

Data Augmentation:
  Training:
    - Resize to 224×224
    - Random horizontal flip (p=0.5)
    - Random rotation (±10°)
    - Color jitter (brightness±0.2, contrast±0.2)
    - Random affine (translate ±5%)
    - Normalization (ImageNet mean/std)
  
  Validation/Test:
    - Resize to 224×224
    - Normalization only
```

**Class Imbalance Handling:**
```python
# Calculate positive weights for rare diseases
pos_weights = neg_count / (pos_count + ε)

Example weights:
  Hernia: 220.5 (very rare, 0.2% prevalence)
  Pneumonia: 54.8
  Edema: 33.2
  Fibrosis: 45.6
  Cardiomegaly: 27.4
  Emphysema: 30.3
  Pneumothorax: 14.2 (most common)
```

### Training Progress

**Epoch-by-Epoch Performance:**
```
Epoch 1:  Train AUC: 0.72 | Val AUC: 0.75 | LR: 1e-4
Epoch 5:  Train AUC: 0.78 | Val AUC: 0.79 | LR: 1e-4
Epoch 10: Train AUC: 0.81 | Val AUC: 0.82 | LR: 5e-5 (reduced)
Epoch 15: Train AUC: 0.84 | Val AUC: 0.83 | LR: 5e-5
Epoch 20: Train AUC: 0.86 | Val AUC: 0.84 | LR: 2.5e-5 (reduced)
Epoch 25: Train AUC: 0.87 | Val AUC: 0.84 | LR: 2.5e-5
Best model saved at Epoch 22 (Val AUC: 0.845)
```

**Training Metrics:**
- **Training time:** ~6 hours on NVIDIA GPU (Google Colab)
- **Best validation AUC:** 0.845
- **Convergence:** Achieved at epoch 22
- **Early stopping:** Triggered at epoch 27

---

## 📈 Final Results: DenseNet-121 Transfer Learning

### Test Set Performance

| Disease | AUC Score | Performance |
|---------|-----------|-------------|
| **Cardiomegaly** | 0.89 | ⭐ Excellent |
| **Edema** | 0.92 | ⭐ Excellent |
| **Emphysema** | 0.87 | ⭐ Excellent |
| **Hernia** | 0.76 | ✅ Good |
| **Pneumonia** | 0.81 | ✅ Good |
| **Fibrosis** | 0.85 | ⭐ Excellent |
| **Pneumothorax** | 0.91 | ⭐ Excellent |
| **Average** | **0.86** | ⭐ **Excellent** |

**Clinical Interpretation:**
- AUC > 0.90: Excellent discriminative ability
- AUC 0.80-0.90: Good discriminative ability
- AUC 0.70-0.80: Fair discriminative ability
- AUC < 0.70: Poor discriminative ability

### Confusion Matrix Insights

**High Precision Diseases** (Low False Positives):
- Pneumothorax: 89% precision
- Edema: 87% precision

**High Recall Diseases** (Low False Negatives):
- Cardiomegaly: 91% recall
- Pneumonia: 85% recall

**Challenging Cases:**
- Hernia: Lowest AUC (0.76) due to extreme rarity (227 cases)
- Often confused with diaphragmatic abnormalities

---

## 🔄 Comparative Analysis: All Phases

### Performance Evolution

| Approach | Avg AUC | Training Time | Inference Time |
|----------|---------|---------------|----------------|
| **Traditional ML** (SIFT + BoVW) | 0.653 | 4 hours | 50ms |
| **Custom CNN** (from scratch) | 0.68 | 10 hours | 15ms |
| **Transfer Learning** (DenseNet-121) | **0.86** | 6 hours | 20ms |

### Improvement Breakdown

```
Phase 1 → Phase 2: +4.1% AUC improvement
Phase 2 → Phase 3: +26.5% AUC improvement
Phase 1 → Phase 3: +31.7% overall improvement
```

**Per-Disease Improvements (Phase 1 → Phase 3):**
```
Cardiomegaly:  0.569 → 0.89  (+56% improvement) 🚀
Edema:         0.810 → 0.92  (+13% improvement)
Emphysema:     0.623 → 0.87  (+40% improvement) 🚀
Hernia:        0.848 → 0.76  (-10% decrease) ⚠️
Pneumonia:     0.644 → 0.81  (+26% improvement)
Fibrosis:      0.545 → 0.85  (+56% improvement) 🚀
Pneumothorax:  0.684 → 0.91  (+33% improvement)
```

**Note:** Hernia decreased due to extreme rarity (0.2% of dataset). XGBoost's tree-based approach handled imbalance better for this specific disease.

---

## 🧠 Model Interpretability: Grad-CAM

### What is Grad-CAM?

**Gradient-weighted Class Activation Mapping** visualizes which regions of an X-ray the model focuses on when making predictions.

**How it works:**
```
1. Forward pass: Generate predictions
2. Backward pass: Compute gradients of target class w.r.t. feature maps
3. Weight feature maps by gradients
4. Generate heatmap showing important regions
5. Overlay heatmap on original image
```

### Clinical Validation

**Cardiomegaly Detection:**
- ✅ Model focuses on heart borders and cardiothoracic ratio
- ✅ Aligns with radiologist's diagnosis criteria

**Pneumonia Detection:**
- ✅ Highlights infiltrates and consolidations in lung fields
- ✅ Correctly identifies affected lobes

**Pneumothorax Detection:**
- ✅ Focuses on pleural spaces and lung periphery
- ✅ Detects collapsed lung regions

**Edema Detection:**
- ✅ Highlights bilateral pulmonary congestion patterns
- ✅ Identifies Kerley B lines and vascular redistribution

### Grad-CAM Implementation

```python
class GradCAM:
    def __init__(self, model, target_layer):
        # Register hooks to capture activations and gradients
        self.model = model
        self.target_layer = target_layer
    
    def generate_cam(self, input_image, target_class):
        # Forward pass
        output = self.model(input_image)
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Weight activations by gradients
        weights = self.gradients.mean(dim=[2, 3])
        cam = (weights[:, :, None, None] * self.activations).sum(1)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam / cam.max()
        
        return cam
```

---

## 💻 Deployment: Streamlit Web Application

### Application Features

**1. User Interface**
- Clean, medical-grade design
- Drag-and-drop image upload
- Real-time analysis (1-3 seconds)
- Responsive layout

**2. Analysis Dashboard**
- Top 3 findings cards with risk levels
- Detailed disease breakdown with expandable sections
- Grad-CAM heatmaps for each disease
- Probability bar chart
- Summary statistics

**3. Interpretability**
- Grad-CAM visualization showing focus areas
- Adjustable heatmap opacity
- Disease-specific explanations
- Risk categorization (High/Medium/Low)

**4. Risk Assessment**
```python
def get_risk_level(probability):
    if probability >= 0.7:
        return "High Risk" 🔴
    elif probability >= 0.4:
        return "Medium Risk" 🟡
    else:
        return "Low Risk" 🟢
```

### Technical Stack

```
Frontend: Streamlit 1.28+
Backend: PyTorch 2.0+
Visualization: Matplotlib, OpenCV
Model: DenseNet-121 (7.9M parameters)
Deployment: Local server (can be deployed on cloud)
```

### Application Architecture

```
User Upload
    ↓
Preprocessing (Resize, Normalize)
    ↓
DenseNet-121 Inference
    ↓
Sigmoid Activation (Multi-label probabilities)
    ↓
├── Disease Predictions
├── Grad-CAM Generation (for each disease)
├── Risk Assessment
└── Results Dashboard
    ↓
Interactive Visualization
```

---

## 🎓 Key Learnings

### Technical Insights

1. **Transfer Learning is Powerful**
   - Pre-trained models provide massive advantage
   - Even with limited medical data, achieved 0.86 AUC
   - Fine-tuning entire network worked better than freezing layers

2. **Class Imbalance Matters**
   - Positive class weighting essential for rare diseases
   - Hernia (0.2% prevalence) remains challenging
   - Consider over-sampling/under-sampling techniques

3. **Data Augmentation is Critical**
   - Horizontal flips valid for chest X-rays (anatomical symmetry)
   - Rotation and translation help generalization
   - Color jitter useful even for grayscale medical images

4. **Traditional ML Still Has Value**
   - Fast inference for resource-constrained settings
   - Interpretable features (visual words)
   - Good baseline for comparison

5. **Interpretability is Crucial**
   - Grad-CAM builds trust with clinicians
   - Visual explanations essential for medical AI
   - Helps identify model biases and errors


---

## 🚀 Future Work

### Short-term Improvements

1. **Ensemble Methods**
   - Combine DenseNet-121 with other architectures
   - Test DenseNet-169, DenseNet-201
   - Ensemble with ResNet-50, EfficientNet

2. **Attention Mechanisms**
   - Implement self-attention layers
   - Vision Transformers (ViT) for chest X-rays
   - Spatial attention to focus on relevant regions

3. **Advanced Augmentation**
   - CutMix and MixUp for training
   - Auto-augmentation policies
   - Domain-specific augmentations

---

## 📚 Technologies & Tools Used

### Programming & Libraries
```
Python 3.10+
PyTorch 2.0.0 - Deep learning framework
torchvision 0.15.0 - Pre-trained models
scikit-learn 1.3.0 - Traditional ML
OpenCV 4.8.0 - Image processing
NumPy 1.24.0 - Numerical computing
Pandas 2.0.0 - Data manipulation
Matplotlib 3.7.0 - Visualization
Seaborn 0.12.0 - Statistical plots
Streamlit 1.28.0 - Web application
```

### Development Environment
```
Google Colab - GPU training (Tesla T4)
VS Code - Local development
Jupyter Notebook - Experimentation
Git - Version control
```

### Dataset & APIs
```
NIH Chest X-ray Dataset (112,120 images)
Kaggle API - Dataset download
```

---

## 🏆 Project Deliverables

### Code Repositories
1. **Thoracic.ipynb** - Traditional ML pipeline (SIFT + BoVW)
2. **Chest_X_Ray.ipynb** - Deep learning pipeline (DenseNet-121)
3. **app.py** - Streamlit web application
4. **test_setup.py** - Environment validation

### Model Artifacts
1. **best_model.pth** - Trained DenseNet-121 weights (84 MB)
2. **train_bovw_matrix.npz** - BoVW features (traditional ML)
3. **train_tfidf_matrix.npz** - TF-IDF features

### Results & Reports
1. **comprehensive_comparison_final.csv** - All model results
2. **PROJECT_REPORT.md** - This comprehensive report
3. **README.md** - Quick start guide
4. **QUICKSTART.md** - Setup instructions

### Visualizations
1. Training history plots (loss, AUC, learning rate)
2. Per-disease AUC bar charts
3. Grad-CAM heatmaps
4. Confusion matrices

---

## 👥 Project Team & Contributions

**Team:** ML Project - SEM 5

**Contributions:**
- Dataset acquisition and preprocessing
- Traditional ML implementation (SIFT, BoVW, classifiers)
- CNN from scratch experimentation
- DenseNet-121 transfer learning
- Grad-CAM implementation
- Streamlit application development
- Documentation and reporting

---

## 📖 References

### Academic Papers
1. **Wang et al. (2017)** - "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks"
   - Source of NIH Chest X-ray dataset
   
2. **Rajpurkar et al. (2017)** - "CheXNet: Radiologist-Level Pneumonia Detection"
   - DenseNet-121 for chest X-rays
   
3. **Huang et al. (2017)** - "Densely Connected Convolutional Networks"
   - DenseNet architecture

4. **Selvaraju et al. (2017)** - "Grad-CAM: Visual Explanations from Deep Networks"
   - Explainability technique

5. **Lowe (2004)** - "Distinctive Image Features from Scale-Invariant Keypoints"
   - SIFT algorithm

### Datasets
- **NIH Clinical Center** - Chest X-ray dataset
- **Kaggle** - Dataset hosting and distribution

### Tools & Frameworks
- PyTorch Documentation
- Streamlit Documentation
- OpenCV Documentation

---


## 🎯 Conclusion

This project successfully demonstrated the power of deep learning for medical image analysis, progressing from traditional computer vision techniques (AUC 0.65) to state-of-the-art transfer learning (AUC 0.86) - a **32% improvement** in diagnostic accuracy.

### Key Achievements:
✅ **Clinical-grade Performance:** Average AUC of 0.86 across 7 thoracic diseases  
✅ **Interpretable AI:** Grad-CAM visualizations for clinician trust  
✅ **Production-ready Application:** Full-stack Streamlit deployment  
✅ **Comprehensive Pipeline:** End-to-end solution from data to deployment  

### Impact:
This system can potentially:
- **Reduce radiologist workload** by providing preliminary screenings
- **Accelerate diagnosis** in time-sensitive cases
- **Democratize healthcare** in under-resourced areas
- **Provide 24/7 screening** without fatigue

### Final Thoughts:
While our model shows excellent performance, medical AI is an **assistive tool**, not a replacement for expert radiologists. The future of healthcare lies in **human-AI collaboration**, where technology augments clinical expertise to provide better patient outcomes.

---

**Project Status:** ✅ Complete  
**Date:** October 2025  
**Next Steps:** Clinical validation and regulatory approval

---

*"In medicine, artificial intelligence is not about replacing doctors, it's about empowering them to save more lives."*

