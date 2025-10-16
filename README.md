# 🏥 Chest X-Ray Disease Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)

> **Automated thoracic disease detection from chest X-rays using deep learning and transfer learning with DenseNet-121**

[🔗 GitHub Repository](https://github.com/MohammedMusharraf11/ML_Project_915_348) | [📄 Full Report](REPORT.md) 

---

## 📋 Project Overview

This project tackles the challenge of **automated thoracic disease detection** from chest X-ray images using AI. We developed a comprehensive machine learning pipeline that evolved from traditional computer vision techniques to state-of-the-art deep learning, achieving **clinically significant performance** with an average AUC of **0.86**.

### 🎯 What Does It Do?

Our AI system can detect **7 thoracic diseases** from a single chest X-ray:

1. 💓 **Cardiomegaly** - Enlarged heart
2. 💧 **Edema** - Fluid in lungs
3. 🫁 **Emphysema** - Lung tissue damage
4. ⚕️ **Hernia** - Organ displacement
5. 🦠 **Pneumonia** - Lung infection
6. 🔬 **Fibrosis** - Lung scarring
7. 💨 **Pneumothorax** - Collapsed lung

### 🚀 Our Journey

```
Phase 1: Traditional ML (SIFT + BoVW + TF-IDF)
├─ AUC: 0.653 ❌ Not clinically viable
│
Phase 2: Custom CNN (Built from scratch)
├─ AUC: ~0.68 ⚠️ Better, but insufficient
│
Phase 3: Transfer Learning (DenseNet-121) ⭐
└─ AUC: 0.86 ✅ CLINICALLY SIGNIFICANT!
```

**🎉 Result:** **32% improvement** from traditional ML to deep learning!

---

## 🏆 Key Achievements

- ✅ **0.86 Average AUC** across 7 diseases (clinical threshold: 0.80)
- ✅ **6 out of 7 diseases** exceed 0.80 AUC
- ✅ **Production-ready web app** with Grad-CAM interpretability
- ✅ **Real-time predictions** in 1-3 seconds
- ✅ **Explainable AI** using Grad-CAM heatmaps

---

## 📊 Performance Results

| Disease | Traditional ML | DenseNet-121 | Improvement |
|---------|----------------|--------------|-------------|
| Cardiomegaly | 0.569 | **0.89** ⭐ | +56% |
| Edema | 0.810 | **0.92** ⭐ | +13% |
| Emphysema | 0.623 | **0.87** ⭐ | +40% |
| Hernia | 0.848 | **0.76** ✓ | -10% |
| Pneumonia | 0.644 | **0.81** ⭐ | +26% |
| Fibrosis | 0.545 | **0.85** ⭐ | +56% |
| Pneumothorax | 0.684 | **0.91** ⭐ | +33% |
| **AVERAGE** | **0.653** | **0.86** | **+32%** |

⭐ = Excellent (AUC > 0.80) | ✓ = Good (AUC > 0.70)

---

## 🎨 Web Application Features

Our Streamlit application provides:

- 🖼️ **Drag-and-drop image upload**
- 🔬 **Real-time analysis** (1-3 seconds)
- 📊 **Interactive dashboard** with disease probabilities
- 🗺️ **Grad-CAM heatmaps** showing model focus areas
- 🚦 **Risk assessment** (High/Medium/Low)
- 📈 **Visual charts** and detailed breakdowns
- ⚙️ **Customizable settings** (threshold, opacity)

### Demo Screenshot

![App Demo](https://via.placeholder.com/800x450.png?text=Streamlit+App+Demo)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB+ RAM (8GB recommended)
- GPU optional (for faster inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MohammedMusharraf11/ML_Project_915_348.git
   cd ML_Project_915_348
   ```

2. **Download the trained model**
   
   ⚠️ **IMPORTANT:** Download the trained DenseNet-121 model (84 MB):
   
   📥 [**Download best_model.pth from Google Drive**](https://drive.google.com/file/d/1sCKoE1KhPfGt5yVNM3-SE-fLSmab0gBh/view?usp=sharing)
   
   After downloading, place `best_model.pth` in the `frontend` directory:
   ```
   frontend/
   ├── app.py
   ├── best_model.pth  ← Place the downloaded file here
   └── requirements.txt
   ```

3. **Install dependencies**
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

---

## 📁 Repository Structure

```
ML_Project_915_348/
├── README.md                          # This file
├── REPORT.md                          # Comprehensive project report
│
│
├── frontend/                          # Streamlit web application
│   ├── app.py                        # Main application
│   ├── requirements.txt              # Python dependencies
│   ├── best_model.pth                # Trained DenseNet-121 model (download separately)
│
├── Traditional_ML.ipynb                # Traditional ML experiments
├── Chest_Xray_DeepLearning_Colab.ipynb  # DenseNet-121 training
├── Final_Notebook.ipynb                  # Pre-trained model inference
```
---

## 💻 Usage

### Web Application

1. **Launch the app:**
   ```bash
   cd frontend
   streamlit run app.py
   ```

2. **Upload an X-ray image:**
   - Click the upload button or drag-and-drop
   - Supported formats: PNG, JPG, JPEG

3. **Analyze:**
   - Click "🔬 Analyze X-Ray" button
   - Wait 1-3 seconds for results

4. **Explore results:**
   - View top 3 findings
   - Check detailed disease probabilities
   - Examine Grad-CAM heatmaps
   - Adjust settings in sidebar

### Jupyter Notebooks

**Traditional ML Pipeline:**
```bash
jupyter notebook Traditonal_ML.ipynb
```

**Deep Learning Training:**
```bash
jupyter notebook Chest_Xray_DeepLearning_Colab.ipynb
```

**Pre-trained model available:** 
```bash
jupyter notebook Final_Notebook.ipynb
```

---

## 🧠 Technical Details

### Model Architecture

- **Base Model:** DenseNet-121 (pre-trained on ImageNet)
- **Parameters:** 7,978,856 (7.9M)
- **Input:** 224×224 RGB images
- **Output:** 7-class multi-label predictions
- **Framework:** PyTorch 2.0+

### Training Configuration

```python
Optimizer: Adam (lr=1e-4)
Loss: BCEWithLogitsLoss (with class weights)
Scheduler: ReduceLROnPlateau
Epochs: 25 (with early stopping)
Batch Size: 32
Augmentation: Horizontal flip, rotation, color jitter
```

### Dataset

- **Source:** NIH Chest X-ray Dataset
- **Total Images:** 112,120 frontal chest X-rays
- **Split:** 70% train / 15% validation / 15% test
- **Class Imbalance:** Handled with weighted loss

---

## 📊 Grad-CAM Interpretability

Our model uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to show which regions of the X-ray influenced the prediction. This builds trust with clinicians and helps identify model focus areas.

**Example:**
- ✅ Cardiomegaly: Model focuses on heart borders
- ✅ Pneumothorax: Model highlights pleural spaces
- ✅ Pneumonia: Model identifies lung infiltrates

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Core** | Python 3.10, PyTorch 2.0, torchvision |
| **ML/DL** | scikit-learn, NumPy, Pandas |
| **Computer Vision** | OpenCV, PIL/Pillow, SIFT (traditional ML) |
| **Web App** | Streamlit, Matplotlib, Seaborn |
| **Training** | Google Colab (Tesla T4 GPU) |
| **Dataset** | NIH Chest X-ray (via Kaggle) |

---

## 📈 Future Enhancements

### Short-term
- [ ] Ensemble multiple architectures (ResNet + EfficientNet)
- [ ] Support for DICOM format
- [ ] Batch processing for multiple images
- [ ] Export predictions to PDF report

### Long-term
- [ ] Multi-modal learning (X-ray + patient history)
- [ ] 3D analysis for CT scans
- [ ] Federated learning across hospitals
- [ ] Clinical trials and FDA approval

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚖️ Medical Disclaimer

⚠️ **IMPORTANT:** This tool is for **educational and research purposes only**.

- **NOT** a substitute for professional medical diagnosis
- Always consult qualified healthcare professionals
- Requires radiologist final review before clinical use
- Not approved by FDA or other regulatory bodies

The model's predictions are probabilistic and may contain errors.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Mohammed Musharraf** - [GitHub](https://github.com/MohammedMusharraf11)
- **Mohammed Shazi** - [GitHub](https://github.com/Mohammed-Shazi-Ul-Islam)

---

## 📚 Documentation

For detailed information, please refer to:

- [📄 Comprehensive Report](REPORT.md) - Full project documentation


---

## 🔗 Links

- **GitHub Repository:** [MohammedMusharraf11/ML_Project_915_348](https://github.com/MohammedMusharraf11/ML_Project_915_348)
- **Model Download:** [Google Drive (84 MB)](https://drive.google.com/file/d/1sCKoE1KhPfGt5yVNM3-SE-fLSmab0gBh/view?usp=sharing)
- **NIH Dataset:** [ChestX-ray8 Database](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)

---

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🌟 Acknowledgments

- **NIH Clinical Center** for providing the chest X-ray dataset
- **Stanford ML Group** for CheXNet research
- **PyTorch Team** for the deep learning framework
- **Streamlit** for the web application framework
- **Google Colab** for free GPU resources

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=MohammedMusharraf11/ML_Project_915_348&type=Date)](https://star-history.com/#MohammedMusharraf11/ML_Project_915_348&Date)

---

<div align="center">

**Built with ❤️ using PyTorch, DenseNet-121, and Streamlit**

*"AI in medicine is not about replacing doctors, it's about empowering them to save more lives."*

</div>
