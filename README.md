---
title: MSARGAN - Agricultural Satellite Image Enhancement
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# MSARGAN - Multi-Scale Attention Super-Resolution GAN

## 🌾 Agricultural Satellite Image Enhancement System

Transform low-resolution 64×64 satellite imagery into high-quality 256×256 outputs with AI-powered enhancement and agricultural analysis.

### 🚀 Features

- **4x Super-Resolution**: Enhances 64×64 images to 256×256 with improved texture and structural clarity
- **Quality Metrics**: Provides PSNR and SSIM measurements for quantifiable image quality
- **NDVI Analysis**: Calculates Normalized Difference Vegetation Index for agricultural monitoring
- **Real-time Processing**: Fast inference using PyTorch and GAN architecture
- **Professional UI**: Modern, responsive web interface with smooth animations

### 🎯 How to Use

1. **Upload** a 64×64 satellite image (or any image - it will be automatically resized)
2. **Wait** for processing (typically 10-30 seconds)
3. **View Results**:
   - Original low-resolution input
   - Enhanced 256×256 super-resolution image with PSNR/SSIM metrics
   - NDVI vegetation analysis with color-coded visualization

### 🏗️ Architecture

- **Generator Network**: 
  - 8 Residual Blocks with skip connections
  - PReLU activation functions
  - Batch Normalization
  - Pixel Shuffle upsampling (4x total resolution boost)

- **Training Strategy**:
  - Adversarial Loss (GAN)
  - Perceptual Loss (VGG19 features)
  - Pixel-wise MSE Loss
  - Adam Optimizer

### 📊 Performance

The model provides measurable quality improvements:
- **PSNR** (Peak Signal-to-Noise Ratio): Measures reconstruction quality
- **SSIM** (Structural Similarity Index): Measures perceptual quality
- **NDVI**: Agricultural vegetation health indicator

### 🛠️ Built With

- PyTorch - Deep learning framework
- Flask - Web framework
- HTML/CSS/JavaScript - Frontend
- Docker - Containerization

### 📝 Use Cases

- Agricultural monitoring and crop health assessment
- Satellite image enhancement for analysis
- Vegetation mapping and NDVI calculations
- Remote sensing applications
- Precision agriculture

### 👨‍💻 Developer

Created as a machine learning project demonstrating:
- Deep learning model deployment
- Full-stack web development
- Agricultural AI applications
- Cloud deployment with Docker

### 📄 License

MIT License - Feel free to use for educational and research purposes.

---

**Note**: Processing is performed on CPU in this Space, so please allow 10-30 seconds for image enhancement.
