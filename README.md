# MSARGAN - Agricultural Satellite Image Enhancement

🛰️ AI-powered system that enhances low-resolution satellite imagery for agricultural monitoring using Generative Adversarial Networks (GANs).

## 🌟 Features

- **4x Super-Resolution**: Transforms 64×64 images to high-quality 256×256 outputs
- **Quality Metrics**: Provides PSNR and SSIM measurements
- **NDVI Analysis**: Calculates Normalized Difference Vegetation Index for crop health
- **Real-time Processing**: Fast inference using PyTorch
- **Professional Web UI**: Modern, responsive interface with Flask

## 🚀 Live Demo

**Try it here:** [https://rjspark-msargan-app.hf.space](https://rjspark-msargan-app.hf.space)
<img width="947" height="444" alt="Image" src="https://github.com/user-attachments/assets/3db89094-1237-42ea-a722-c593046b9e4c" />
<img width="711" height="433" alt="Image" src="https://github.com/user-attachments/assets/65884142-cb1d-4af1-8b21-1772c59ccdce" />
<img width="783" height="425" alt="Image" src="https://github.com/user-attachments/assets/e7d2d0a9-fbac-4d76-ad72-2c5022e7c73b" />
<img width="765" height="440" alt="Image" src="https://github.com/user-attachments/assets/9f637939-d36b-4f3d-b9b2-d0b156f248cc" />

## 🏗️ Architecture

- **Generator Network**: 8 Residual Blocks with PReLU activation
- **Training Strategy**: Adversarial + Perceptual + Pixel-wise losses
- **Framework**: PyTorch with VGG19 feature extraction
- **Upsampling**: Pixel Shuffle (2x → 2x for 4x total)

## 🛠️ Tech Stack

- **Backend**: Flask, PyTorch, Python
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Docker, Hugging Face Spaces
- **Image Processing**: PIL, NumPy, scikit-image, matplotlib

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/rjspark/msargan-satellite-enhancement.git
cd msargan-satellite-enhancement
```

### 2. Download the trained model
Download `generator_latest.pth` from [Hugging Face](https://huggingface.co/spaces/rjspark/msargan-app/blob/main/generator_latest.pth) and place it in the root directory.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

Visit: `http://localhost:7860`

## 🐳 Docker Deployment
```bash
docker build -t msargan-app .
docker run -p 7860:7860 msargan-app
```

## 📊 Model Performance

- **Input**: 64×64 RGB satellite images
- **Output**: 256×256 enhanced images
- **Metrics**: PSNR and SSIM for quality assessment
- **Additional**: NDVI calculation for vegetation analysis

## 🌾 Use Cases

- Agricultural crop health monitoring
- Satellite image enhancement for analysis
- Vegetation mapping and NDVI calculations
- Remote sensing applications
- Precision agriculture

## 📁 Project Structure
```
msargan-satellite-enhancement/
├── app.py                  # Flask backend
├── generator_latest.pth    # Trained model (download separately)
├── requirements.txt        # Dependencies
├── Dockerfile             # Docker configuration
├── templates/
│   └── index.html         # Frontend HTML
└── static/
    ├── style.css          # Styling
    └── script.js          # JavaScript
```

## 🎯 Future Improvements

- [ ] Batch processing for multiple images
- [ ] GPU acceleration option
- [ ] Additional vegetation indices (EVI, SAVI)
- [ ] Comparison with other SR methods
- [ ] Mobile app version

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

**rjspark**
- Hugging Face: [@rjspark](https://huggingface.co/rjspark)
- Live Demo: [MSARGAN App](https://rjspark-msargan-app.hf.space)

## 🙏 Acknowledgments

- Built using PyTorch deep learning framework
- Deployed on Hugging Face Spaces
- Inspired by SRGAN architecture

## 📞 Contact

For questions or collaboration opportunities, reach out via GitHub issues.

---

⭐ **Star this repo if you find it useful!**
