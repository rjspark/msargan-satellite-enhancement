from flask import Flask, render_template, request, jsonify, send_file
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import base64
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture (same as your training code)
class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, num_residuals=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock() for _ in range(num_residuals)])
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out = self.conv2(out)
        out = out1 + out
        out = self.upsample(out)
        out = self.conv3(out)
        return out

# Load the trained model
generator = Generator().to(device)
try:
    generator.load_state_dict(torch.load('generator_latest.pth', map_location=device))
    generator.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

# Image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

def calculate_ndvi(image_array):
    """
    Calculate NDVI from RGB image
    NDVI = (NIR - Red) / (NIR + Red)
    For RGB images, we approximate: NIR ≈ Green, Red = Red
    """
    # Ensure image is in the right format [H, W, C]
    if len(image_array.shape) == 3 and image_array.shape[0] == 3:
        image_array = np.transpose(image_array, (1, 2, 0))
    
    red = image_array[:, :, 0].astype(float)
    green = image_array[:, :, 1].astype(float)
    
    # Avoid division by zero
    denominator = green + red
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    ndvi = (green - red) / denominator
    ndvi = np.clip(ndvi, -1, 1)
    
    return ndvi

def interpret_ndvi(ndvi_value):
    """Interpret NDVI value"""
    if ndvi_value < 0:
        return "Water or non-vegetated surface"
    elif ndvi_value < 0.1:
        return "Barren rock, sand, or snow"
    elif ndvi_value < 0.2:
        return "Sparse vegetation or dry region"
    elif ndvi_value < 0.5:
        return "Moderate vegetation (grassland/shrub)"
    elif ndvi_value < 0.7:
        return "Dense vegetation (forest/crops)"
    else:
        return "Very dense, healthy vegetation"

def create_ndvi_image(ndvi_array):
    """Create a professional NDVI visualization with matplotlib-style colormap"""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create figure with specific size to match 256x256 output
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    
    # Create custom colormap similar to RdYlGn (Red-Yellow-Green)
    colors = ['#8B0000', '#FF0000', '#FF4500', '#FFA500', '#FFFF00', 
              '#9ACD32', '#00FF00', '#008000', '#006400']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('ndvi', colors, N=n_bins)
    
    # Display NDVI with colorbar
    im = ax.imshow(ndvi_array, cmap=cmap, vmin=-1, vmax=1)
    ax.set_title('Approximate Vegetation Index', fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')  # Remove axes
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pseudo NDVI', rotation=270, labelpad=20, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor='white')
    buffer.seek(0)
    plt.close(fig)
    
    # Read back as PIL Image and convert to array
    ndvi_img = Image.open(buffer)
    ndvi_colored = np.array(ndvi_img.convert('RGB'))
    
    return ndvi_colored

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    # Ensure the array is in the correct format
    if len(image_array.shape) == 3:
        if image_array.shape[0] == 3:  # CHW format
            image_array = np.transpose(image_array, (1, 2, 0))
    
    # Convert to uint8 if needed
    if image_array.dtype != np.uint8:
        image_array = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    
    # Create PIL Image
    img = Image.fromarray(image_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process the image
        image = Image.open(file.stream).convert('RGB')
        
        # Resize to 64x64 if not already
        if image.size != (64, 64):
            image = image.resize((64, 64), Image.BICUBIC)
        
        # Convert to tensor
        lr_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate high-resolution image
        with torch.no_grad():
            sr_tensor = generator(lr_tensor)
        
        # Convert tensors to numpy arrays
        lr_np = lr_tensor.squeeze(0).cpu().numpy()
        sr_np = torch.clamp(sr_tensor.squeeze(0), 0, 1).cpu().numpy()
        
        # Calculate PSNR and SSIM (comparing SR image with upscaled LR as reference)
        lr_upscaled = np.array(image.resize((256, 256), Image.BICUBIC))
        lr_upscaled = np.transpose(lr_upscaled, (2, 0, 1)) / 255.0
        
        # Convert to HWC format for metrics
        sr_np_hwc = np.transpose(sr_np, (1, 2, 0))
        lr_upscaled_hwc = np.transpose(lr_upscaled, (1, 2, 0))
        
        psnr_value = psnr(lr_upscaled_hwc, sr_np_hwc, data_range=1.0)
        ssim_value = ssim(lr_upscaled_hwc, sr_np_hwc, 
                         data_range=1.0, channel_axis=2)
        
        # Calculate NDVI
        ndvi_array = calculate_ndvi(sr_np)
        avg_ndvi = float(np.mean(ndvi_array))
        ndvi_interpretation = interpret_ndvi(avg_ndvi)
        
        # Create NDVI visualization
        ndvi_colored = create_ndvi_image(ndvi_array)
        
        # Convert images to base64
        lr_base64 = image_to_base64(lr_np)
        sr_base64 = image_to_base64(sr_np)
        ndvi_base64 = image_to_base64(ndvi_colored)
        
        return jsonify({
            'success': True,
            'lr_image': lr_base64,
            'sr_image': sr_base64,
            'ndvi_image': ndvi_base64,
            'psnr': round(psnr_value, 2),
            'ssim': round(ssim_value, 4),
            'avg_ndvi': round(avg_ndvi, 3),
            'ndvi_interpretation': ndvi_interpretation
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Hugging Face Spaces runs on port 7860
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
