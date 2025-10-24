// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsContainer = document.getElementById('resultsContainer');
const resetButton = document.getElementById('resetButton');

// Image display elements
const inputImage = document.getElementById('inputImage');
const enhancedImage = document.getElementById('enhancedImage');
const ndviImage = document.getElementById('ndviImage');

// Metrics display elements
const psnrValue = document.getElementById('psnrValue');
const ssimValue = document.getElementById('ssimValue');
const ndviValue = document.getElementById('ndviValue');
const ndviInterpretation = document.getElementById('ndviInterpretation');

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Upload area click handler
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary-color)';
    uploadArea.style.background = 'rgba(37, 99, 235, 0.05)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--border-color)';
    uploadArea.style.background = 'white';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--border-color)';
    uploadArea.style.background = 'white';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File input change handler
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Reset button handler
resetButton.addEventListener('click', () => {
    resetDemo();
});

// Handle file upload and processing
function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload a valid image file');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }

    // Show loading state
    uploadArea.style.display = 'none';
    loadingSpinner.style.display = 'block';
    resultsContainer.style.display = 'none';

    // Create FormData and send request
    const formData = new FormData();
    formData.append('image', file);

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Processing failed');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while processing the image');
        resetDemo();
    });
}

// Display processing results
function displayResults(data) {
    // Hide loading spinner
    loadingSpinner.style.display = 'none';

    // Set images
    inputImage.src = data.lr_image;
    enhancedImage.src = data.sr_image;
    ndviImage.src = data.ndvi_image;

    // Set metrics with animation
    animateValue(psnrValue, 0, data.psnr, 1000, 2);
    animateValue(ssimValue, 0, data.ssim, 1000, 4);
    animateValue(ndviValue, 0, data.avg_ndvi, 1000, 3);

    // Set NDVI interpretation
    ndviInterpretation.textContent = data.ndvi_interpretation;

    // Set color based on NDVI value
    const ndviVal = data.avg_ndvi;
    let color;
    if (ndviVal < 0) color = '#3b82f6';
    else if (ndviVal < 0.2) color = '#f59e0b';
    else if (ndviVal < 0.5) color = '#eab308';
    else color = '#22c55e';
    
    ndviValue.style.color = color;

    // Show results with animation
    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Animate number counting
function animateValue(element, start, end, duration, decimals) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = current.toFixed(decimals);
    }, 16);
}

// Show error message
function showError(message) {
    alert('Error: ' + message);
}

// Reset demo to initial state
function resetDemo() {
    uploadArea.style.display = 'block';
    loadingSpinner.style.display = 'none';
    resultsContainer.style.display = 'none';
    imageInput.value = '';

    // Scroll to upload area
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Add parallax effect to hero background
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const heroBackground = document.querySelector('.hero-background');
    if (heroBackground) {
        heroBackground.style.transform = `translateY(${scrolled * 0.5}px)`;
    }
});

// Add fade-in animation on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all cards and sections
document.querySelectorAll('.about-card, .process-step, .tech-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
    observer.observe(el);
});

// Navbar scroll effect
let lastScroll = 0;
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    const currentScroll = window.pageYOffset;

    if (currentScroll > 100) {
        navbar.style.boxShadow = '0 2px 30px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.05)';
    }

    lastScroll = currentScroll;
});

// Add loading animation
document.addEventListener('DOMContentLoaded', () => {
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s ease-in';
        document.body.style.opacity = '1';
    }, 100);
});
