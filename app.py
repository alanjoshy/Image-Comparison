import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_details(image_path):
    img = Image.open(image_path)
    return {
        'format': img.format,
        'size': img.size,
        'mode': img.mode
    }

def calculate_ssim(img1, img2):
    """Calculate structural similarity between two images."""
    return ssim(img1, img2, channel_axis=2, win_size=3, data_range=1.0)  # data_range=1.0 since images are normalized to [0,1]

def calculate_histogram_similarity(img1, img2):
    """Calculate histogram similarity between two images."""
    # Calculate histograms for each color channel
    hist_similarity = 0
    for i in range(3):  # For each color channel
        hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Compare histograms
        hist_similarity += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return (hist_similarity / 3 + 1) * 50  # Average and convert to percentage

def calculate_pixel_difference(img1, img2):
    """Calculate pixel-wise difference between images."""
    diff = cv2.absdiff(img1, img2)
    diff_percentage = 100 - (np.sum(diff) / (diff.size * 255) * 100)
    return diff_percentage

def compare_images(image1_path, image2_path):
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Store original dimensions
    orig_dim1 = img1.shape
    orig_dim2 = img2.shape
    
    # Convert to same size for comparison
    img1_resized = cv2.resize(img1, (500, 500))
    img2_resized = cv2.resize(img2, (500, 500))
    
    # Convert to float32 for SSIM calculation
    img1_float = img1_resized.astype(np.float32) / 255.0
    img2_float = img2_resized.astype(np.float32) / 255.0
    
    try:
        # Calculate SSIM (structural similarity)
        ssim_score = calculate_ssim(img1_float, img2_float)
        ssim_percentage = (ssim_score + 1) * 50
        
        # Calculate histogram similarity
        hist_similarity = calculate_histogram_similarity(img1_resized, img2_resized)
        
        # Calculate pixel difference
        pixel_diff = calculate_pixel_difference(img1_resized, img2_resized)
        
        # Calculate perceptual hash similarity
        hash1 = imagehash.average_hash(Image.open(image1_path))
        hash2 = imagehash.average_hash(Image.open(image2_path))
        hash_similarity = 100 - (hash1 - hash2) * 100 / len(hash1.hash) ** 2
        
        # Get image details
        details1 = get_image_details(image1_path)
        details2 = get_image_details(image2_path)
        
        # Calculate weighted similarity score
        weights = {
            'ssim': 0.4,        # Structural similarity (most important)
            'pixel': 0.3,       # Pixel difference
            'hist': 0.2,        # Color distribution
            'hash': 0.1         # Perceptual hash (least important)
        }
        
        overall_similarity = (
            ssim_percentage * weights['ssim'] +
            pixel_diff * weights['pixel'] +
            hist_similarity * weights['hist'] +
            hash_similarity * weights['hash']
        )
        
        # Prepare detailed analysis
        analysis = {
            'overall_similarity': float(round(overall_similarity, 2)),
            'structural_similarity': float(round(ssim_percentage, 2)),
            'pixel_similarity': float(round(pixel_diff, 2)),
            'histogram_similarity': float(round(hist_similarity, 2)),
            'hash_similarity': float(round(hash_similarity, 2)),
            'dimension_match': bool(orig_dim1 == orig_dim2),
            'original_dimensions': {
                'image1': f"{orig_dim1[1]}x{orig_dim1[0]}",
                'image2': f"{orig_dim2[1]}x{orig_dim2[0]}"
            },
            'format_details': {
                'image1': details1,
                'image2': details2
            }
        }
        
        # Determine verdict based on similarity scores
        if overall_similarity >= 98:
            analysis['verdict'] = "The images are identical"
        elif overall_similarity >= 95:
            analysis['verdict'] = "The images are nearly identical"
        elif overall_similarity >= 90:
            analysis['verdict'] = "The images are very similar"
        elif overall_similarity >= 80:
            analysis['verdict'] = "The images have significant similarities"
        elif overall_similarity >= 60:
            analysis['verdict'] = "The images have moderate similarities"
        else:
            analysis['verdict'] = "The images are significantly different"
        
        # Add detailed differences
        differences = []
        if not analysis['dimension_match']:
            differences.append("Image dimensions don't match")
        if details1['format'] != details2['format']:
            differences.append("Image formats are different")
        if details1['mode'] != details2['mode']:
            differences.append("Color modes are different")
        if abs(hist_similarity - pixel_diff) > 20:
            differences.append("Significant color distribution differences detected")
        if ssim_percentage < 70:
            differences.append("Significant structural differences detected")
        
        analysis['differences'] = differences
        
        return analysis
        
    except Exception as e:
        raise Exception(f"Error during image comparison: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    image1 = request.files['image1']
    image2 = request.files['image2']
    
    if image1.filename == '' or image2.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not (image1 and allowed_file(image1.filename) and 
            image2 and allowed_file(image2.filename)):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save files with timestamp to avoid naming conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename1 = f"{timestamp}_1_{secure_filename(image1.filename)}"
        filename2 = f"{timestamp}_2_{secure_filename(image2.filename)}"
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        image1.save(filepath1)
        image2.save(filepath2)
        
        # Compare images and get detailed analysis
        analysis = compare_images(filepath1, filepath2)
        
        # Add image URLs to the response
        analysis['image1_url'] = f"/static/uploads/{filename1}"
        analysis['image2_url'] = f"/static/uploads/{filename2}"
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
