# Advanced Image Comparison Tool 🖼️

A sophisticated web application that performs detailed image similarity analysis using multiple comparison metrics. This tool provides comprehensive insights into how similar two images are, using various scientific approaches to measure similarity.

## 🌟 Features

- **Multiple Comparison Metrics**:
  - Structural Similarity Index (SSIM) - 40% weight
  - Pixel-wise Difference - 30% weight
  - Histogram Comparison - 20% weight
  - Perceptual Hash - 10% weight

- **Detailed Analysis**:
  - Overall similarity score
  - Individual metric scores
  - Dimension comparison
  - Format details
  - Comprehensive similarity verdict

- **User-Friendly Interface**:
  - Drag-and-drop image upload
  - Real-time analysis
  - Visual similarity indicators
  - Detailed results display

## 🛠️ Technology Stack

- **Backend**: Python/Flask
- **Frontend**: HTML, Tailwind CSS, Alpine.js
- **Image Processing**:
  - OpenCV (cv2)
  - NumPy
  - Pillow
  - scikit-image
  - ImageHash

## 📊 Similarity Verdicts

- **98-100%**: Identical images
- **95-98%**: Nearly identical (minor compression differences)
- **90-95%**: Very similar (slight modifications)
- **80-90%**: Significant similarities
- **60-80%**: Moderate similarities
- **<60%**: Significantly different

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alanjoshy/Image-Comparison.git
   cd Image-Comparison
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## 🔒 Security Features

- File type validation
- Maximum file size limit (16MB)
- Secure filename handling
- Temporary file cleanup

## 🎯 Use Cases

- Compare similar-looking images
- Detect duplicate images
- Verify image modifications
- Quality control in image processing
- Image authenticity verification

## 🔧 Technical Details

### Image Processing Pipeline

1. **Image Loading and Preprocessing**:
   - Format validation
   - Size normalization
   - Color space conversion

2. **Similarity Calculations**:
   - SSIM for structural similarity
   - Pixel-wise comparison
   - Color histogram analysis
   - Perceptual hash comparison

3. **Result Aggregation**:
   - Weighted scoring system
   - Detailed analysis generation
   - Verdict determination

## 📝 License

MIT License

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/alanjoshy/Image-Comparison/issues).

## ✨ Future Improvements

- Batch image comparison
- More comparison algorithms
- Downloadable comparison reports
- Image metadata analysis
- Support for more image formats
- Performance optimization for large images
