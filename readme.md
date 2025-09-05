# 🚦 Traffic Light Detection System

A comprehensive real-time traffic light detection system that uses OpenCV and HSV color segmentation to detect Red, Yellow, and Green traffic lights from webcam feeds, video files, or static images. The system features both a command-line interface and a modern web-based Streamlit interface.

## ✨ Features

### Core Detection Capabilities

- **Multi-color detection**: Robust HSV-based detection for Red, Yellow, and Green traffic lights
- **Adaptive preprocessing**: Gray-world white balance, auto gamma correction, and CLAHE enhancement
- **Advanced filtering**: Morphology operations, contour filtering, and brightness/saturation gates
- **Smart tracking**: IoU-based track management with confidence smoothing and lock-on detection
- **False positive reduction**: Heuristics to distinguish traffic lights from vehicle lights
- **Dynamic scaling**: Automatically adjusts parameters based on input resolution

### User Interfaces

- **🌐 Web Interface**: Modern Streamlit-based web app with real-time processing
- **🖥️ Desktop UI**: OpenCV-based GUI for local desktop usage
- **⌨️ Command Line**: Direct CLI access for automation and scripting

### Input Support

- **📷 Webcam**: Real-time detection from camera feeds
- **📹 Video Files**: Process MP4, AVI, MOV, MKV, and other video formats
- **🖼️ Images**: Single image processing with JPG, PNG, BMP, TIFF support
- **📁 Sample Images**: Pre-loaded test images for demonstration

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

Launch the modern Streamlit web interface:

```bash
./run_streamlit.sh
```

Then open your browser to `http://localhost:8501` and enjoy the interactive interface!

### Option 2: Desktop UI

Run the OpenCV-based desktop interface:

```bash
./run_detector.sh
```

### Option 3: Command Line

For direct CLI usage:

```bash
# Webcam detection
python traffic_light_detector.py --source 0 --output

# Video file processing
python traffic_light_detector.py -s "/path/to/video.mp4" -o

# Image processing
python traffic_light_detector.py -s "/path/to/image.jpg" -i -o
```

## 📦 Installation

### Automated Setup (Recommended)

The provided shell scripts handle everything automatically:

```bash
# For web interface
./run_streamlit.sh

# For desktop interface
./run_detector.sh
```

### Manual Setup

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### Dependencies

- **opencv-python-headless** (≥ 4.8.0): Computer vision processing
- **numpy** (≥ 1.20.0): Numerical computations
- **streamlit** (≥ 1.28.0): Web interface framework
- **Pillow** (≥ 8.0.0): Image processing support

## 🎮 Usage Guide

### Web Interface Controls

- **Input Methods**: Upload images/videos, use webcam, or try sample images
- **Detection Settings**: Adjust confidence threshold and debug visualization
- **Real-time Stats**: View detection counts and performance metrics
- **Debug Mode**: Toggle color mask visualization for troubleshooting

### Desktop Interface Controls

- **Menu Navigation**: Use number keys (1-4) to select options
- **Video Controls**: Press `q` to quit, `d` to toggle debug masks
- **Output Options**: Choose whether to save processed videos/images

### Command Line Options

```bash
python traffic_light_detector.py [OPTIONS]

Options:
  -s, --source SOURCE    Camera index (0), video file path, or image file path
  -o, --output          Save processed video/image to file
  -i, --image           Process as image file instead of video/camera
  -h, --help            Show help message
```

## 📁 Project Structure

```
Traffic-Lights-Tracking-and-Color-Detection-OpenCV/
├── app.py                          # Streamlit web application
├── traffic_light_detector.py       # Core detection engine
├── simple_ui.py                    # OpenCV desktop interface
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages (Linux)
├── run_streamlit.sh               # Web app launcher script
├── run_detector.sh                # Desktop app launcher script
├── sample_images/                 # Test images
│   ├── sample_red_light.jpg
│   ├── sample_yellow_light.jpg
│   ├── sample_green_light.jpg
│   ├── sample_all_lights.jpg
│   ├── sample_multiple_lights.jpg
│   ├── sample_night_scene.jpg
│   └── sample_challenging_scene.jpg
└── venv/                          # Virtual environment (created automatically)
```

## 🔧 Technical Details

### Detection Algorithm

1. **Preprocessing**: White-balance correction, gamma adjustment, Gaussian blur
2. **Color Segmentation**: HSV-based masking for Red, Yellow, and Green ranges
3. **Morphology**: Closing and opening operations to clean up masks
4. **Contour Analysis**: Size, aspect ratio, circularity, and brightness filtering
5. **Confidence Scoring**: Geometry, color purity, and brightness-based scoring
6. **Tracking**: IoU-based association with temporal smoothing
7. **Classification**: Traffic light vs vehicle light heuristics

### Performance Features

- **Dynamic Scaling**: Parameters adjust automatically based on input resolution
- **Multi-threading**: Efficient processing for real-time performance
- **Memory Optimization**: Minimal memory footprint for long-running sessions
- **Error Handling**: Robust error recovery and user feedback

## 🛠️ Troubleshooting

### Common Issues

**No camera feed / black window**

- Check camera permissions and availability
- Try different camera indices (0, 1, 2, etc.)
- Ensure no other applications are using the camera

**Could not open video source**

- Verify file path and format support
- Try absolute paths instead of relative paths
- Check file permissions and corruption

**Low detection accuracy**

- Ensure adequate lighting conditions
- Try adjusting confidence threshold
- Use debug mode to verify color mask generation
- Point camera at larger, clearer traffic signals

**Performance issues**

- Close other resource-intensive applications
- Reduce input resolution if possible
- Consider using image processing instead of video for testing

### Debug Mode

Enable debug visualization to see:

- Color segmentation masks
- Contour detection results
- Brightness and saturation filtering
- Final detection bounding boxes

## 📊 Sample Images

The project includes several sample images for testing:

- **🔴 Red Light**: Various red traffic light scenarios
- **🟡 Yellow Light**: Yellow/amber light detection
- **🟢 Green Light**: Green traffic light examples
- **🚦 All Lights**: Multiple traffic lights in one scene
- **🏙️ Multiple Lights**: Complex urban traffic scenarios
- **🌙 Night Scene**: Low-light detection challenges
- **🎯 Challenging Scene**: Difficult detection scenarios

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source. Please check the repository for specific licensing terms.

## 🙏 Acknowledgements

Built with:

- **OpenCV**: Computer vision processing
- **Streamlit**: Web interface framework
- **NumPy**: Numerical computations
- **Pillow**: Image processing

Thanks to the open-source community for tools and prior art on HSV-based traffic light detection.
