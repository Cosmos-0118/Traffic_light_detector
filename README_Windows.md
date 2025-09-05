# Traffic Light Detection - Windows Setup Guide

This guide will help you set up and run the Traffic Light Detection application on Windows.

## Prerequisites

### 1. Python Installation

- Download Python 3.7 or higher from [python.org](https://python.org)
- **Important**: During installation, check "Add Python to PATH"
- Verify installation by opening Command Prompt and running:
  ```cmd
  python --version
  ```

### 2. Git (Optional)

- Download Git from [git-scm.com](https://git-scm.com) if you want to clone the repository

## Quick Start

### Method 1: Using Batch Files (Recommended)

1. **Download the project files** to a folder on your computer
2. **Open Command Prompt** in the project folder:

   - Press `Win + R`, type `cmd`, press Enter
   - Navigate to the project folder: `cd "C:\path\to\your\project"`

3. **Run the application**:
   - For Simple UI: `run_detector.bat`
   - For Web Interface: `run_streamlit.bat`

### Method 2: Manual Setup

1. **Create virtual environment**:

   ```cmd
   python -m venv venv
   ```

2. **Activate virtual environment**:

   ```cmd
   venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```cmd
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```cmd
   python simple_ui.py
   ```

## Application Modes

### 1. Simple UI Mode (`run_detector.bat`)

- Interactive menu-driven interface
- Supports webcam, image files, and video files
- Press 'q' to quit, 'd' to toggle debug info

### 2. Web Interface Mode (`run_streamlit.bat`)

- Modern web-based interface
- Upload images/videos through browser
- Access at: http://localhost:8501
- Press Ctrl+C to stop

## Troubleshooting

### Common Issues

#### "Python is not recognized"

- **Solution**: Reinstall Python with "Add Python to PATH" checked
- Or manually add Python to your system PATH

#### "OpenCV installation failed"

- **Solution**: Try installing OpenCV separately:
  ```cmd
  pip install opencv-python-headless
  ```

#### "Permission denied" errors

- **Solution**: Run Command Prompt as Administrator
- Or check if antivirus is blocking the application

#### "Module not found" errors

- **Solution**: Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### Performance Issues

#### Slow detection

- Try disabling preprocessing: Add `--no-preprocessing` flag
- Try disabling tracking: Add `--no-tracking` flag

#### High CPU usage

- Close other applications
- Reduce video resolution if using webcam

## File Structure

```
Traffic-Lights-Tracking-and-Color-Detection-OpenCV/
├── run_detector.bat          # Windows batch file for Simple UI
├── run_streamlit.bat         # Windows batch file for Web Interface
├── run_detector.sh           # Linux/Mac shell script
├── run_streamlit.sh          # Linux/Mac shell script
├── traffic_light_detector.py # Main detection engine
├── simple_ui.py              # Simple UI interface
├── app.py                    # Streamlit web interface
├── requirements.txt          # Python dependencies
└── sample_images/            # Sample images for testing
```

## Command Line Options

### Simple UI Mode

```cmd
python simple_ui.py [options]
```

### Direct Detection Mode

```cmd
python traffic_light_detector.py [options]
```

Options:

- `-s, --source`: Camera index (0) or file path
- `-o, --output`: Save processed output
- `-i, --image`: Process as image file
- `--no-preprocessing`: Disable image preprocessing
- `--no-tracking`: Disable object tracking

## Sample Usage

### Process an image file:

```cmd
python traffic_light_detector.py -i -s "path/to/image.jpg" -o
```

### Use webcam:

```cmd
python traffic_light_detector.py -s 0
```

### Process video file:

```cmd
python traffic_light_detector.py -s "path/to/video.mp4" -o
```

## System Requirements

- **OS**: Windows 10 or later
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **Camera**: Any USB webcam (for real-time detection)

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Verify Python installation and PATH
3. Try running in Command Prompt as Administrator
4. Check Windows Defender/Antivirus settings
5. Ensure all dependencies are installed correctly

## Features

- **Real-time detection** from webcam
- **Image processing** for single images
- **Video processing** for video files
- **Multiple color detection** (Red, Yellow, Green)
- **Object tracking** for stable detection
- **Debug visualization** for troubleshooting
- **Cross-platform compatibility** (Windows, Linux, Mac)

## License

This project is open source. Please check the main README.md for license information.
