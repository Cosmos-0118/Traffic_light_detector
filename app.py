#!/usr/bin/env python3
"""
Traffic Light Detection System - Professional Web Interface

A clean, modern web interface for real-time traffic light detection.
Features: Image upload, video processing, sample content, and live detection.
"""

# Robust OpenCV import with fallback - MUST be first
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError as e:
    OPENCV_AVAILABLE = False
    cv2 = None

import streamlit as st
import numpy as np
import tempfile
import os
from datetime import datetime
from pathlib import Path

# Show error if OpenCV failed to import
if not OPENCV_AVAILABLE:
    st.error(
        "OpenCV import failed. Please ensure opencv-python-headless is installed."
    )
    st.error("Try: pip install opencv-python-headless")
    st.stop()

# Only import detector if OpenCV is available
if OPENCV_AVAILABLE:
    try:
        from traffic_light_detector import AutoTrafficLightDetector
        DETECTOR_AVAILABLE = True
    except ImportError as e:
        st.error(f"Traffic light detector import failed: {e}")
        DETECTOR_AVAILABLE = False
        AutoTrafficLightDetector = None
else:
    DETECTOR_AVAILABLE = False
    AutoTrafficLightDetector = None

# Page configuration
st.set_page_config(page_title="Traffic Light Detection System",
                   page_icon="🚦",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Global session state initialization - do this at the very beginning
# (Webcam functionality removed) Session state no longer initializes webcam flags.

# Professional CSS styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f4e3d;
        --secondary-color: #28a745;
        --accent-color: #ffc107;
        --text-color: #2c3e50;
        --bg-color: #f8f9fa;
        --card-bg: #ffffff;
        --border-color: #dee2e6;
    }

    /* Global styles */
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-color);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        opacity: 0.8;
    }

    /* Card styling */
    .detection-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
    }

    .stats-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--secondary-color) 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #218838 0%, #1ea085 100%);
    }

    /* Sample button styling */
    .sample-btn {
        background: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin: 0.25rem 0 !important;
    }

    .sample-btn:hover {
        border-color: var(--secondary-color) !important;
        background: #f8f9fa !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    /* Upload area styling */
    .upload-area {
        border: 2px dashed var(--secondary-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        background: #e8f5e8;
        border-color: var(--primary-color);
    }

    /* Metric styling */
    .metric-container {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid var(--border-color);
        margin: 0.5rem 0;
    }

    /* Sidebar styling */
    .sidebar .stSelectbox label,
    .sidebar .stSlider label,
    .sidebar .stCheckbox label {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }

    /* Hide Streamlit branding */
        /* Hide Streamlit branding (keep header so sidebar can be reopened) */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* NOTE: Do NOT hide the header. Hiding it removes the hamburger needed to reopen the sidebar
           after a user collapses it. */
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--secondary-color);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
</style>

<!-- Webcam JavaScript removed -->
""",
            unsafe_allow_html=True)


# Initialize the detector
@st.cache_resource
def get_detector():
    if DETECTOR_AVAILABLE and AutoTrafficLightDetector is not None:
        return AutoTrafficLightDetector()
    return None


detector = get_detector()

# ---------------------------------------------------------------------------
# Resource path resolution helpers (for sample images/videos)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Try to auto-detect project root that actually contains sample folders
_candidate_roots = [
    BASE_DIR,
    BASE_DIR.parent,
    Path.cwd(),
]
PROJECT_ROOT = BASE_DIR
for _cand in _candidate_roots:
    if (_cand / 'sample_videos').is_dir() and (_cand /
                                               'sample_images').is_dir():
        PROJECT_ROOT = _cand
        break


def resolve_resource(rel_path: str):
    """Return absolute path for a sample resource or None if not found.

    Tries several candidate locations to be robust against different run dirs.
    """
    rel_path = rel_path.lstrip('/').replace('..', '')  # basic sanitation
    candidates = [
        Path(rel_path),
        Path.cwd() / rel_path,
        PROJECT_ROOT / rel_path,
        BASE_DIR / rel_path,
        BASE_DIR.parent / rel_path,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def debug_missing_resource_message(rel_path: str):
    attempted = [
        Path(rel_path),
        Path.cwd() / rel_path,
        PROJECT_ROOT / rel_path,
        BASE_DIR / rel_path,
        BASE_DIR.parent / rel_path,
    ]
    attempted_str = '\n'.join(f" - {p}" for p in attempted)
    return (
        f"Resource '{rel_path}' not found. Tried:\n{attempted_str}\n"
        f"Current working dir: {Path.cwd()} | Script dir: {BASE_DIR} | Project root guess: {PROJECT_ROOT}"
    )


def create_debug_masks(image, detector):
    """Create debug mask visualization showing color segmentation"""
    if image is None:
        return None

    try:
        # Calculate dynamic parameters
        dynamic_params = detector._calculate_dynamic_params(image)

        # Preprocessing: white-balance, gamma, blur
        pre = detector._gray_world_white_balance(image)
        pre = detector._auto_gamma_correct(pre)
        blurred = cv2.GaussianBlur(pre, dynamic_params['blur_kernel_size'], 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])

        # Brightness and saturation gates
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        bright_thresh = max(20, min(60, int(np.percentile(gray, 15))))
        _, bright_mask = cv2.threshold(gray, bright_thresh, 255,
                                       cv2.THRESH_BINARY)

        if np.mean(s) < 15:
            combined_mask = bright_mask
        else:
            _, saturation_mask = cv2.threshold(s, 5, 255, cv2.THRESH_BINARY)
            combined_mask = cv2.bitwise_and(bright_mask, saturation_mask)

        # Create debug panel
        debug_panel = np.zeros((image.shape[0], image.shape[1], 3),
                               dtype=np.uint8)

        # Generate color masks for each color
        for color, ranges in detector.color_ranges.items():
            color_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for r in ranges:
                mask = cv2.inRange(hsv, r['lower'], r['upper'])
                mask = cv2.bitwise_and(mask, combined_mask)
                color_mask = cv2.bitwise_or(color_mask, mask)

            # Apply morphology
            kernel_small = np.ones((dynamic_params['small_kernel_size'],
                                    dynamic_params['small_kernel_size']),
                                   np.uint8)
            kernel_close = np.ones(
                (max(7, dynamic_params['large_kernel_size']),
                 max(7, dynamic_params['large_kernel_size'])), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE,
                                          kernel_close)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,
                                          kernel_small)
            color_mask = cv2.medianBlur(color_mask, 3)

            # Color the mask
            if color == 'Red':
                color_bgr = (0, 0, 255)
            elif color == 'Yellow':
                color_bgr = (0, 255, 255)
            else:  # Green
                color_bgr = (0, 255, 0)

            colored = cv2.merge([
                (color_mask //
                 2) if color_bgr[0] > 0 else np.zeros_like(color_mask),
                (color_mask //
                 2) if color_bgr[1] > 0 else np.zeros_like(color_mask),
                (color_mask //
                 2) if color_bgr[2] > 0 else np.zeros_like(color_mask),
            ])
            debug_panel = cv2.add(debug_panel, colored)

        return debug_panel

    except Exception as e:
        st.error(f"Error creating debug masks: {str(e)}")
        return None


def process_image(image, filename):
    """Process uploaded image and display results"""
    if image is None:
        st.error("❌ Could not process image. Please try again.")
        return

    # Get image properties
    height, width = image.shape[:2]

    # Create placeholders for image display
    image_placeholder = st.empty()
    status_text = st.empty()

    # Process image
    status_text.text("🔍 Processing image...")

    # Detect traffic lights once
    detections = detector.detect_traffic_lights(image)

    # Create annotated image manually to avoid double detection
    result_image = image.copy()
    for i, detection in enumerate(detections):
        x, y, w, h = detection['box']
        color_name = detection['color']
        confidence = detection['confidence']

        # Choose box color based on detected color
        if color_name == 'Red':
            box_color = (0, 0, 255)  # Red in BGR
        elif color_name == 'Yellow':
            box_color = (0, 255, 255)  # Yellow in BGR
        else:  # Green
            box_color = (0, 255, 0)  # Green in BGR

        # Draw bounding box
        cv2.rectangle(result_image, (x, y), (x + w, y + h), box_color, 3)

        # Draw label background
        cv2.rectangle(result_image, (x, y - 25), (x + w, y), box_color, -1)

        # Draw label text
        label = f"{color_name} ({confidence:.2f})"
        cv2.putText(result_image, label, (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Count detections
    red_count = sum(1 for d in detections if d['color'] == 'Red')
    yellow_count = sum(1 for d in detections if d['color'] == 'Yellow')
    green_count = sum(1 for d in detections if d['color'] == 'Green')

    # Draw info bar
    info_height = 40
    overlay = result_image.copy()
    cv2.rectangle(overlay, (0, 0), (result_image.shape[1], info_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)

    # Add detection counts
    info_text = f"Red: {red_count} | Yellow: {yellow_count} | Green: {green_count} | Total: {len(detections)}"
    cv2.putText(result_image, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # Update session state for sidebar stats
    current_detection_counts = {
        'Red': red_count,
        'Yellow': yellow_count,
        'Green': green_count
    }
    st.session_state.detection_counts = current_detection_counts.copy()

    # Convert BGR to RGB for display
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Display results
    if show_debug:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📸 Original")
            original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, channels="RGB", use_container_width=True)
        with col2:
            st.subheader("🎯 Detection Results")
            st.image(result_rgb, channels="RGB", use_container_width=True)
        with col3:
            st.subheader("🔍 Debug Masks")
            debug_image = create_debug_masks(image, detector)
            if debug_image is not None:
                debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
                st.image(debug_rgb, channels="RGB", use_container_width=True)
            else:
                st.error("Could not generate debug masks")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Original Image")
            original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, channels="RGB", use_container_width=True)
        with col2:
            st.subheader("🎯 Detection Results")
            st.image(result_rgb, channels="RGB", use_container_width=True)

    # Update status
    status_text.text(
        f"✅ Processing complete! | Red: {red_count} | Yellow: {yellow_count} | Green: {green_count}"
    )

    # Display final statistics
    st.success("✅ Image processing complete!")

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Red Lights", red_count)
    with col2:
        st.metric("🟡 Yellow Lights", yellow_count)
    with col3:
        st.metric("🟢 Green Lights", green_count)

    # Show detection details
    if any(current_detection_counts.values()):
        with st.expander("🔍 Detection Details", expanded=False):
            for i, detection in enumerate(detections):
                x, y, w, h = detection['box']
                color = detection.get('color', 'Unknown')
                confidence = detection.get('confidence', 0)
                st.write(
                    f"**{color} Light #{i+1}** - Confidence: {confidence:.2f} - Position: ({x}, {y}) - Size: {w}x{h}"
                )


## Webcam functionality removed to improve stability.


def process_video(video_path):
    """Process uploaded video and display results"""
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Could not open video file. Please check the file format.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    st.info(
        f"📹 Video Info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration"
    )

    # Create placeholders for video display
    video_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Detection counters
    detection_counts = {'Red': 0, 'Yellow': 0, 'Green': 0}
    frame_count = 0

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        result_frame = detector.process_frame(frame)

        # Count detections in current frame
        frame_detections = detector.detect_traffic_lights(frame)
        for detection in frame_detections:
            color = detection.get('color', 'Unknown')
            if color in detection_counts:
                detection_counts[color] += 1

        # Convert BGR to RGB for display
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # Display frame with or without debug masks
        if show_debug:
            debug_image = create_debug_masks(frame, detector)
            if debug_image is not None:
                debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
                # Create side-by-side comparison
                combined = np.hstack([result_rgb, debug_rgb])
                video_placeholder.image(combined,
                                        channels="RGB",
                                        use_container_width=True)
            else:
                video_placeholder.image(result_rgb,
                                        channels="RGB",
                                        use_container_width=True)
        else:
            video_placeholder.image(result_rgb,
                                    channels="RGB",
                                    use_container_width=True)

        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)

        # Update status
        status_text.text(
            f"Processing frame {frame_count}/{total_frames} | Red: {detection_counts['Red']} | Yellow: {detection_counts['Yellow']} | Green: {detection_counts['Green']}"
        )

        # Add small delay to make it viewable
        import time
        time.sleep(0.1)

    # Cleanup
    cap.release()

    # Update session state for sidebar stats
    st.session_state.detection_counts = detection_counts.copy()

    # Final results
    st.success("✅ Video processing complete!")

    # Display final statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Red Lights", detection_counts['Red'])
    with col2:
        st.metric("🟡 Yellow Lights", detection_counts['Yellow'])
    with col3:
        st.metric("🟢 Green Lights", detection_counts['Green'])

    # Clean up temporary file
    try:
        os.unlink(video_path)
    except:
        pass


# Check if dependencies are available
if not OPENCV_AVAILABLE or not DETECTOR_AVAILABLE or detector is None:
    st.error(
        "❌ Required dependencies are not available. Please check the installation."
    )
    st.stop()

# Main header
st.markdown('<h1 class="main-header">🚦 Traffic Light Detection System</h1>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Professional real-time detection of Red, Yellow, and Green traffic lights using advanced computer vision</p>',
    unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.title("🎛️ Controls")

    # Detection settings
    st.subheader("⚙️ Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Higher values = fewer but more confident detections")

    show_debug = st.checkbox(
        "Show Debug Masks",
        value=False,
        help="Toggle color mask visualization for advanced users")

    # Update detector settings
    detector.confidence_threshold = confidence_threshold
    detector.show_debug_masks = show_debug

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📸 Input Source")

    # Input method selection with better styling
    input_method = st.radio(
        "Choose your input method:",
        ["🖼️ Upload Image", "📹 Upload Video", "📁 Sample Content"],
        horizontal=True,
        help=
        "Select how you want to provide input for traffic light detection (webcam disabled for stability)"
    )

    # Store current input method in session state for webcam control
    st.session_state.current_input_method = input_method

    st.markdown("---")

    if input_method == "🖼️ Upload Image":
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to detect traffic lights",
            label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_image is not None:
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_image.read()),
                                    dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            st.success(f"✅ Image uploaded: {uploaded_image.name}")

            # Process image
            if st.button("🚀 Detect Traffic Lights",
                         type="primary",
                         use_container_width=True):
                process_image(image, uploaded_image.name)

    elif input_method == "📹 Upload Video":
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
            help="Upload a video file to detect traffic lights",
            label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f'.{uploaded_file.name.split(".")[-1]}'
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            st.success(f"✅ Video uploaded: {uploaded_file.name}")

            # Process video
            if st.button("🚀 Start Detection",
                         type="primary",
                         use_container_width=True):
                process_video(video_path)

    elif input_method == "📁 Sample Content":
        st.markdown("### 🎯 Try Our Sample Content")

        # (Webcam option removed) Proceed with existing sample logic below.

        # Sample content selection
        sample_type = st.radio("Choose sample type:",
                               ["📸 Sample Images", "🎬 Sample Video"],
                               horizontal=True)

        if sample_type == "📸 Sample Images":
            st.markdown("Select a sample image to test the detection system:")

            # Sample images with better organization
            sample_images = {
                "🔴 Red Light": "sample_images/sample_red_light.jpg",
                "🟡 Yellow Light": "sample_images/sample_yellow_light.jpg",
                "🟢 Green Light": "sample_images/sample_green_light.jpg",
                "🚦 All Lights": "sample_images/sample_all_lights.jpg",
                "🏙️ Multiple Lights":
                "sample_images/sample_multiple_lights.jpg",
                "🌙 Night Scene": "sample_images/sample_night_scene.jpg",
                "🎯 Challenging Scene":
                "sample_images/sample_challenging_scene.jpg"
            }

            # Initialize selected sample in session state
            if 'selected_sample' not in st.session_state:
                st.session_state.selected_sample = None

            # Create a grid of sample buttons
            cols = st.columns(2)
            for i, (name, path) in enumerate(sample_images.items()):
                with cols[i % 2]:
                    if st.button(name,
                                 key=f"sample_{i}",
                                 use_container_width=True):
                        st.session_state.selected_sample = path
                        st.rerun()

            # Process selected sample
            if st.session_state.selected_sample:
                rel_path = st.session_state.selected_sample
                image_path = resolve_resource(rel_path)

                if image_path is not None:
                    image = cv2.imread(image_path)
                    if image is not None:
                        st.success(
                            f"✅ Selected: {os.path.basename(image_path)}")

                        # Display the sample image
                        if show_debug:
                            col_img1, col_img2 = st.columns(2)
                            with col_img1:
                                st.image(
                                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                    channels="RGB",
                                    use_container_width=True,
                                    caption=
                                    f"Sample: {os.path.basename(image_path)}")
                            with col_img2:
                                debug_image = create_debug_masks(
                                    image, detector)
                                if debug_image is not None:
                                    debug_rgb = cv2.cvtColor(
                                        debug_image, cv2.COLOR_BGR2RGB)
                                    st.image(debug_rgb,
                                             channels="RGB",
                                             use_container_width=True,
                                             caption="Debug Masks")
                                else:
                                    st.error("Could not generate debug masks")
                        else:
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                     channels="RGB",
                                     use_container_width=True,
                                     caption=
                                     f"Sample: {os.path.basename(image_path)}")

                        # Process button
                        col_btn1, col_btn2 = st.columns([2, 1])
                        with col_btn1:
                            if st.button("🚀 Detect Traffic Lights in Sample",
                                         type="primary",
                                         use_container_width=True):
                                process_image(image,
                                              os.path.basename(image_path))
                        with col_btn2:
                            if st.button("❌ Clear", use_container_width=True):
                                st.session_state.selected_sample = None
                    else:
                        st.error("❌ Could not load the selected sample image.")
                else:
                    st.warning("⚠️ Sample image not found.")
                    with st.expander("Why is this missing?", expanded=False):
                        st.code(debug_missing_resource_message(rel_path))

        elif sample_type == "🎬 Sample Video":
            st.markdown("### 🎬 Sample Video Detection")
            st.info("Select a sample video to test the detection system.")

            # Sample videos with better organization
            sample_videos = {
                "🔴 Red Light Video": "sample_videos/sample_red_light.mp4",
                "🟡 Yellow Light Video":
                "sample_videos/sample_yellow_light.mp4",
                "🟢 Green Light Video": "sample_videos/sample_green_light.mp4",
                "🚦 All Lights Video": "sample_videos/sample_all_lights.mp4",
                "🏙️ Multiple Lights Video":
                "sample_videos/sample_multiple_lights.mp4",
                "🌙 Night Scene Video": "sample_videos/sample_night_scene.mp4",
                "🎯 Challenging Scene Video":
                "sample_videos/sample_challenging_scene.mp4",
                "🔴 Red Bottom Video": "sample_videos/sample_red_bottom.mp4",
                "🔴 Red Left Video": "sample_videos/sample_red_left.mp4",
                "🔴 Red Right Video": "sample_videos/sample_red_right.mp4",
                "🔴 Red Top Video": "sample_videos/sample_red_top.mp4"
            }

            # Initialize selected sample video in session state
            if 'selected_sample_video' not in st.session_state:
                st.session_state.selected_sample_video = None

            # Create a grid of sample video buttons
            cols = st.columns(2)
            for i, (name, path) in enumerate(sample_videos.items()):
                with cols[i % 2]:
                    if st.button(name,
                                 key=f"sample_video_{i}",
                                 use_container_width=True):
                        st.session_state.selected_sample_video = path
                        st.rerun()

            # Process selected sample video
            if st.session_state.selected_sample_video:
                rel_video = st.session_state.selected_sample_video
                video_path = resolve_resource(rel_video)

                if video_path is not None:
                    st.success(f"✅ Selected: {os.path.basename(video_path)}")

                    # Get video info
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = total_frames / fps if fps > 0 else 0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()

                        st.info(
                            f"📹 Video Info: {width}x{height}, {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration"
                        )

                        # Process button
                        col_btn1, col_btn2 = st.columns([2, 1])
                        with col_btn1:
                            if st.button("🚀 Process Sample Video",
                                         type="primary",
                                         use_container_width=True):
                                process_video(video_path)
                        with col_btn2:
                            if st.button("❌ Clear", use_container_width=True):
                                st.session_state.selected_sample_video = None
                                st.rerun()
                    else:
                        st.error(
                            "❌ Could not read the selected sample video file.")
                else:
                    st.warning("⚠️ Sample video not found.")
                    with st.expander("Why is this missing?", expanded=False):
                        st.code(debug_missing_resource_message(rel_video))

with col2:
    st.subheader("📊 Detection Statistics")

    # Initialize detection counts
    if 'detection_counts' not in st.session_state:
        st.session_state.detection_counts = {'Red': 0, 'Yellow': 0, 'Green': 0}

    # Display stats using custom styling
    st.markdown('<div class="stats-card">', unsafe_allow_html=True)

    total_detections = sum(st.session_state.detection_counts.values())
    if total_detections == 0:
        st.info(
            "ℹ️ No detections yet. Select content and click 'Detect Traffic Lights' to see results."
        )
    else:
        col_red, col_yellow, col_green = st.columns(3)
        with col_red:
            st.metric("🔴 Red", st.session_state.detection_counts['Red'])
        with col_yellow:
            st.metric("🟡 Yellow", st.session_state.detection_counts['Yellow'])
        with col_green:
            st.metric("🟢 Green", st.session_state.detection_counts['Green'])

    st.markdown('</div>', unsafe_allow_html=True)

    # Clear stats button
    if st.button("🔄 Clear Stats", use_container_width=True):
        st.session_state.detection_counts = {'Red': 0, 'Yellow': 0, 'Green': 0}
        st.session_state.cumulative_detection_counts = {
            'Red': 0,
            'Yellow': 0,
            'Green': 0
        }

    # Information panel
    st.markdown("""
    <div class="detection-card">
        <h4>🔍 How it works:</h4>
        <ul>
            <li>HSV color segmentation</li>
            <li>Adaptive preprocessing</li>
            <li>Contour filtering</li>
            <li>Real-time tracking</li>
        </ul>
        <br>
        <h4>📸 Supported formats:</h4>
        <ul>
            <li><strong>Images:</strong> JPG, PNG, BMP, TIFF</li>
            <li><strong>Videos:</strong> MP4, AVI, MOV, MKV</li>
        </ul>
    </div>
    """,
                unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>🚦 Traffic Light Detection System</strong></p>
    <p>Built with OpenCV & Streamlit | Professional Computer Vision Solution</p>
</div>
""",
            unsafe_allow_html=True)

# Advanced options in sidebar
with st.sidebar:
    st.markdown("---")
    if st.checkbox("🔧 Advanced Options"):
        st.markdown("""
        **Local Testing:**
        1. Install: `pip install streamlit`
        2. Run: `streamlit run app.py`
        3. Open: http://localhost:8501
        
        **Deployment:**
        1. Push code to GitHub
        2. Go to share.streamlit.io
        3. Connect your repo
        4. Deploy!
        """)
