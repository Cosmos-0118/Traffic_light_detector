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
import math
import random
from PIL import Image
import io

# Show error if OpenCV failed to import
if not OPENCV_AVAILABLE:
    st.error(
        "OpenCV import failed. Please ensure opencv-python-headless is installed."
    )
    st.error("Try: pip install opencv-python-headless")
    st.stop()


# Function to check and install dependencies
def ensure_dependencies():
    missing_deps = []
    try:
        import cv2
    except ImportError:
        missing_deps.append('opencv-python-headless>=4.8.0')

    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy>=1.20.0')

    try:
        from PIL import Image
    except ImportError:
        missing_deps.append('Pillow>=8.0.0')

    # Auto-install missing dependencies (streamlit is obviously available if we're here)
    if missing_deps:
        st.warning(
            f"Installing missing dependencies: {', '.join(missing_deps)}")
        import subprocess
        for dep in missing_deps:
            try:
                subprocess.check_call(['pip', 'install', dep])
                st.success(f"Installed {dep}")
            except Exception as e:
                st.error(f"Failed to install {dep}: {e}")
        st.info("Please restart the application after installing dependencies")
        st.stop()


# Call dependency check function
if st.session_state.get('is_web_environment', False):
    ensure_dependencies()

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
# Detect environment and optimize accordingly
if 'is_web_environment' not in st.session_state:
    # Check if running in a cloud environment by looking at environment variables
    import os
    is_cloud = any([
        os.environ.get('STREAMLIT_SHARING') == 'true',
        os.environ.get('IS_STREAMLIT_CLOUD') == 'true',
        os.environ.get('HOSTNAME', '').endswith('.streamlit.app'),
        os.environ.get('STREAMLIT_SERVER_HEADLESS') == 'true',
    ])
    st.session_state.is_web_environment = is_cloud

# Set performance optimizations based on environment
if 'performance_optimized' not in st.session_state:
    st.session_state.performance_optimized = True  # Enable by default
    st.session_state.frame_skip = 0  # No frame skipping by default

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
_candidate_roots = [BASE_DIR, BASE_DIR.parent, Path.cwd()]

# Walk up a few parent levels for robustness in hosted environments
_cur = BASE_DIR
for _ in range(5):
    _cur = _cur.parent
    _candidate_roots.append(_cur)

# Optional env var override
env_root = os.getenv('TRAFFIC_LIGHT_DET_ROOT')
if env_root:
    _candidate_roots.insert(0, Path(env_root))

PROJECT_ROOT = BASE_DIR
for _cand in _candidate_roots:
    try:
        if (_cand / 'sample_videos').is_dir() and (_cand /
                                                   'sample_images').is_dir():
            PROJECT_ROOT = _cand
            break
    except Exception:
        continue


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


# ---------------------------------------------------------------------------
# Synthetic sample video generation (replaces static bundled video assets)
# ---------------------------------------------------------------------------
def _apply_effect(img, i, total, effect):
    """Return a frame with a simple visual effect.

    Supported effects: zoom, pulse, panx, flicker, jitter, crop_jitter, steady
    """
    h, w = img.shape[:2]
    t = i / max(1, total - 1)
    frame = img.copy()

    try:
        if effect == 'zoom':
            scale = 1.0 + 0.10 * math.sin(2 * math.pi * t)
            nh, nw = int(h * scale), int(w * scale)
            resized = cv2.resize(frame, (nw, nh))
            y0 = max(0, (nh - h) // 2)
            x0 = max(0, (nw - w) // 2)
            frame = resized[y0:y0 + h, x0:x0 + w]
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
        elif effect == 'pulse':
            # Deeper dimming for robustness: brightness cycles between ~45% and 100%
            # Old range was ~75%-110% which rarely darkened the light; this new range
            # helps test low-intensity detection without oversaturating.
            alpha = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(2 * math.pi * t)
                                   )  # [0.45, 1.0]
            # Operate in HSV for more natural dimming (reduce V channel only)
            try:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h_ch, s_ch, v_ch = cv2.split(hsv)
                v_ch = np.clip(v_ch.astype(np.float32) * alpha, 0,
                               255).astype(np.uint8)
                hsv = cv2.merge([h_ch, s_ch, v_ch])
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            except Exception:
                # Fallback to simple global scaling if HSV conversion fails
                frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        elif effect == 'panx':
            shift = int(18 * math.sin(2 * math.pi * t))
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            frame = cv2.warpAffine(frame,
                                   M, (w, h),
                                   borderMode=cv2.BORDER_REFLECT)
        elif effect == 'flicker':
            alpha = 0.65 + 0.7 * random.random()
            beta = random.randint(-20, 20)
            noisy = frame.astype(np.float32)
            noise = np.random.normal(0, 10, size=frame.shape)
            noisy = np.clip(noisy + noise, 0, 255)
            frame = cv2.convertScaleAbs(noisy, alpha=alpha, beta=beta)
        elif effect == 'jitter':
            sx = random.randint(-6, 6)
            sy = random.randint(-6, 6)
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            frame = cv2.warpAffine(frame,
                                   M, (w, h),
                                   borderMode=cv2.BORDER_REFLECT)
        elif effect == 'crop_jitter':
            crop_scale = 0.88 + 0.1 * random.random()
            ch, cw = int(h * crop_scale), int(w * crop_scale)
            y0 = random.randint(0, max(0, h - ch))
            x0 = random.randint(0, max(0, w - cw))
            cropped = frame[y0:y0 + ch, x0:x0 + cw]
            frame = cv2.resize(cropped, (w, h))
        # steady: leave unchanged
    except Exception:
        pass
    return frame


def _write_video(output_path, frames, fps=18):
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        vw.write(f)
    vw.release()
    return True


def generate_sample_videos(force=False, fps=18, seconds=5):
    """Create lightweight synthetic videos derived from sample images.

    Duration default increased to 5s for better testing. Existing videos
    shorter than the new target auto-regenerate even if force=False.
    """
    out_dir = PROJECT_ROOT / 'sample_videos'
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        'sample_red_light.mp4':
        (['sample_images/sample_red_light.jpg'], 'zoom'),
        'sample_yellow_light.mp4':
        (['sample_images/sample_yellow_light.jpg'], 'pulse'),
        'sample_green_light.mp4': (['sample_images/sample_green_light.jpg'],
                                   'panx'),
        'sample_all_lights.mp4': ([
            'sample_images/sample_red_light.jpg',
            'sample_images/sample_yellow_light.jpg',
            'sample_images/sample_green_light.jpg'
        ], 'steady'),
        'sample_multiple_lights.mp4':
        (['sample_images/sample_multiple_lights.jpg'], 'jitter'),
        'sample_night_scene.mp4': (['sample_images/sample_night_scene.jpg'],
                                   'flicker'),
        'sample_challenging_scene.mp4':
        (['sample_images/sample_challenging_scene.jpg'], 'crop_jitter'),
        'sample_red_bottom.mp4': (['sample_images/sample_red_bottom.jpg'],
                                  'pulse'),
        'sample_red_left.mp4': (['sample_images/sample_red_left.jpg'], 'zoom'),
        'sample_red_right.mp4': (['sample_images/sample_red_right.jpg'],
                                 'zoom'),
        'sample_red_top.mp4': (['sample_images/sample_red_top.jpg'], 'pulse'),
    }

    total_frames = int(fps * seconds)
    created, skipped = [], []

    for filename, (image_list, effect) in specs.items():
        out_path = out_dir / filename
        regenerate = False
        if out_path.exists():
            if force:
                regenerate = True
            else:
                # Inspect existing video length; regenerate if too short
                cap = cv2.VideoCapture(str(out_path))
                if cap.isOpened():
                    existing_frames = int(
                        cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    existing_fps = cap.get(cv2.CAP_PROP_FPS) or fps
                    cap.release()
                    target_frames = int(fps * seconds *
                                        0.9)  # allow some slack
                    if existing_frames < target_frames:
                        regenerate = True
                else:
                    regenerate = True
        else:
            regenerate = True

        if not regenerate:
            skipped.append(filename)
            continue
        # Remove old file if present
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        frames = []
        imgs = []
        for rel in image_list:
            p = resolve_resource(rel)
            if p is None:
                continue
            img = cv2.imread(p)
            if img is not None:
                imgs.append(img)
        if not imgs:
            continue
        for i in range(total_frames):
            if effect == 'steady' and len(imgs) > 1:
                idx = int((i / total_frames) * len(imgs)) % len(imgs)
                frame = imgs[idx].copy()
            else:
                base = imgs[i % len(imgs)].copy()
                frame = _apply_effect(base, i, total_frames, effect)
            frames.append(frame)
        if _write_video(str(out_path), frames, fps=fps):
            created.append(filename)
    return created, skipped


def synthetic_video_specs():
    """Mapping of synthetic video file names to (image list, effect).

    Used for hosted fallback when encoded mp4 cannot be properly decoded
    (some Streamlit hosting containers lack certain codecs). In that case
    we regenerate frames in-memory so motion/effects still appear.
    """
    return {
        'sample_red_light.mp4':
        (['sample_images/sample_red_light.jpg'], 'zoom'),
        'sample_yellow_light.mp4':
        (['sample_images/sample_yellow_light.jpg'], 'pulse'),
        'sample_green_light.mp4': (['sample_images/sample_green_light.jpg'],
                                   'panx'),
        'sample_all_lights.mp4': ([
            'sample_images/sample_red_light.jpg',
            'sample_images/sample_yellow_light.jpg',
            'sample_images/sample_green_light.jpg'
        ], 'steady'),
        'sample_multiple_lights.mp4':
        (['sample_images/sample_multiple_lights.jpg'], 'jitter'),
        'sample_night_scene.mp4': (['sample_images/sample_night_scene.jpg'],
                                   'flicker'),
        'sample_challenging_scene.mp4':
        (['sample_images/sample_challenging_scene.jpg'], 'crop_jitter'),
        'sample_red_bottom.mp4': (['sample_images/sample_red_bottom.jpg'],
                                  'pulse'),
        'sample_red_left.mp4': (['sample_images/sample_red_left.jpg'], 'zoom'),
        'sample_red_right.mp4': (['sample_images/sample_red_right.jpg'],
                                 'zoom'),
        'sample_red_top.mp4': (['sample_images/sample_red_top.jpg'], 'pulse'),
    }


@st.cache_resource(show_spinner=False)
def ensure_sample_videos():
    return generate_sample_videos(force=False)


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
    """Process uploaded or sample video efficiently without duplicate detection calls."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Could not open video file. Please check the file format.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 18
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Enhanced codec handling for web environments
    # Rebuild synthetic frames in memory for better web compatibility
    in_memory_frames = None

    # More aggressive frame rebuilding in web environments or if frames are too few
    rebuild_threshold = 8 if not st.session_state.get('is_web_environment',
                                                      False) else 20
    if total_frames < rebuild_threshold:  # More aggressive for web environments
        fname = Path(video_path).name
        specs = synthetic_video_specs()
        if fname in specs:
            images, effect = specs[fname]
            imgs = []
            for rel in images:
                p = resolve_resource(rel)
                if not p:
                    continue
                img = cv2.imread(p)
                if img is not None:
                    imgs.append(img)
            if imgs:
                seconds = 5
                total_frames_target = int(fps * seconds)
                in_memory_frames = []
                for i in range(total_frames_target):
                    if effect == 'steady' and len(imgs) > 1:
                        idx = int(
                            (i / total_frames_target) * len(imgs)) % len(imgs)
                        frame = imgs[idx].copy()
                    else:
                        base = imgs[i % len(imgs)].copy()
                        frame = _apply_effect(base, i, total_frames_target,
                                              effect)
                    in_memory_frames.append(frame)
                total_frames = len(in_memory_frames)

    duration = total_frames / fps if fps > 0 else 0
    st.info(
        f"📹 Video Info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration"
    )

    video_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # Counters (count unique confirmed tracks per frame rather than raw detections to avoid overcount)
    cumulative_counts = {'Red': 0, 'Yellow': 0, 'Green': 0}
    frame_index = 0
    processed_frames = 0
    skipped_frames = 0

    # Performance optimization settings
    is_web = st.session_state.get('is_web_environment', False)
    # Use user-selected frame skip from session state if available
    frame_skip = st.session_state.get('frame_skip', 0)
    # Default behavior if not set in session state
    if frame_skip == 0 and st.session_state.get('performance_optimized',
                                                True) and is_web:
        frame_skip = 2  # Default to aggressive optimization for web

    # Calculate processing metrics
    import time
    t0 = time.time()
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    # Detect if detector.process_frame supports the new keyword arg once (compat for hosted older code)
    import inspect
    try:
        _sig = inspect.signature(detector.process_frame)
        _supports_kw = 'return_detections' in _sig.parameters
    except Exception:
        _supports_kw = False

    frame_iter = 0
    while True:
        if in_memory_frames is not None:
            if frame_iter >= len(in_memory_frames):
                break
            frame = in_memory_frames[frame_iter]
            ret = True
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame_iter += 1
        frame_index += 1

        # Implement frame skipping for performance optimization
        # Skip frames based on environment and settings, but process every keyframe
        if frame_skip > 0 and (frame_index %
                               (frame_skip + 1) != 0) and frame_index > 1:
            skipped_frames += 1
            continue

        processed_frames += 1

        # Calculate real-time FPS for performance monitoring
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        try:
            if _supports_kw:
                # Preferred modern path
                annotated, confirmed = detector.process_frame(
                    frame, return_detections=True)
            else:
                # Legacy fallback: may return only annotated frame
                pf_res = detector.process_frame(frame)
                if isinstance(pf_res, tuple) and len(pf_res) == 2:
                    annotated, confirmed = pf_res
                else:
                    annotated = pf_res
                    # Separate detection call (may be heavier but ensures counts)
                    try:
                        confirmed = detector.detect_traffic_lights(frame)
                    except Exception:
                        confirmed = []
        except TypeError:
            # Parameter not accepted (legacy), fallback like above
            try:
                pf_res = detector.process_frame(frame)
                if isinstance(pf_res, tuple) and len(pf_res) == 2:
                    annotated, confirmed = pf_res
                else:
                    annotated = pf_res
                    confirmed = detector.detect_traffic_lights(frame)
            except Exception:
                annotated = frame
                confirmed = []
        except Exception:
            annotated = frame
            confirmed = []

        # Update cumulative counts (per frame uniqueness)
        frame_colors = {d['color'] for d in confirmed}
        for c in frame_colors:
            if c in cumulative_counts:
                cumulative_counts[c] += 1

        # Prepare display image
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        if show_debug:
            debug_image = create_debug_masks(frame, detector)
            if debug_image is not None:
                debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
                frame_rgb = np.hstack([frame_rgb, debug_rgb])

        # Optimize image display with reduced quality in web environments for performance
        if st.session_state.get('is_web_environment', False):
            # For web: Compress image to reduce bandwidth and improve performance
            # Convert to PIL Image
            pil_img = Image.fromarray(frame_rgb)

            # Reduce size for web display if needed
            max_width = 800  # Limit width to reduce data transfer
            if pil_img.width > max_width:
                ratio = max_width / pil_img.width
                new_height = int(pil_img.height * ratio)
                pil_img = pil_img.resize((max_width, new_height),
                                         Image.LANCZOS)

            # Display the optimized image
            video_placeholder.image(pil_img,
                                    channels="RGB",
                                    use_container_width=True)
        else:
            # For local: Use full quality
            video_placeholder.image(frame_rgb,
                                    channels="RGB",
                                    use_container_width=True)

        # Progress & status with enhanced information
        if total_frames > 0:
            progress_bar.progress(min(1.0, frame_index / total_frames))

        # Show real-time performance metrics
        status_msg = f"Frame {frame_index}/{total_frames if total_frames else '?'}"
        status_msg += f" | FPS: {current_fps:.1f}" if current_fps > 0 else ""
        status_msg += f" | Red: {cumulative_counts['Red']} | Yellow: {cumulative_counts['Yellow']} | Green: {cumulative_counts['Green']}"
        if skipped_frames > 0:
            status_msg += f" | Skipped: {skipped_frames}"
        status_text.text(status_msg)

        # Adaptive frame processing for web vs local environments
        # Reduce sleep times significantly, especially for web environments
        # Only apply minimal sleep to prevent UI thread blocking
        if st.session_state.get('is_web_environment', False):
            # Web environment: minimal to no sleep
            pass  # No sleep for web deployment to maximize performance
        else:
            # Local environment: minimal sleep to prevent UI overload
            if frame_index < 10:
                time.sleep(0.01)  # Reduced from 0.03
            else:
                time.sleep(0.005)  # Reduced from 0.01

    cap.release()
    st.session_state.detection_counts = cumulative_counts.copy()

    # Show appropriate completion message with performance stats
    elapsed_time = time.time() - t0
    avg_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0

    completion_msg = f"✅ Video processing complete! Processed {processed_frames} frames in {elapsed_time:.1f}s ({avg_fps:.1f} FPS)"
    if skipped_frames > 0:
        completion_msg += f", skipped {skipped_frames} frames for performance"

    st.success(completion_msg)

    # Show web optimization message if applicable
    if st.session_state.get('is_web_environment', False):
        st.info(
            "🌐 Web optimization active - performance has been enhanced for smoother online experience"
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Red Lights", cumulative_counts['Red'])
    with col2:
        st.metric("🟡 Yellow Lights", cumulative_counts['Yellow'])
    with col3:
        st.metric("🟢 Green Lights", cumulative_counts['Green'])

    # Best-effort cleanup
    try:
        if os.path.exists(video_path) and video_path.startswith(
                tempfile.gettempdir()):
            os.unlink(video_path)
    except Exception:
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

    # Performance optimization settings
    st.subheader("🚀 Performance Settings")

    performance_optimized = st.checkbox(
        "Optimize for Web Performance",
        value=st.session_state.get('performance_optimized', True),
        help="Enable performance optimizations for better web experience")

    if performance_optimized:
        frame_skip_options = st.selectbox(
            "Frame Processing",
            options=[
                "Normal (Process All)", "Fast (Skip 1 Frame)",
                "Fastest (Skip 2 Frames)"
            ],
            index=2
            if st.session_state.get('is_web_environment', False) else 0,
            help=
            "Controls how many frames are skipped during processing. Skip more frames for better performance but potentially reduced detection accuracy."
        )

        # Update frame skip based on selection
        if frame_skip_options == "Normal (Process All)":
            st.session_state.frame_skip = 0
        elif frame_skip_options == "Fast (Skip 1 Frame)":
            st.session_state.frame_skip = 1
        else:  # Fastest
            st.session_state.frame_skip = 2
    else:
        st.session_state.frame_skip = 0

    # Status indicator for environment
    if st.session_state.get('is_web_environment', False):
        st.info("🌐 Running in web environment")
    else:
        st.success("🖥️ Running in local environment")

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

            # Generate synthetic sample videos if missing
            with st.spinner("Preparing synthetic sample videos..."):
                created, skipped = ensure_sample_videos()
            if created:
                st.success(f"Generated: {', '.join(created)}")
            # (Skip message suppressed to reduce noise)

            # Regenerate control
            regen_col1, regen_col2 = st.columns([2, 1])
            with regen_col1:
                st.caption(
                    "Need fresh variations? Regenerate with new random jitters."
                )
            with regen_col2:
                if st.button(
                        "♻️ Regenerate",
                        help=
                        "Force re-create all synthetic sample videos with new random effects variations."
                ):
                    # Clear cache first
                    try:
                        ensure_sample_videos.clear()
                    except Exception:
                        pass
                    with st.spinner("Regenerating videos..."):
                        new_created, _ = generate_sample_videos(force=True)
                    if new_created:
                        st.success(f"Recreated: {', '.join(new_created)}")
                    else:
                        st.warning("No videos regenerated.")
                    st.rerun()

            # Sample videos with better organization
            sample_videos = {
                "🔴 Red Light (Zoom)": "sample_videos/sample_red_light.mp4",
                "🟡 Yellow Light (Pulse)":
                "sample_videos/sample_yellow_light.mp4",
                "🟢 Green Light (Pan)": "sample_videos/sample_green_light.mp4",
                "🚦 All Lights (Cycle)": "sample_videos/sample_all_lights.mp4",
                "🏙️ Multiple Lights (Jitter)":
                "sample_videos/sample_multiple_lights.mp4",
                "🌙 Night Scene (Flicker)":
                "sample_videos/sample_night_scene.mp4",
                "🎯 Challenging Scene (Crop Jitter)":
                "sample_videos/sample_challenging_scene.mp4",
                "🔴 Red Bottom (Pulse)": "sample_videos/sample_red_bottom.mp4",
                "🔴 Red Left (Zoom)": "sample_videos/sample_red_left.mp4",
                "🔴 Red Right (Zoom)": "sample_videos/sample_red_right.mp4",
                "🔴 Red Top (Pulse)": "sample_videos/sample_red_top.mp4"
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
