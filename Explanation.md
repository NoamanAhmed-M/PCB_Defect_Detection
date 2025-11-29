# PCB Defect Detection App - Crash Course Guide

## Overview

This is a Streamlit web application that uses YOLOv12 for real-time PCB defect detection. It allows users to upload images and get instant defect analysis.

---

## Section-by-Section Breakdown

### 1. **Imports (Lines 1-6)**

```python
import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
```

**What they do:**

- `streamlit` - Web framework for the UI
- `cv2` - OpenCV, for image processing (color conversion, etc.)
- `numpy` - For array operations
- `os` - For file path checking
- `ultralytics` - YOLO model from Ultralytics
- `PIL.Image` - Python Imaging Library for image loading

---

### 2. **Page Configuration (Lines 8-9)**

```python
st.set_page_config(page_title="PCB Defect Detector", layout="wide", initial_sidebar_state="expanded")
```

**Does:**

- Sets the browser tab title to "PCB Defect Detector"
- Uses wide layout (more horizontal space)
- Opens sidebar by default

---

### 3. **Custom CSS Styling (Lines 11-20)**

```python
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .header { color: #1f77b4; font-size: 2.5em; font-weight: bold; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)
```

**Does:**

- Adds custom styling with CSS
- `.main` - Adds padding around main content
- `.header` - Styles the title (blue color, large font, bold)

---

### 4. **Title Display (Lines 22-23)**

```python
st.markdown('<div class="header">🔬 PCB Defect Detection System</div>', unsafe_allow_html=True)
st.write("Upload a PCB image to detect defects using YOLOv12 model")
```

**Does:**

- Displays the main header with emoji and custom styling
- Shows subtitle explaining what the app does

---

### 5. **Load Model Function (Lines 25-31)**

```python
@st.cache_resource
def load_model():
    """Load the trained YOLO model"""
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found!")
        return None
    return YOLO(model_path)
```

**What it does:**

- `@st.cache_resource` - Decorator that caches the model (loads only once, not every refresh)
- Checks if `best.pt` exists in the current directory
- If not found, displays error message and returns `None`
- If found, loads the YOLO model with `YOLO(model_path)`

**🐛 Common Issue:** If `best.pt` isn't in the same folder as `app.py`, this will fail. Make sure `best.pt` is in your project directory.

---

### 6. **Model Loading & Error Handling (Lines 33-36)**

```python
model = load_model()

if model is None:
    st.stop()
```

**Does:**

- Calls the `load_model()` function
- If model loading fails (returns `None`), `st.stop()` halts the app

---

### 7. **Sidebar Settings (Lines 38-42)**

```python
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    iou = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.01)
```

**Does:**

- Creates a sidebar (right panel) with settings
- **Confidence Threshold:** Controls how confident the model must be (0.5 = 50%)
  - Higher = fewer detections but more accurate
  - Lower = more detections but might include false positives
- **IOU Threshold:** Controls how much objects can overlap before being merged
  - Higher = objects must be very different to count as separate
  - Lower = stricter separation

---

### 8. **Main Layout (Lines 44-48)**

```python
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose a PCB image", type=["jpg", "jpeg", "png", "bmp"])
```

**Does:**

- `st.columns(2)` - Splits the page into 2 equal columns
- Left column: File uploader for images
- `st.file_uploader()` - Creates a drag-and-drop upload box
- `type=["jpg", "jpeg", "png", "bmp"]` - Only allows these image formats

---

### 9. **Image Processing & Inference (Lines 57-66)**

```python
if uploaded_file is not None or use_webcam:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

    with st.spinner("Detecting defects..."):
        results = model.predict(img_array, conf=confidence, iou=iou)
        result = results[0]
        annotated_img = result.plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
```

**Step-by-step:**

1. `if uploaded_file is not None` - Check if user uploaded an image
2. `Image.open(uploaded_file)` - Opens the image file
3. `np.array(image)` - Converts to numpy array (what YOLO needs)
4. `st.spinner("...")` - Shows loading spinner while processing
5. `model.predict()` - Runs YOLO detection
   - `conf=confidence` - Uses the slider value
   - `iou=iou` - Uses the IOU slider value
6. `results[0]` - Gets first result (only one image)
7. `result.plot()` - Draws bounding boxes on the image
8. `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` - Converts BGR (OpenCV format) to RGB (for display)

---

### 10. **Display Results (Lines 68-75)**

```python
col_result, col_stats = st.columns([2, 1])

with col_result:
    st.subheader("Detection Results")
    st.image(annotated_img_rgb, use_column_width=True)
```

**Does:**

- Creates 2 columns: left (2x wider) for image, right for stats
- Displays the annotated image with bounding boxes

---

### 11. **Statistics Display (Lines 77-96)**

```python
with col_stats:
    st.subheader("Statistics")

    if len(result.boxes) > 0:
        st.metric("Total Defects Detected", len(result.boxes))

        class_names = model.names
        detected_classes = result.boxes.cls.cpu().numpy()

        for class_id in np.unique(detected_classes):
            class_id = int(class_id)
            count = np.sum(detected_classes == class_id)
            st.metric(f"Class: {class_names[class_id]}", int(count))
```

**Breaks down:**

- `if len(result.boxes) > 0` - If detections exist:
  - `st.metric()` - Displays a big number (total count)
  - `model.names` - Gets class labels (e.g., "solder_defect", "crack")
  - `result.boxes.cls` - Gets the class ID for each detection
  - Loop through each unique class and count them
  - Display count for each defect type

---

### 12. **Instructions (Lines 99-111)**

```python
with st.expander("How to Use"):
    st.write("""
    1. **Upload an image**: Choose a PCB image from your device
    ...
    """)
```

**Does:**

- `st.expander()` - Creates a collapsible section
- Users can click to expand/collapse help text

---

## 🐛 Common Errors & How to Fix

| Error                            | Cause                         | Solution                                  |
| -------------------------------- | ----------------------------- | ----------------------------------------- |
| `FileNotFoundError: best.pt`     | Model file missing            | Move `best.pt` to same folder as `app.py` |
| `ModuleNotFoundError: streamlit` | Package not installed         | Run `python -m pip install streamlit`     |
| `Image not showing`              | Wrong image format            | Use JPG, PNG, or BMP files                |
| `No detections found`            | Confidence threshold too high | Lower the slider in Settings              |

---

## Debugging Tips

1. **Print detected values:**
   Add after line 63:

   ```python
   st.write(f"Debug - Image shape: {img_array.shape}")
   st.write(f"Debug - Number of boxes: {len(result.boxes)}")
   ```

2. **Check model predictions:**

   ```python
   st.write(result)  # Shows raw YOLO output
   ```

3. **Verify model is loaded:**
   ```python
   st.write(model.names)  # Shows class names
   ```

---

## Key Variables Explained

| Variable            | What It Is                        | Example                               |
| ------------------- | --------------------------------- | ------------------------------------- |
| `confidence`        | Detection confidence (0-1)        | 0.5 means 50% confident               |
| `iou`               | Intersection-over-Union threshold | 0.5 means boxes must be 50% different |
| `result.boxes`      | All detected objects              | List of detections                    |
| `result.boxes.cls`  | Class ID of each detection        | [0, 1, 0] = defect types              |
| `result.boxes.conf` | Confidence score (0-1)            | [0.95, 0.87, 0.92]                    |

---

## Testing Checklist

- [ ] `best.pt` is in the project folder
- [ ] All packages installed: `streamlit`, `pillow`, `opencv-python`, `ultralytics`
- [ ] App runs with: `python -m streamlit run app.py`
- [ ] Can upload image (JPG/PNG)
- [ ] Model detects at least one defect
- [ ] Confidence slider changes results
- [ ] Stats show correct count

---
