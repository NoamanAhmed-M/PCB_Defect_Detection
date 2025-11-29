import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os

# Instructions
def Instructions():
    with st.expander("How does it work?"):
        st.write("""
        1. **Upload an image**: Choose a PCB image from your device
        2. **Adjust settings**: Use the sidebar to fine-tune confidence and IOU thresholds
        3. **View results**: See detected defects highlighted in the image
        4. **Analyze statistics**: Check the detection summary on the right
        
        **Tips:**
        - Higher confidence threshold = fewer but more confident detections
        - Lower IOU threshold = more strict object separation
        """)
# Set page config
st.set_page_config(page_title="PCB Defect Detector", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .header {
        color: #ffffff;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="header"> PCB Defect Detection System </div>', unsafe_allow_html=True)
st.write("Upload a PCB image to detect defects using the 'YOLOv12' or 'OpenCV'")
Instructions()

# Load model
@st.cache_resource
def load_model():
    """Load the trained YOLO model"""
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found!")
        return None
    return YOLO(model_path)

model = load_model()

if model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Parameters", help="Adjust parameters of: 'IOU Threshold' and 'Confidence Threshold' to optimize results of the detection.")
    confidence = st.number_input("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
    iou = st.number_input("IOU Threshold", 0.0, 1.0, 0.5, 0.01)
    st.divider()

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("YOLO")
    uploaded_file = st.file_uploader("Attach your PCB image", type=["jpg", "jpeg", "png", "bmp"])

# other model option, if Noaman wanna add it.
with col2:
    st.subheader("Open CV")
    uploaded_file_opencv = st.file_uploader("Use the OpenCV library", type=["jpg", "jpeg", "png", "bmp"], disabled=True, help="This feature is under development m'man.")

# Process image
if uploaded_file is not None:
    # Open with PIL and force RGB (drops alpha if present)
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    
    # Run inference
    with st.spinner("Detecting defects..."):
        results = model.predict(img_array, conf=confidence, iou=iou)
        result = results[0]
        
        # Get annotated image
        annotated_img = result.plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Display results
    col_result, col_stats = st.columns([2, 1])
    
    with col_result:
        st.subheader("Detection Results")
        st.image(annotated_img_rgb, use_column_width=True)
    
    with col_stats:
        st.subheader("Statistics")
        
        # Count detections by class
        if len(result.boxes) > 0:
            st.metric("Total Defects Detected", len(result.boxes))
            
            # Get class names and counts
            class_names = model.names
            detected_classes = result.boxes.cls.cpu().numpy()
            
            for class_id in np.unique(detected_classes):
                class_id = int(class_id)
                count = np.sum(detected_classes == class_id)
                st.metric(f"Class: {class_names[class_id]}", int(count))
            
            # Confidence scores
            st.divider()
            st.write("**Confidence Scores:**")
            confidences = result.boxes.conf.cpu().numpy()
            for i, conf in enumerate(confidences):
                st.write(f"Detection {i+1}: {conf:.2%}")
        else:
            st.info("No defects detected!")
            st.metric("Total Defects Detected", 0)



# Footer
st.divider()
st.caption("PCB Defect Detection | Project Work Machine Vision")