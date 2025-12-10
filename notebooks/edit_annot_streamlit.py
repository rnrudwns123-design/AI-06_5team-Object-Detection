import streamlit as st
import json
import os
from PIL import Image, ImageDraw, ImageFont

# --- Settings ---
ROOT_DIR = os.path.dirname(os.getcwd())  # scripts -> root
if os.getcwd().endswith("100_DL_ObjectDetection"):
    ROOT_DIR = os.getcwd()

DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "train_images")

RAW_JSON_PATH = os.path.join(ROOT_DIR, "_annotations.coco.json")
FIXED_JSON_PATH = os.path.join(ROOT_DIR, "_annotations_fixed.coco.json")

# --- Helper Functions ---

def load_data():
    """Loads JSON data. Prioritizes the fixed version if it exists."""
    path = FIXED_JSON_PATH if os.path.exists(FIXED_JSON_PATH) else RAW_JSON_PATH
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_data(data):
    """Saves the modified data to the fixed JSON path."""
    with open(FIXED_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    st.toast(f"저장 완료! ({FIXED_JSON_PATH})")

# --- Main App ---

st.set_page_config(layout="wide", page_title="Simple Label Editor")

if 'data' not in st.session_state:
    st.session_state.data = load_data()

data = st.session_state.data

if data:
    # Sidebar: Image List
    st.sidebar.header("Image List")
    
    # Data structure is now: {"filename.png": [{"label": "x", "bbox": [...]}, ...]}
    image_files = sorted(list(data.keys()))
    
    if not image_files:
        st.warning("No images found in JSON.")
        st.stop()

    selected_filename = st.sidebar.selectbox("Select Image", image_files)
    
    # Resolve Image Path
    # The keys are filenames like "K-003351...png".
    # Assuming they exist in IMAGE_DIR directly.
    image_path = os.path.join(IMAGE_DIR, selected_filename)
    
    # Annotations for this image
    current_annots = data[selected_filename]

    # --- Main Content ---
    col_img, col_ui = st.columns([3, 2])

    with col_img:
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()

            for i, ann in enumerate(current_annots):
                bbox = ann['bbox'] # [x, y, w, h]
                x, y, w, h = bbox
                label = ann.get('label')
                
                # Draw Box
                rect = [x, y, x+w, y+h]
                color = "green" if label else "red"
                draw.rectangle(rect, outline=color, width=3)
                
                # Draw Label
                display_text = f"{i}: {label}" if label else f"{i}: (None)"
                draw.rectangle([x, y-15, x+100, y], fill=color)
                draw.text((x+2, y-15), display_text, fill="white", font=font)
            
            st.image(img, use_container_width=True, caption=f"{selected_filename}")
        else:
            st.error(f"Image not found at: {image_path}")

    with col_ui:
        st.subheader("Edit Annotations")
        st.info(f"Found {len(current_annots)} annotations.")

        with st.form("edit_form"):
            st.write("Target Annotations:")
            
            selected_indices = []
            for i, ann in enumerate(current_annots):
                label = ann.get('label', 'None')
                c1, c2 = st.columns([1, 10])
                with c1:
                    if st.checkbox(f"{i}", key=f"chk_{selected_filename}_{i}"):
                        selected_indices.append(i)
                with c2:
                    st.text(f"Box {i} : {label}")
            
            st.divider()
            
            st.write("Apply New Label")
            new_label = st.text_input("Enter Label Name", placeholder="e.g. car, person")
            
            if st.form_submit_button("Update Selected Labels"):
                if not new_label:
                    st.warning("Please enter a label name.")
                elif not selected_indices:
                    st.warning("No boxes selected.")
                else:
                    count = 0
                    for idx in selected_indices:
                        current_annots[idx]['label'] = new_label
                        count += 1
                    
                    st.success(f"Updated {count} annotations to '{new_label}'")
                    st.session_state.data = data # Ensure session sync
                    st.rerun()

        st.divider()
        if st.button("Save Changes to Disk", type="primary"):
            save_data(st.session_state.data)
            st.success("Saved successfully!")