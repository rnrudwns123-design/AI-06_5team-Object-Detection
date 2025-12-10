import streamlit as st
import json
import os
import math
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# --- Settings ---
ROOT_DIR = os.path.dirname(os.getcwd())
if os.getcwd().endswith("100_DL_ObjectDetection"):
    ROOT_DIR = os.getcwd()

DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "train_images")

RAW_JSON_PATH = os.path.join(ROOT_DIR, "_annotations.coco.json")
FIXED_JSON_PATH = os.path.join(ROOT_DIR, "_annotations_fixed.coco.json")
UNIQUE_CLASSES_PATH = os.path.join(DATA_DIR, "unique_classes.json")

# --- Helper Functions ---

# @st.cache_data
def load_data():
    path = FIXED_JSON_PATH if os.path.exists(FIXED_JSON_PATH) else RAW_JSON_PATH
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_classes():
    if not os.path.exists(UNIQUE_CLASSES_PATH):
        return {}
    with open(UNIQUE_CLASSES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    with open(FIXED_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    st.toast(f"저장 완료: {FIXED_JSON_PATH} (Saved)")

def get_crop(filename, bbox):
    img_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(img_path):
        return None
    try:
        full_img = Image.open(img_path)
        x, y, w, h = bbox
        img_w, img_h = full_img.size
        x = max(0, x); y = max(0, y)
        w = min(w, img_w - x); h = min(h, img_h - y)
        if w > 0 and h > 0:
            return full_img.crop((x, y, x+w, y+h))
    except:
        pass
    return None

def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- Main App ---

st.set_page_config(layout="wide", page_title="알약 라벨링 도구")

if 'data' not in st.session_state:
    st.session_state.data = load_data()

data = st.session_state.data
classes_map = load_classes() 

# Sort Classes by drug_N
sorted_class_keys = sorted(
    classes_map.keys(), 
    key=lambda k: classes_map[k].get('drug_N', '999999')
)

class_display_map = {k: f"{classes_map[k].get('drug_N', '?')}" for k in sorted_class_keys}

# --- Sidebar: Reference Grid (CSS Toggle Hack) ---
st.sidebar.title("📚 참고 이미지 (Reference)")

# CSS for Logic: Checkbox + Label = Toggle
st.sidebar.markdown("""
<style>
.ref-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 3px;
}
.ref-item {
    position: relative;
    border: 1px solid #eee;
    background: white;
    border-radius: 4px;
}
/* Hidden Checkbox */
.toggle-chk {
    display: none;
}
/* Label acts as the click target */
.toggle-label {
    cursor: pointer;
    display: block;
}
.ref-img {
    width: 100%;
    aspect-ratio: 1/1;
    object-fit: contain;
    background: #f0f0f0;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    display: block;
    transition: transform 0.2s;
}
.ref-caption {
    font-size: 9px;
    text-align: center;
    color: #333;
    padding: 1px;
    background: #f8f8f8;
    white-space: nowrap; 
    overflow: hidden; 
    text-overflow: ellipsis;
}

/* Checked State -> Enlarge */
.toggle-chk:checked + .toggle-label {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) scale(5); /* Giant Zoom at Center */
    z-index: 99999;
    box-shadow: 0 0 100px rgba(0,0,0,0.5);
    background: white;
    border-radius: 2px;
    padding: 2px;
}
/* Overlay to darken background when zoomed? Hard with just pure CSS in this structure.
   Instead, we just make the zoomed item huge. */
</style>
""", unsafe_allow_html=True)

st.sidebar.caption("💡 이미지를 클릭하면 확대/축소됩니다.")

html_content = '<div class="ref-grid">'
count = 0 
for key in sorted_class_keys:
    info = classes_map[key]
    dn = info.get('drug_N', '')
    crop = get_crop(info['file_name'], info['bbox'])
    
    if crop:
        crop.thumbnail((100, 100))
        b64 = img_to_base64(crop)
        
        # Unique ID for checkbox
        chk_id = f"chk_{key}"
        
        html_content += f"""<div class="ref-item"><input type="checkbox" id="{chk_id}" class="toggle-chk"><label for="{chk_id}" class="toggle-label" title="{dn}"><img src="data:image/png;base64,{b64}" class="ref-img"><div class="ref-caption">{dn}</div></label></div>"""
    count += 1
html_content += '</div>'
st.sidebar.markdown(html_content, unsafe_allow_html=True)
st.sidebar.caption(f"총 {count}개 클래스")


# --- Main Content ---
st.title("💊 전체 알약 라벨링 도구")

# Global Settings
c_gs1, c_gs2 = st.columns(2)
page_size = c_gs1.number_input("페이지당 항목 수 (Items per Page)", min_value=10, max_value=200, value=50)
filter_unlabeled = c_gs2.checkbox("라벨 없는 항목만 보기 (Unlabeled Only)", value=False)

all_items = []
for filename, annots in data.items():
    for idx, ann in enumerate(annots):
        if filter_unlabeled and ann.get('label') is not None:
            continue
        all_items.append({
            "filename": filename,
            "idx": idx,
            "ann": ann
        })

total_items = len(all_items)
total_pages = math.ceil(total_items / page_size) if total_items > 0 else 1

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# Pagination
c_p1, c_p2, c_p3 = st.columns([1, 2, 1])
if c_p1.button("◀ 이전 (Prev)"): st.session_state.current_page = max(1, st.session_state.current_page - 1)
if c_p3.button("다음 (Next) ▶"): st.session_state.current_page = min(total_pages, st.session_state.current_page + 1)
c_p2.write(f"**페이지 (Page) {st.session_state.current_page} / {total_pages}** (총 {total_items}개)")


# --- Batch Form ---
# We use a FORM to prevent reruns on checkbox selection.
with st.form("batch_action_form"):
    
    # Batch Update Controls
    st.write("### 일괄 적용 (Batch Apply)")
    c_sel, c_empty = st.columns([3, 1])
    with c_sel:
        target_class_key = st.selectbox(
            "적용할 클래스 선택 (Select Class)", 
            options=[""] + sorted_class_keys,
            format_func=lambda x: class_display_map.get(x, "클래스 선택...") if x else "클래스 선택..."
        )
    
    st.divider()

    # Grid Display
    start_idx = (st.session_state.current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_items)
    page_items = all_items[start_idx:end_idx]
    
    # Store selection keys to check after submit
    selection_keys = []

    if not page_items:
        st.info("표시할 항목이 없습니다.")
    else:
        cols = st.columns(5)
        for i, item in enumerate(page_items):
            filename = item['filename']
            idx = item['idx']
            ann = item['ann']
            
            # Unique key for widget
            # Note: We don't rely on persisted session state for checkboxes inside form, 
            # we read the return value of st.checkbox upon submit.
            # But wait, to keep selection across pages? 
            # Forms don't support persistence across pages well if we re-render entirely.
            # However, for ONE page operations, this is fine.
            chk_key = f"chk_{filename}_{idx}"
            selection_keys.append((chk_key, item))
            
            with cols[i % 5]:
                crop = get_crop(filename, ann['bbox'])
                if crop:
                    dn = ann.get('drug_N')
                    # Overlay Logic
                    if dn:
                        draw = ImageDraw.Draw(crop)
                        try:
                            font = ImageFont.truetype("Arial", 14)
                        except:
                            font = ImageFont.load_default()
                        text = str(dn)
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                        draw.rectangle([0, 0, text_w + 4, text_h + 4], fill="black")
                        draw.text((2, 0), text, fill="white", font=font)

                    st.image(crop, use_container_width=True)
                    
                    # Checkbox for selection
                    # If previously selected (in data?), no we don't track that.
                    st.checkbox(f"선택 {i+1}", key=chk_key, label_visibility="collapsed")
                    
                    st.caption(f"{dn if dn else '-'}")
                else:
                    st.error("Img Error")

    # Submit Button (Applied to Form)
    # This renders at the bottom of the form
    st.markdown("<br>", unsafe_allow_html=True)
    apply_submitted = st.form_submit_button("선택 항목에 일괄 적용 (Updates Selected Items)", type="primary")

# --- Process Submission ---
if apply_submitted:
    if not target_class_key:
        st.warning("먼저 적용할 클래스를 선택해주세요.")
    else:
        target_info = classes_map.get(target_class_key, {})
        target_drug_n = target_info.get("drug_N", "Unknown")
        
        updated_count = 0
        
        # Iterate over keys created in the form
        for key, item in selection_keys:
            # Check if widget is True in session state (updated by form submit)
            if st.session_state.get(key):
                item['ann']['label'] = target_class_key
                item['ann']['drug_N'] = target_drug_n
                updated_count += 1
                # Reset checkbox? Hard inside form without rerun.
                # Actually, rerun happens after this block.
        
        if updated_count > 0:
            st.success(f"{updated_count}개 항목 업데이트 완료!")
            save_data(st.session_state.data)
            # Rerun to show updates
            st.rerun()
        else:
            st.warning("선택된 항목이 없습니다.")