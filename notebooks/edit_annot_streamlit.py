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
    # 1. Base: Raw Data
    data = {}
    if os.path.exists(RAW_JSON_PATH):
        with open(RAW_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # 2. Merge: Modifications (_annotations_fixed.coco.json)
    # If a file has been modified (exists in fixed json), use that version.
    if os.path.exists(FIXED_JSON_PATH):
        try:
            with open(FIXED_JSON_PATH, 'r', encoding='utf-8') as f:
                fixed_data = json.load(f)
                # Update base data with fixed data entries
                for filename, annots in fixed_data.items():
                    data[filename] = annots
        except json.JSONDecodeError:
            pass # Ignore if corrupt
            
    # 3. Merge: Confirmations (FIXED_DICT.json)
    # Confirmed data holds the highest precedence.
    FIXED_DICT_PATH = os.path.join(DATA_DIR, "FIXED_DICT.json")
    if os.path.exists(FIXED_DICT_PATH):
        try:
            with open(FIXED_DICT_PATH, 'r', encoding='utf-8') as f:
                confirmed_data = json.load(f)
                # Override with confirmed data
                for filename, annots in confirmed_data.items():
                    data[filename] = annots
        except json.JSONDecodeError:
            pass

    return data

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

tab_mod, tab_confirm = st.tabs(["[수정] 라벨링 (Modification)", "[컨펌] 검수 (Confirmation)"])

# -----------------------------------------------------------------------------
# TAB 1: [수정] Modification
# -----------------------------------------------------------------------------
with tab_mod:
    # Global Settings for Mod Tab
    c_gs1, c_gs2 = st.columns(2)
    page_size = c_gs1.number_input("페이지당 항목 수 (Items per Page)", min_value=10, max_value=200, value=50, key="mod_page_size")
    filter_unlabeled = c_gs2.checkbox("라벨 없는 항목만 보기 (Unlabeled Only)", value=False, key="mod_filter")

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

    if "mod_current_page" not in st.session_state:
        st.session_state.mod_current_page = 1

    # Pagination
    c_p1, c_p2, c_p3 = st.columns([1, 2, 1])
    if c_p1.button("◀ 이전 (Prev)", key="mod_prev"): 
        st.session_state.mod_current_page = max(1, st.session_state.mod_current_page - 1)
    if c_p3.button("다음 (Next) ▶", key="mod_next"): 
        st.session_state.mod_current_page = min(total_pages, st.session_state.mod_current_page + 1)
    
    c_p2.write(f"**페이지 (Page) {st.session_state.mod_current_page} / {total_pages}** (총 {total_items}개)")


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
        start_idx = (st.session_state.mod_current_page - 1) * page_size
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
                        
                        st.checkbox(f"선택 {i+1}", key=chk_key, label_visibility="collapsed")
                        
                        st.caption(f"{dn if dn else '-'}")
                    else:
                        st.error("Img Error")

        # Submit Button (Applied to Form)
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
                if st.session_state.get(key):
                    item['ann']['label'] = target_class_key
                    item['ann']['drug_N'] = target_drug_n
                    updated_count += 1
            
            if updated_count > 0:
                st.success(f"{updated_count}개 항목 업데이트 완료!")
                save_data(st.session_state.data)
                st.rerun()
            else:
                st.warning("선택된 항목이 없습니다.")


# -----------------------------------------------------------------------------
# TAB 2: [컨펌] Confirmation
# -----------------------------------------------------------------------------
with tab_confirm:
    st.write("### ✅ 라벨링 최종 검수 (Confirmation)")

    # 1. Load FIXED_DICT to see what's already done (optional, but good for persistence)
    FIXED_DICT_PATH = os.path.join(DATA_DIR, "FIXED_DICT.json")
    
    @st.cache_data
    def load_fixed_dict():
        if os.path.exists(FIXED_DICT_PATH):
            try:
                with open(FIXED_DICT_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {} # Return empty dict if file is empty or invalid
        return {}

    fixed_dict = load_fixed_dict()

    def save_fixed_dict(new_dict):
        with open(FIXED_DICT_PATH, 'w', encoding='utf-8') as f:
            json.dump(new_dict, f, indent=4, ensure_ascii=False)
        # load_fixed_dict.clear() # Should update cache if needed, simplified for now
    
    # 2. Filter images that have valid labels
    # We iterate over unique images in 'data'
    valid_images = []
    
    for filename, annots in data.items():
        # Check if ANY annot in this image has a label
        # Also filter out ones that don't allow drawing (invalid bbox?)
        # User constraint: "Only show bboxes if label is not null"
        
        # We need at least one labeled annotation to "Confirm" an image? 
        # Or just show all images and let user confirm? 
        # Requirement: "Composed of results labeled in [Modification] stage"
        
        has_label = any(ann.get('label') is not None for ann in annots)
        if has_label:
            valid_images.append(filename)

    if not valid_images:
        st.info("검수할 이미지가 없습니다. [수정] 탭에서 라벨을 먼저 지정해주세요.")
    else:
        # Pagination for Images
        if "conf_current_page" not in st.session_state:
            st.session_state.conf_current_page = 1
            
        total_conf_items = len(valid_images)
        conf_page_size = 1 # View one image at a time usually better for confirmation, or grid? 
        # User said: "Can see output with bboxes... on AN image" -> implies Single Image View mode might be best.
        
        c_cp1, c_cp2, c_cp3 = st.columns([1, 2, 1])
        if c_cp1.button("◀ 이전 이미지", key="conf_prev"):
            st.session_state.conf_current_page = max(1, st.session_state.conf_current_page - 1)
        if c_cp3.button("다음 이미지 ▶", key="conf_next"):
            st.session_state.conf_current_page = min(total_conf_items, st.session_state.conf_current_page + 1)
            
        current_idx = st.session_state.conf_current_page - 1
        current_filename = valid_images[current_idx]
        
        c_cp2.write(f"**Image {st.session_state.conf_current_page} / {total_conf_items}**")
        c_cp2.caption(f"{current_filename}")

        # Show Confirmation Status
        is_confirmed = current_filename in fixed_dict
        if is_confirmed:
            st.success("✅ 이미 컨펌된 이미지입니다. (Saved in FIXED_DICT)")
        else:
            st.warning("⚠️ 아직 컨펌되지 않았습니다.")

        # Display Logic
        img_path = os.path.join(IMAGE_DIR, current_filename)
        if os.path.exists(img_path):
            try:
                # Open Image
                base_img = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(base_img)
                
                # Draw BBoxes
                annots = data[current_filename]
                
                # Collect valid annotations for saving
                valid_annots_for_save = []

                for ann in annots:
                    lbl = ann.get('label')
                    if lbl: # Only draw if label is present (User Req)
                        bbox = ann['bbox']
                        x, y, w, h = bbox
                        dn = ann.get('drug_N', lbl)
                        
                        # Draw Rectangle
                        draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
                        
                        # Draw Text Background & Text
                        try:
                            font = ImageFont.truetype("Arial", 20)
                        except:
                            font = ImageFont.load_default()
                            
                        # Improve Text Drawing
                        draw.text((x, y-20 if y>20 else y+h), str(dn), fill="red", font=font)
                        
                        valid_annots_for_save.append(ann)
                
                st.image(base_img, use_container_width=True)
                
                # Confirm Button
                # Logic: If checked/clicked, update FIXED_DICT with valid annotations
                conf_btn_label = "Update Confirmation" if is_confirmed else "Confirm & Save"
                if st.button(conf_btn_label, key=f"btn_conf_{current_filename}", type="primary" if not is_confirmed else "secondary"):
                    fixed_dict[current_filename] = valid_annots_for_save
                    save_fixed_dict(fixed_dict)
                    st.toast(f"저장 완료: {current_filename}")
                    load_fixed_dict.clear() # Clear cache to reflect change instantly if we re-read
                    st.rerun()

            except Exception as e:
                st.error(f"Image load error: {e}")
        else:
            st.error(f"File not found: {img_path}")