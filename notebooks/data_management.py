# 학습 데이터셋 설정

import json
import os
from datetime import datetime

ROOT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "train_images")
ANNOT_DIR = os.path.join(DATA_DIR, "train_annotations")


full_dict_path = os.path.join(DATA_DIR, "FULL_DICT.json")
err_txt_path = os.path.join(DATA_DIR, "err_image_paths.txt")
fixed_dict_path = os.path.join(DATA_DIR, "FIXED_DICT.json")
partial_dict_path = os.path.join(DATA_DIR, "PARTIAL_DICT.json")


# FULL_DICT 불러오기
def load_full_dict() -> dict:
    with open(full_dict_path, "r", encoding="utf-8") as f:
        FULL_DICT = json.load(f)

    modification_time = os.path.getmtime(full_dict_path)
    modification_time = datetime.fromtimestamp(modification_time)

    print(f"[FULL_DICT]")
    print(f"· {len(FULL_DICT)}개")
    print(f"· 업데이트 날짜: {modification_time}")

    return FULL_DICT


# PARTIAL_DICT 불러오기
def load_partial_dict() -> dict:
    with open(partial_dict_path, "r", encoding="utf-8") as f:
        PARTIAL_DICT = json.load(f)

    modification_time = os.path.getmtime(partial_dict_path)
    modification_time = datetime.fromtimestamp(modification_time)

    print(f"[PARTIAL_DICT]")
    print(f"· {len(PARTIAL_DICT)}개")
    print(f"· 업데이트 날짜: {modification_time}")

    return PARTIAL_DICT


# err_image_paths 불러오기
def load_err_image_paths() -> list:
    try:
        with open(err_txt_path, "r", encoding="utf-8") as f:
            err_image_paths = f.read().split()

        modification_time = os.path.getmtime(err_txt_path)
        modification_time = datetime.fromtimestamp(modification_time)

        print(f"[err_image_paths]")
        print(f"· {len(err_image_paths)}개")
        print(f"· 업데이트 날짜: {modification_time}")

        return err_image_paths
    except:
        pass


# FIXED_DICT 불러오기
def load_fixed_dict() -> dict:
    try:
        with open(fixed_dict_path, "r", encoding="utf-8") as f:
            FIXED_DICT = json.load(f)

        modification_time = os.path.getmtime(fixed_dict_path)
        modification_time = datetime.fromtimestamp(modification_time)

        print(f"[FIXED_DICT]")
        print(f"· {len(fixed_dict_path)}개")
        print(f"· 업데이트 날짜: {modification_time}")

        return FIXED_DICT
    except:
        pass


# FINAL_DICT 불러오기
def load_final_dict() -> dict:

    FINAL_DICT = {}

    # FULL_DICT 더하기
    try: 
        FULL_DICT = load_full_dict()
    except: 
        pass

    for key, value in FULL_DICT.items():
        image_path = os.path.join(IMAGE_DIR, key)

        tmp_list = []

        for json_path in value:
            tmp_list.append(os.path.join(ANNOT_DIR, json_path))
            
        FINAL_DICT[image_path] = tmp_list
        

    # err_image_paths 빼기
    try:
        err_image_paths = load_err_image_paths()

        for err in err_image_paths:
            err_path = os.path.join(IMAGE_DIR, err)
            del FINAL_DICT[err_path]
        
    except:
        pass


    # FINAL_DICT 서식 수정
    for image_path, annot_paths in FINAL_DICT.items():

        tmp_list = []

        for path in annot_paths:
            with open(path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            xywh_bbox = json_data["annotations"][0]["bbox"]

            tmp_list.append({"bbox": xywh_bbox,
                            "label": json_data["categories"][0]["id"]})
            
        FINAL_DICT[image_path] = tmp_list


    # FIXED_DICT 더하기
    try:
        FIXED_DICT = load_fixed_dict()

        for key, value in FIXED_DICT.items():

            tmp_list = []

            for di in value:
                del di["drug_N"]

            FINAL_DICT[os.path.join(IMAGE_DIR, key)] = value
        
    
    except:
        pass

    print(f"\n· FINAL_DICT 불러오기 완료 ({len(FINAL_DICT)}개)")

    return FINAL_DICT


# def common_to_personal(data):
#     if type(data) == dict:
#         new_dict = {}

#         for key, value in data.items():
#             new_key = os.path.join(IMAGE_DIR, key)
#             new_value_list = []
            
#             for item in value:
#                 new_item = os.path.join(ANNOT_DIR, item)
#                 new_value_list.append(new_item)

#             new_dict[new_key] = new_value_list

#         return new_dict

#     else:
#         for value in data:
#             pass

# def personal_to_common(data):
#     pass
