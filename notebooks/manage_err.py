import os
import json
import torchvision
import matplotlib.pyplot as plt
from matplotlib import patches
from IPython import display

# 기본 경로 설정
ROOT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "train_images")
ANNOT_DIR = os.path.join(DATA_DIR, "train_annotations")

full_dict_path = os.path.join(DATA_DIR, "FULL_DICT.json")
err_txt_path = os.path.join(DATA_DIR, "err_image_paths.txt")
fixed_dict_path = os.path.join(DATA_DIR, "FIXED_DICT.json")


import data_preparation as dp
import shutil

def find_err():
    # 육안으로 오류 식별
    # 식별된 오류는 err_image_path.txt에 추가
    pass

def errs_to_desktop():
    # 고쳐보면서 코드 써야 할 듯. 뭐가 필요한지 모르겠음.
    # 고칠 err 이미지들 바탕화면으로 복사

    err_image_paths = dp.load_err_image_paths()

    for path in err_image_paths:
        from_path = os.path.join(IMAGE_DIR, path)
        to_path = os.path.join(ROOT_DIR, path)
        shutil.copyfile(from_path, to_path)

def edit_dicts():
    # err_image_paths.txt에서 삭제
    # FIXED_DICT.json에 추가
    pass