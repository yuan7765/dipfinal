import os
import sys
import torch.nn.functional as F
import torch
import time

PACKAGE_PARENT = '..'
WISE_DIR = '../wise/'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, WISE_DIR)))


import numpy as np
from PIL import Image
import streamlit as st
# https://github.com/andfanilo/streamlit-drawable-canvas
from streamlit_drawable_canvas import st_canvas

from effects.minimal_pipeline import MinimalPipelineEffect
from helpers.visual_parameter_def import minimal_pipeline_presets, minimal_pipeline_bump_mapping_preset, minimal_pipeline_xdog_preset
from helpers import torch_to_np, np_to_torch
from effects import get_default_settings
from demo_config import HUGGING_FACE


st.set_page_config(page_title="Combine Two Style Demo", layout="wide")

st.session_state["action"] = "switch_page_from_presets" # on switchback, remember effect input


mix_intensity = st.sidebar.slider("Mix Intensity : ", 0.0, 1.0, 1.0, 0.01)
edge_intensity = st.sidebar.slider("Edge Intensity : ", 0.0, 1.0, 0.5, 0.01)

st.sidebar.text("Drawing options:")
stroke_width = st.sidebar.slider("Stroke width: ", 1, 80, 40)
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)

def DefaultMainImage():
    img = Image.open("./main.jpg")
    img = img.convert('RGB')
    st.session_state["main_img_for_combine"] = img
    st.session_state["main_cuda_for_combine"] = np_to_torch(st.session_state["main_img_for_combine"]).cuda()

def DefaultBGImage():
    img = Image.open("./bg.jpg")
    img = img.convert('RGB')
    st.session_state["bg_img_for_combine"] = img
    # resize image
    st.session_state["bg_cuda_for_combine"] = F.interpolate(np_to_torch(st.session_state["bg_img_for_combine"]).cuda(), (st.session_state["main_cuda_for_combine"] .size(2), st.session_state["main_cuda_for_combine"] .size(3)))


st.session_state["preset_canvas_key"] ="preset_canvas"
if "bg_img_for_combine" not in st.session_state:
    st.session_state["bg_img_for_combine"] = None
if "main_img_for_combine" not in st.session_state:
    st.session_state["main_img_for_combine"] = None
if "main_cuda_for_combine" not in st.session_state:
    DefaultMainImage()
if "bg_cuda_for_combine" not in st.session_state:
    DefaultBGImage()
if "img_id" not in st.session_state:
    st.session_state["img_id"] = "default"


coll1, coll2 = st.columns(2)
coll1.header("Draw Mask")
coll2.header("Live Result")

def greyscale_original(_main_cuda): #content_id is used for hashing
    if HUGGING_FACE:
        wsize = 450
        img_org_height, img_org_width = _main_cuda.shape[-2:]
        wpercent = (wsize / float(img_org_width))
        hsize = int((float(img_org_height) * float(wpercent)))
    else:
        # longest_edge = 670
        # img_org_height, img_org_width = _org_cuda.shape[-2:]
        # max_width_height = max(img_org_width, img_org_height)
        # hsize = int((float(longest_edge) * float(float(img_org_height) / max_width_height)))
        # wsize = int((float(longest_edge) * float(float(img_org_width) / max_width_height)))
        wsize = 450
        img_org_height, img_org_width = _main_cuda.shape[-2:]
        wpercent = (wsize / float(img_org_width))
        hsize = int((float(img_org_height) * float(wpercent)))

    gray_img = F.interpolate(_main_cuda, (hsize, wsize), mode="bilinear")
    gray_img = torch.mean(gray_img, dim=1, keepdim=True) / 2.0
    gray_img = torch_to_np(gray_img, multiply_by_255=True)[..., np.newaxis].repeat(3, axis=2)
    gray_img = Image.fromarray(gray_img.astype(np.uint8))
    return gray_img, hsize, wsize

greyscale_img, hsize, wsize = greyscale_original(st.session_state["main_cuda_for_combine"])
    
with coll1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        background_image=greyscale_img,
        width=greyscale_img.width,
        height=greyscale_img.height,
        drawing_mode=drawing_mode,
        key = st.session_state["img_id"]
    )
    
mask_cuda = None
mask = None
final = None
# canvas_result.image_data 是 筆畫圖
if canvas_result.image_data is not None:
    mask_cuda = np_to_torch(canvas_result.image_data.astype(np.float32)).sum(dim=1, keepdim=True).cuda()
    img_org_width = st.session_state["main_cuda_for_combine"].shape[-1] 
    img_org_height = st.session_state["main_cuda_for_combine"].shape[-2] 
    mask = F.interpolate(mask_cuda, (img_org_height, img_org_width)).squeeze(1)

with torch.no_grad():
    if mask != None:
        # mask1 為 1的部分是 background，為0的部分是 main
        # mask2 為 1的部分是 main，為0的部分是 background
        mask2 = mask.clone()
        mask1 = mask.clone()
        mask1[(mask!=0).logical_and(mask!=1)] = edge_intensity
        mask1[mask==1] = mix_intensity
        mask2[mask==1] = 0
        mask2[mask==0] = 1
        mask2[mask2==0] = 1-mix_intensity
        mask2[(mask!=0).logical_and(mask!=1)] = 1-edge_intensity
        final = st.session_state["bg_cuda_for_combine"]*mask1 + st.session_state["main_cuda_for_combine"]*mask2 

if final != None:
    img_res = Image.fromarray((torch_to_np(final) * 255.0).astype(np.uint8))
    coll2.image(img_res)


# apply_btn = st.sidebar.button("Apply")
# if apply_btn:
#     st.session_state["result_vp"] = vp

st.info("Note: Press apply to make changes permanent")


def resizeBG2Main():
    main_cuda = np_to_torch(st.session_state["main_img_for_combine"]).cuda()
    bg_cuda = F.interpolate(np_to_torch(st.session_state["bg_img_for_combine"]).cuda(), (main_cuda.size(2), main_cuda.size(3)))
    st.session_state["main_cuda_for_combine"] = main_cuda
    st.session_state["bg_cuda_for_combine"] = bg_cuda
    greyscale_img, hsize, wsize = greyscale_original(st.session_state["main_cuda_for_combine"])
    st.session_state["img_id"] = hash(time.time())
    st.experimental_rerun()

with st.form("Main_img", clear_on_submit=True):
    uploaded_im = st.file_uploader(f"Load Main image:", type=["png", "jpg"], )
    upload_pressed = st.form_submit_button("Upload")
    if upload_pressed and uploaded_im is not None:
        img = Image.open(uploaded_im)
        img = img.convert('RGB')
        st.session_state["main_img_for_combine"] = img
        resizeBG2Main()

# 已經做好風格轉換的background
with st.form("Background_img", clear_on_submit=True):
    uploaded_im = st.file_uploader(f"Load BG image:", type=["png", "jpg"], )
    upload_pressed = st.form_submit_button("Upload")
    if upload_pressed and uploaded_im is not None:
        img = Image.open(uploaded_im)
        img = img.convert('RGB')
        st.session_state["bg_img_for_combine"] = img
        resizeBG2Main()



        