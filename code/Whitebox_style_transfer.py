import base64
import datetime
import os
import sys
from io import BytesIO
from pathlib import Path
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
import time

PACKAGE_PARENT = 'wise'
# os.path.expanduser() 用來把shell中的~/xxx展開，~即當前user的home directory
# __file__是當前所運行的python file路徑
# os.path.join() 拼接路徑，如果接的是絕對路徑，就會捨棄掉前面接好的
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# os.path.normpath() 用來去掉路經中多餘的斜線
# sys.path is a list of directories where the Python interpreter searches for modules
# 所以這邊的操作只是為了用wise資料夾下的files而已
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import streamlit as st
from streamlit.logger import get_logger
# https://github.com/vivien000/st-click-detector
# st-click-detector is a Streamlit component to display some HTML content and detect when hyperlinks are clicked on.
from st_click_detector import click_detector
import streamlit.components.v1 as components
from streamlit.source_util import get_pages
# https://github.com/arnaudmiribel/streamlit-extras
from streamlit_extras.switch_page_button import switch_page

from demo_config import HUGGING_FACE
from parameter_optimization.parametric_styletransfer import single_optimize
from parameter_optimization.parametric_styletransfer import CONFIG as ST_CONFIG
from parameter_optimization.strotss_org import strotss, pil_resize_long_edge_to
import helpers.session_state as session_state
from helpers import torch_to_np, np_to_torch
from effects import get_default_settings, MinimalPipelineEffect 

# Configures the default settings of the page.
# This must be the first Streamlit command used in your app, and must only be set once.
# layout: How the page content should be laid out.
st.set_page_config(layout="wide")
# 網頁中的默認可用圖片的來源網址
BASE_URL = "https://ivpg.hpi3d.de/wise/wise-demo/images/"
# __name__ is a built-in variable which evaluates to the name of the current module
# If the source file is executed as the main program, the interpreter sets the __name__ variable to have a value “__main__”
# If this file is being imported from another module, __name__ will be set to the module’s name.
# A good convention to use when naming loggers is to use a module-level logger, in each module which uses logging
# This means that logger names track the package/module hierarchy, and it’s intuitively obvious where events are logged just from the logger name.
# 其實我覺得這個LOGGER沒什麼用處
LOGGER = get_logger(__name__)

effect_type = "minimal_pipeline"

# Session State is a way to share variables between reruns, for each user session
# 基本上session_state就是一個dict，可存儲任何對象
# session state只可能在application初始化時被整體重置。而rerun(不會重置session state。可以理解成“狀態緩存”

# Initialize values in Session State
# 
if "click_counter" not in st.session_state:
    # Session State also supports attribute based syntax
    st.session_state.click_counter = 1

if "action" not in st.session_state:
    # The Session State API follows a field-based API, which is very similar to Python dictionaries:
    st.session_state["action"] = ""
    
if "user" not in st.session_state:
    # hash(time.time()) 只是為了創 user ID
    st.session_state["user"] = hash(time.time())

if "transferMode" not in st.session_state:
    # hash(time.time()) 只是為了創 user ID
    st.session_state["transferMode"] = "STROTSS"

# 默認可用的content image們
content_urls = [
    {
        "name": "Portrait", "id": "portrait",
        "src": BASE_URL + "/content/portrait.jpeg"
    },
    {
        "name": "Tuebingen", "id": "tubingen",
        "src": BASE_URL + "/content/tubingen.jpeg"
    },
    {
        "name": "Colibri", "id": "colibri",
        "src": BASE_URL + "/content/colibri.jpeg"
    }
]

# 默認可用的content image們
style_urls = [
    {
        "name": "Starry Night, Van Gogh", "id": "starry_night",
        "src": BASE_URL + "/style/starry_night.jpg"
    },
    {
        "name": "The Scream, Edward Munch", "id": "the_scream",
        "src": BASE_URL + "/style/the_scream.jpg"
    },
    {
        "name": "The Great Wave, Ukiyo-e", "id": "wave",
        "src": BASE_URL + "/style/wave.jpg"
    },
    {
        "name": "Woman with Hat, Henry Matisse", "id": "woman_with_hat",
        "src": BASE_URL + "/style/woman_with_hat.jpg"
    }
]


def last_image_clicked(type="content", action=None, ):
    kw = "last_image_clicked" + "_" + type
    if action:
        # **是用來將dict內容展開成參數們
        # session_state是作者自己寫的
        # 以下等同session_state.get(kw=action)
        # get(xxx:yyy)應該是用來新增&更新xxx:yyy
        session_state.get(**{kw: action})
    elif kw not in session_state.get():
        return None
    else:
        return session_state.get()[kw]

# 對function加了@st.cache的話
# 如果之後call該function，就會把結果cache起來
# 下次再用一模一樣的參數call該function時，不會真的執行該function，而是拿local cache的結果出來
# 當然如果用的是不同的參數，就會執行function
@st.cache
def _retrieve_from_id(clicked, urls):
    # 取得id對應的img網址
    src = [x["src"] for x in urls if x["id"] == clicked][0]
    # If we set stream=True in requests.get(...) then headers['Transfer-Encoding'] = 'chunked' is set in the HTTP headers. 
    # Thus specifying the Chunked transfer encoding. In chunked transfer encoding, the data stream is divided into a series of non-overlapping "chunks". 
    # The chunks are sent out independently of one another by the server
    # raw: get the raw socket response from the server (stream必須為True)
    img = Image.open(requests.get(src, stream=True).raw)
    return img, src


def store_img_from_id(clicked, urls, imgtype):
    img, src = _retrieve_from_id(clicked, urls)
    # 也是新增&更新session_state
    session_state.get(**{f"{imgtype}_im": img, f"{imgtype}_render_src": src, f"{imgtype}_id": clicked})


def img_choice_panel(imgtype, urls, default_choice, expanded):
    # imgtype是Content或Style

    # 在result_image_placeholder下方的新container
    # st.expander() Inserts a container into your app that can be used to hold multiple elements and can be expanded or collapsed by the user.
    # expanded為True，所以container一開始就是展開的
    with st.expander(f"Select {imgtype} image:", expanded=expanded):
        # 用來顯示預設可選圖片的html code
        html_code = '<div class="column" style="display: flex; flex-wrap: wrap; padding: 0 4px;">'
        for url in urls:
            html_code += f"<a href='#' id='{url['id']}' style='padding: 0px 5px'><img height='160px' style='margin-top: 8px;' src='{url['src']}'></a>"
        html_code += "</div>"
        # content to display and from which clicks should be detected
        # clicked是點到的html物件的id
        clicked = click_detector(html_code)

        # 默認選擇的content & style images
        # 沒有click特定link時clicked應該就是None
        if not clicked and st.session_state["action"] not in ("uploaded", "switch_page_from_local_edits", "switch_page_from_presets", "slider_change", "reset"):  # default val
            store_img_from_id(default_choice, urls, imgtype)

        # write(string) : Prints the formatted Markdown string
        # 在expander中，寫一行OR:
        st.write("OR:  ")

        # A form is a container that visually groups other elements and widgets together, and contains a Submit button. 
        # When the form's Submit button is pressed, all widget values inside the form will be sent to Streamlit in a batch.
        with st.form(imgtype + "-form", clear_on_submit=True):
            uploaded_im = st.file_uploader(f"Load {imgtype} image:", type=["png", "jpg", "jpeg"], )
            upload_pressed = st.form_submit_button("Upload")

            if upload_pressed and uploaded_im is not None:
                img = Image.open(uploaded_im)
                img = img.convert('RGB')
                # 用來操作bytes type的數據
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                # decode()是默認以utf-8來解碼bytes object
                # 這個是為了在網頁上顯示上傳的照片
                encoded = base64.b64encode(buffered.getvalue()).decode()
                # session_state.get(uploaded_im=img, content_render_src=f"data:image/jpeg;base64,{encoded}")
                session_state.get(**{f"{imgtype}_im": img, f"{imgtype}_render_src": f"data:image/jpeg;base64,{encoded}",
                                     f"{imgtype}_id": "uploaded"})
                st.session_state["action"] = "uploaded"
                st.write("uploaded.")

        last_clicked = last_image_clicked(type=imgtype)
        print(st.session_state["user"], " last_clicked", last_clicked, "clicked", clicked, "action", st.session_state["action"] )
        if not upload_pressed and clicked != "":  # trigger when no file uploaded
            if last_clicked != clicked:  # only activate when content was actually clicked
                store_img_from_id(clicked, urls, imgtype)
                last_image_clicked(type=imgtype, action=clicked)
                st.session_state["action"] = "clicked"
                st.session_state.click_counter += 1  # hack to get page to reload at top


        state = session_state.get()
        # Add widgets to sidebar
        st.sidebar.write(f'Selected {imgtype} image:')
        # 圖片顯示在sidebar
        st.sidebar.markdown(f'<img src="{state[f"{imgtype}_render_src"]}" width=240px></img>', unsafe_allow_html=True)
        


def optimize(effect, preset, result_image_placeholder):
    # 取得images，是Image.open()後的
    content = st.session_state["Content_im"]
    style = st.session_state["Style_im"]
    # 設回False
    st.session_state["optimize_next"] = False
    # Temporarily displays a message while executing a block of code (會有個loading那樣的轉圈圈icon)
    # 當with內的code跑完，text就會消失
    with st.spinner(text="Optimizing parameters.."):
        print("optimizing for user", st.session_state["user"])
        # 是local端，所以不會跑這個
        if HUGGING_FACE:
            optimize_on_server(content, style, result_image_placeholder)
        # 跑這個
        else:
            # 除了在網頁上顯示外，結果也會存到result/
            optimize_params(effect, preset, content, style, result_image_placeholder)

# optimize()的前置顯示
def optimize_next(result_image_placeholder):
    # Write fixed-width and preformatted text.
    result_image_placeholder.text("<- Custom content/style needs to be style transferred")
    # 在local端跑的，HUGGING_FACE為False，所以就是queue_length = 0
    queue_length = 0 if not HUGGING_FACE else get_queue_length()
    if queue_length > 0:
        st.sidebar.warning(f"WARNING: Already {queue_length} tasks in the queue. It will take approx {(queue_length+1) * 5} min for your image to be completed.")
    else:
        # Display warning message.
        st.sidebar.warning("Note: Optimizing takes up to 5 minutes.")
    # Display a button widget.
    optimize_button = st.sidebar.button("Optimize Style Transfer")
    # 如果按了button
    if optimize_button:
        st.session_state["optimize_next"] = True
        # When st.experimental_rerun() is called, the script is halted - no more statements will be run, 
        # and the script will be queued to re-run from the top.
        # 也就是該python script會重新run一次吧
        st.experimental_rerun()
    else:
        # result_vp會在optimize_params()後加進st.session_state
        if not "result_vp" in st.session_state:
            # Streamlit will not run any statements after st.stop()
            st.stop()
        else:
            return st.session_state["effect_input"], st.session_state["result_vp"]


# To override the streamlit's default hashing behavior, pass a custom hash function. 
# You can do that by mapping a type (e.g. MinimalPipelineEffect) to a hash function (id)
# 下面的這個id應該是python 的 id function
@st.cache(hash_funcs={MinimalPipelineEffect: id})
def create_effect():
    # effect_type 是 "minimal_pipeline"
    # get_default_settings() 定義再 effects資料夾中的__init__.py
    # effect 是 MinimalPipelineEffect class的instance
    # preset 是 一個list，裝著各項可調slider的名稱及其默認數值的tuple
    # param_set 是 各項可調slider的名稱
    effect, preset, param_set = get_default_settings(effect_type)
    # enable_checkpoints()是從EffectBase繼承來的
    effect.enable_checkpoints()
    # cuda()是從torch.nn.Module繼承來的
    effect.cuda()
    return effect, preset

# -> 是 function annotation
# 只是在提示該function的return type 是 torch.Tensor
def load_visual_params(vp_path: str, img_org: Image, org_cuda: torch.Tensor, effect) -> torch.Tensor:
    if Path(vp_path).exists():
        vp = torch.load(vp_path).detach().clone()
        vp = F.interpolate(vp, (img_org.height, img_org.width))
        if len(effect.vpd.vp_ranges) == vp.shape[1]:
            return vp
    # use preset and save it
    vp = effect.vpd.preset_tensor(preset, org_cuda, add_local_dims=True)
    torch.save(vp, vp_path)
    return vp


# @st.cache(hash_funcs={torch.Tensor: id})
# @st.experimental_memo: Function decorator to memoize function executions.
# Memoized data is stored in "pickled" form, which means that the return value of a memoized function must be pickleable.
@st.experimental_memo
def load_params(content_id, style_id):#, effect):
    # 網頁上的預設圖片都預先算好了
    preoptim_param_path = os.path.join("precomputed", effect_type, content_id, style_id)
    img_org = Image.open(os.path.join(preoptim_param_path, "input.png"))
    content_cuda = np_to_torch(img_org).cuda()
    vp_path = os.path.join(preoptim_param_path, "vp.pt")
    vp = load_visual_params(vp_path, img_org, content_cuda, effect)
    return content_cuda, vp

def render_effect(effect, content_cuda, vp):
    with torch.no_grad():
        result_cuda = effect(content_cuda, vp)
    img_res = Image.fromarray((torch_to_np(result_cuda) * 255.0).astype(np.uint8))
    return img_res


# Inserts an invisible container (container本身是看不到的) into your app that can be used to hold multiple elements (elements看的到). 
result_container = st.container()
# coll1佔container的3/5，coll2佔container的2/5
coll1, coll2 = result_container.columns([3,2])
# Display text in header formatting.
# 預計用來放結果圖的，左邊會是transfered後的圖，右邊會是各種可調的sliders
coll1.header("Result")
coll2.header("Global Edits")
# Inserts a container into your app that can be used to hold a single element.
result_image_placeholder = coll1.empty()
# Display string formatted as Markdown.
result_image_placeholder.markdown("## loading..")

from tasks import optimize_on_server, optimize_params, monitor_task, get_queue_length

if "current_server_task_id" not in st.session_state:
    st.session_state['current_server_task_id'] = None

if "optimize_next" not in st.session_state:
    st.session_state['optimize_next'] = False

# effect 是 MinimalPipelineEffect class的instance
# preset 是 一個list，裝著各項可調slider的名稱及其默認數值的tuple
effect, preset = create_effect()

# 因為是local端，HUGGING_FACE是false
# 所以這串都可以先不管
if HUGGING_FACE and st.session_state['current_server_task_id'] is not None: 
    with st.spinner(text="Optimizing parameters.."):
        monitor_task(result_image_placeholder)

# optimize_button按下後，st.session_state["optimize_next"]就會變True
if st.session_state["optimize_next"]:
    print("optimize now")
    optimize(effect, preset, result_image_placeholder)

# 選擇(或上傳)content & style images
img_choice_panel("Content", content_urls, "portrait", expanded=True)
img_choice_panel("Style", style_urls, "starry_night", expanded=True)

state = session_state.get()
content_id = state["Content_id"]
style_id = state["Style_id"]


print("content id, style id", content_id, style_id  )
# content & style圖片至少一方自己上傳的情況(算完當下，action仍是uploaded)
if st.session_state["action"] == "uploaded":
    content_img, _vp = optimize_next(result_image_placeholder)
# 手動調整結果時、或者在上面if運算完後，將content或style image其中一個換成默認圖時，
elif st.session_state["action"] in ("switch_page_from_local_edits", "switch_page_from_presets", "slider_change") or \
      content_id == "uploaded" or style_id == "uploaded":
    print(st.session_state["user"], "restore param")
    _vp = st.session_state["result_vp"]
    content_img = st.session_state["effect_input"]
# 如果求的都是網頁上已經算好的圖
else:
    print(st.session_state["user"], "load_params")
    content_img, _vp = load_params(content_id, style_id)#, effect)
print("st.session_state",st.session_state["action"])

vp = torch.clone(_vp)


def reset_params(means, names):
    for i, name in enumerate(names):
        st.session_state["slider_" + name] = means[i]

def on_slider():
    st.session_state["action"] = "slider_change"


# 各種可調的sliders
with coll2:
    # bumpiness是一些筆觸明顯程度
    # bumpSpecular是筆觸的反光程度(就是筆觸會帶白色高光，看起來會像版畫一樣，較有立體感)
    # bumpSpecular在bumpiness不強時，其實看不太出來
    # contours是輪廓線明顯程度(黑色)
    show_params_names = [ 'bumpiness',"bumpSpecular", "contours"]
    display_means = []
    params_mapping = {"bumpiness": ['bumpScale', "bumpOpacity"], "bumpSpecular": ["bumpSpecular"], "contours": [ "contourOpacity", "contour"]}
    def create_slider(name):
        params = params_mapping[name] if name in params_mapping else [name]
        means = [torch.mean(vp[:, effect.vpd.name2idx[n]]).item() for n in params]
        display_mean = np.average(means) + 0.5
        print(display_mean)
        # display_means存的會是上次的值(而非當前slider的值)
        display_means.append(display_mean)
        if "slider_" + name not in st.session_state or st.session_state["action"] != "slider_change": 
          st.session_state["slider_" + name] = display_mean
        # Display a slider widget.
        # This supports int, float, date, time, and datetime types.
        # 似乎只要使用者動了slider，script就會rerun
        # 似乎slider的默認值是看st.session_state["slider_" + name]裡所存的值
        slider = st.slider(f"Mean {name}: ", 0.0, 1.0, step=0.01, key="slider_" + name, on_change=on_slider)
        for i, param_name in enumerate(params):
            vp[:, effect.vpd.name2idx[param_name]] += slider - (means[i] + 0.5)
            vp.clamp_(-0.5, 0.5)
    
    for name in show_params_names:
        create_slider(name)

    others_idx = set(range(len(effect.vpd.vp_ranges))) - set([effect.vpd.name2idx[name] for name in sum(params_mapping.values(), [])])
    others_names = [effect.vpd.vp_ranges[i][0] for i in sorted(list(others_idx))]
    # others_names有hueShift(改圖片色調)、colorfulness(感覺像飽和度，越少越黑白)、luminosityOffset(感覺像明度)、contrast
    # Display a select widget.
    other_param = st.selectbox("Other parameters: ", ["hueShift"] + [n for n in others_names if n != "hueShift"] )
    create_slider(other_param)


    reset_button = st.button("Reset Parameters", on_click=reset_params, args=(display_means, show_params_names))
    if reset_button:
        st.session_state["action"] = "reset"
        st.experimental_rerun()

    apply_presets = st.button("Paint Presets")
    edit_locally_btn = st.button("Edit Local Parameter Maps")
    if edit_locally_btn:
        switch_page("Local_edits")

    if apply_presets:
        switch_page('Apply_preset')



img_res = render_effect(effect, content_img, vp)

# st.session_state["result_vp"] = vp
# 我加的
st.session_state["result_vp"] = _vp
st.session_state["result_vp_"] = vp
st.session_state["effect_input"] = content_img
st.session_state["last_result"] = img_res

with coll1:
    # width = int(img_res.width * 500 / img_res.height)
    result_image_placeholder.image(img_res)#, width=width)
    option = st.selectbox('Which Style Transfer Model ?',('STROTSS', 'AesUST'))
    st.session_state["transferMode"] = option
    
    

# a bit hacky way to return focus to top of page after clicking on images
components.html(
    f"""
        <p>{st.session_state.click_counter}</p>
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
    """,
    height=0
)

