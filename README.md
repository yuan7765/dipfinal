# DIP Final
DIP Final Report
We would like to combine different methods to form a better result for style transferring tdtm, aka 武士雞, and give stable outputs for videos. 
## File Structure

## Methods Used
1. AesUST from [Official Pytorch code for AesUST](https://github.com/EndyWon/AesUST)
2. WISE from [White-box Style Transfer Editing (WISE)](https://github.com/winfried-ripken/wise)
3. STROTSS from [Memory Efficient Version Of Strotts](https://github.com/futscdav/strotss)
4. Fast Artistic Videos from [Fast Artistic Videos in pyTorch](https://github.com/pgalatic/fast-artistic-videos-pytorch)
5. ReReVST from [ReReVST-Code](https://github.com/daooshee/ReReVST-Code?fbclid=IwAR0cMbVQ100brf97DcybltNrZ6bEGjxAg769LZP0rWLnGM6VYfHgRvGWwFM)
## How to use Streamlit
### Prerequisites
We use Python 3.10 (In Anaconda environment)  
Please run the below code to install the required packages.
```
pip install imageio
pip install imageio-ffmpeg
pip install matplotlib
pip install Pillow
pip install numpy
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install streamlit==1.10.0
pip install streamlit_drawable_canvas==0.8.0
pip install streamlit-extras==0.1.5
pip install st-click-detector
pip install scipy
pip install huggingface-hub
pip install segmentation-models-pytorch
```
### launch the Streamlit Interface
Run below command in [code/](https://github.com/yuan7765/dipfinal/tree/main/code)
```
streamlit run Whitebox_style_transfer.py
```
### Whitebox style transfer Page

1. Choose the style transfer model in selectbox (STROTSS or AesUST).
2. Upload your style & content images from local, and the two images will show in sidebar.
3. Press the Optimize Style Transfer Button in sidebar.
4. Wait about 3 minutes for style transfer and parameter values optimization.

> **Warning**  
> - The default style & content images (like starry night & girl portrait images) are provided by WISE'authors and only precomputed for STROTSS.  
> - If you want to choose default images for style or content, make sure that you choose default image first, then upload your own image. Otherwise, Optimize Style Transfer Button in sidebar won't show up.  
> - You can not use default images for both style and content to run AesUST.
