# DIP Final
DIP Final Report
We would like to combine different methods to form a better result for style transferring tdtm, aka 武士雞, and give stable output of the videos. 
## File Structure

## Methods Used
1. AesUST from [Official Pytorch code for AesUST](https://github.com/EndyWon/AesUST)
2. WISE from [White-box Style Transfer Editing (WISE)](https://github.com/winfried-ripken/wise)
3. STROTSS from [Memory Efficient Version Of Strotts](https://github.com/futscdav/strotss)
4. Fast Artistic Videos from [Fast Artistic Videos in pyTorch](https://github.com/pgalatic/fast-artistic-videos-pytorch)
5. ReReVST from [ReReVST-Code](https://github.com/daooshee/ReReVST-Code?fbclid=IwAR0cMbVQ100brf97DcybltNrZ6bEGjxAg769LZP0rWLnGM6VYfHgRvGWwFM)
## How to use Streamlit
### Prerequisites
We use Python 3.10. Please run the below code to install the required packages.
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
To launch the Streamlit Interface, enter code/ and run:
```
streamlit run Whitebox_style_transfer.py
```

## Acknowledgement
@inproceedings{wang2022aesust,
  title={AesUST: towards aesthetic-enhanced universal style transfer},
  author={Wang, Zhizhong and Zhang, Zhanjie and Zhao, Lei and Zuo, Zhiwen and Li, Ailin and Xing, Wei and Lu, Dongming},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1095--1106},
  year={2022}
}
