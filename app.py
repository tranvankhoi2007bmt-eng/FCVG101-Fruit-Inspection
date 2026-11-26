"""Streamlit application: an operator selects an image-analysis task."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
from src.vision import filters, edges, frequency, features, match_views, calibrate, undistort
from src.model import FruitCNN, predict
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Smart Organic Farm Assistant',page_icon='🍎',layout='wide')
st.title('Smart Organic Farm Assistant')
st.caption('FCVG101 | Visual inspection prototype — not a food safety certification.')
mode=st.sidebar.radio('Select analysis', ['Classify fruit','Filter & edges','Frequency analysis','Detect features','Match two views','Calibrate camera','Undistort image'])
def decode(file):
    return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(Image.open(file)).convert('RGB')),cv2.COLOR_RGB2BGR)
def show(img,caption):
    st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB) if img.ndim==3 else img,caption=caption,use_container_width=True,clamp=True)
@st.cache_resource
def load_model(path, modified):
    m=FruitCNN(); checkpoint=torch.load(path,map_location='cpu',weights_only=True);m.load_state_dict(checkpoint['state_dict']);m.eval();return m
try:
    if mode=='Calibrate camera':
        files=st.file_uploader('Upload five or more checkerboard photos from the same camera',type=['jpg','jpeg','png'],accept_multiple_files=True)
        c=st.number_input('Inner corners across',min_value=2,value=9); r=st.number_input('Inner corners down',min_value=2,value=6); square=st.number_input('Square size (mm)',min_value=.1,value=25.)
        if st.button('Calibrate',disabled=len(files)<5):
            images=[decode(f) for f in files];result,preview=calibrate(images,(int(c),int(r)),square)
            st.session_state['calibration']=result;st.metric('RMS reprojection error (px)',f"{result['rms_px']:.3f}");st.write('Intrinsic matrix K',result['K']);st.write('Distortion coefficients',result['distortion']);show(preview,'Detected checkerboard');show(undistort(images[result['accepted'][0]],result['K'],result['distortion'],result['size']),'Undistorted')
    else:
        file=st.file_uploader('Upload an image',type=['jpg','jpeg','png'])
        if file:
            image=decode(file);show(image,'Input')
            if mode=='Classify fruit':
                path=ROOT/'models'/'fruit_cnn.pt'
                if not path.exists(): st.warning('Model not trained. Run python -m src.download_data and python -m src.train first.')
                else:
                    probs=predict(load_model(path,path.stat().st_mtime),Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
                    label=max(probs,key=probs.get);st.metric('Predicted class',label.title(),f'{probs[label]:.1%} confidence');st.bar_chart(probs);st.caption('Model confidence is not a guarantee. Inspect the fruit manually.')
            elif mode=='Filter & edges':
                for name,img in {**filters(image),**edges(image)}.items():show(img,name)
            elif mode=='Frequency analysis':
                for name,img in frequency(image).items():show(cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8) if img.dtype!=np.uint8 else img,name)
            elif mode=='Detect features':
                imgs,counts=features(image);st.json(counts)
                for name,img in imgs.items():show(img,name)
            elif mode=='Match two views':
                second=st.file_uploader('Upload another view of the same object',type=['jpg','jpeg','png']);method=st.selectbox('Detector',['ORB','SIFT'])
                if second:
                    result,stats=match_views(image,decode(second),method);st.json(stats);show(result,'Ratio-test matches (up to 50 drawn)')
            elif mode=='Undistort image':
                cal=st.session_state.get('calibration')
                if cal is None:st.info('First calibrate your camera in the Calibrate camera mode.')
                else:show(undistort(image,cal['K'],cal['distortion'],cal['size']),'Undistorted image')
except (ValueError,OSError,cv2.error,RuntimeError) as e:
    st.error(str(e))
