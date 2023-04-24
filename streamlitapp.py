import streamlit as st
import numpy as np
import nibabel as nib
import os
import cv2
import tensorflow as tf

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

from model_s.unet import multi_unet_model as unet
from model_s.mnet import multi_unet_model as mnet
from model_s.amnet import multi_unet_model as attention_mnet
from model_s.aunet import multi_unet_model as attention_unet


model_name = {
        'U-Net': [unet, '/app/cardiac-mri-image-segmentor/weights/unet.h5'], 
        'M-Net': [mnet, '/app/cardiac-mri-image-segmentor/weights/mnet.h5'], 
        'Attention U-Net': [attention_unet, '/app/cardiac-mri-image-segmentor/weights/aunet.h5'], 
        'Attention M-Net': [attention_mnet, '/app/cardiac-mri-image-segmentor/weights/amnet.h5'],
    }

st.set_page_config(layout="wide", page_icon=':heart:', page_title='MIS')

st.title('Cardiac MRI Segmentor')

cot1 = st.container()
cot3 = st.container()
cot2 = st.container()
cot4 = st.container()


cot1_col1, cot1_col2 = cot1.columns(2, gap='large')
cot2_col1, cot2_col2 = cot2.columns(2, gap='large')

#if st.button('Upload your .nii.gz file'):

def file_change():
    st.session_state['run'] = False

uploaded_file = cot1_col1.file_uploader("Upload your .nii.gz file", on_change=file_change)

if uploaded_file is None:
    cot1_col1.write("Please upload a file")
else:
    with open(os.path.join("tmp.nii.gz"),"wb") as f:
        f.write(uploaded_file.getbuffer())

    img = nib.load("tmp.nii.gz")
    img = np.array(img.dataobj)/255.0

    cot2_col1.subheader('Input Image')
    st.session_state['slice'] = cot3.slider('Slice Number?', 0, img.shape[2]-1, 0)

    img_res = []
    for i in range(img.shape[2]):
        img_res.append(cv2.resize(img[:, :, i], dsize=(256, 256), interpolation=cv2.INTER_CUBIC))

    cot2_col1.image(img_res[st.session_state['slice']], clamp=True, caption='MRI', use_column_width='always')

    img_res_np = np.asarray(img_res)
    img_res = np.expand_dims(img_res_np, -1)  

    with cot1_col2:
        if 'run' not in st.session_state or st.session_state['run']==False:
            form = st.empty()
            with form.form("form"):
                model_choice = st.radio(
                    "Choose the Segmentation Model:",
                    ('U-Net', 'M-Net', 'Attention U-Net', 'Attention M-Net'))
                
                submitted = st.form_submit_button("Run")
                if submitted:
                    st.session_state['run'] = True
                    st.session_state['model_choice'] = model_choice
                    model = model_name[model_choice][0](4, 256, 256, 1)
                    model.load_weights(model_name[model_choice][1])
                    model.compile(optimizer='adam', loss=sm.losses.CategoricalFocalLoss(), metrics=['accuracy'])
                    
                    pred = model.predict(img_res)
                    pred_images = np.argmax(pred, axis=3)

                    # layers = [layer.name for layer in model.layers[1:] if 'conv' in layer.name and 'conv2d_transpose' not in layer.name]
                    # layers = [layer for i, layer in enumerate(layers) if i%2==1]
                    
                    successive_outputs = [layer.output for layer in model.layers[1:] if 'conv' in layer.name and 'conv2d_transpose' not in layer.name]
                    successive_outputs = [layer for i, layer in enumerate(successive_outputs) if i%2==1]
                    
                    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

                    st.session_state['fmaps'] = visualization_model.predict(img_res)
                    # st.session_state['layers'] = layers

                    rgb_images = []

                    for i in range(img_res_np.shape[0]):
                        rgb = cv2.cvtColor(img_res_np[i],cv2.COLOR_GRAY2RGB)
                        for j in range(rgb.shape[0]):
                            for k in range(rgb.shape[1]):
                                # ( (1-p)R1 + p*R2, (1-p)*G1 + p*G2, (1-p)*B1 + p*B2 )
                                pixel = pred_images[i][j][k]
                                opacity = 0.01
                                if pixel!=0:
                                    rgb[j][k][pixel-1] = opacity*255.0 + (1-opacity)*rgb[j][k][pixel-1]
                        rgb_images.append(rgb)

                    rgb_images = np.asarray(rgb_images)
                    
                    st.session_state['pred'] = rgb_images
                    form.empty()

        if 'run' in st.session_state and st.session_state['run']:
            st.subheader('Current Model: ' + st.session_state['model_choice'])
            button = st.empty()
            if button.button('Choose Another Model'):
                st.session_state['run'] = False
                button.empty()
                st.experimental_rerun()

    with cot2_col2:
        if 'run' in st.session_state and st.session_state['run']:
            # st.snow()
            st.subheader('Segmented Image')
            st.image(st.session_state['pred'][st.session_state['slice']], caption='Segmented Image', use_column_width='always', channels='RGB', clamp=True)

    if 'run' in st.session_state and st.session_state['run']:
        # st.write(st.session_state['layers'])
        if cot4.button('Show Feature Maps'):
            tabs = []

            for i in range(len(st.session_state['fmaps'])):
                tabs.append(f'Layer{i}')
            
            tabs = st.tabs(tabs)

            for i in range(len(st.session_state['fmaps'])):
                fmap = np.asarray(st.session_state['fmaps'][i])
                images = []
                for j in range(fmap.shape[3]):
                    images.append(fmap[st.session_state['slice'],:,:,j])
                
                tabs[i].write(len(images))
                tabs[i].image(images, clamp=True, width=128)






