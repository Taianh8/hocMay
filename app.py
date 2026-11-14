"""Streamlit demonstration of the frozen V2 E6 flower classifier."""
import os
from pathlib import Path
import streamlit as st
from src.inference import MAX_BYTES, load_frozen_model, predict_image

st.set_page_config(page_title='Oxford Flowers — E6', page_icon='🌼')
st.title('Oxford 102 Flowers: E6 demo')
st.write('Upload a JPEG or PNG to predict its Oxford class ID (1–102).')
st.caption('Research baseline; the model has not been evaluated as V2 on the official test set. It does not provide calibrated confidence.')

@st.cache_resource
def get_model():
    selected = os.environ.get('FLOWERS_MODEL_MANIFEST')
    return load_frozen_model(Path(selected)) if selected else load_frozen_model()

upload = st.file_uploader('Flower image (up to 10 MiB)', type=['jpg', 'jpeg', 'png'])
if upload is not None:
    data = upload.getvalue()
    if len(data) > MAX_BYTES:
        st.error('The image exceeds 10 MiB.')
    else:
        try:
            manifest, model = get_model()
            class_id = predict_image(data, manifest, model)
            st.image(data, caption='Uploaded image')
            st.success(f'Predicted Oxford class ID: {class_id}')
        except (ValueError, OSError, KeyError) as exc:
            st.error(f'Unable to predict: {exc}')
