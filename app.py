"""Streamlit demonstration of the frozen V2 E6 flower classifier."""
import streamlit as st
from src.inference import MAX_BYTES, default_manifest_path, load_frozen_model, predict_image

st.set_page_config(page_title='Phân loại hoa Oxford — E6', page_icon='🌼')
st.title('Phân loại hoa Oxford 102: mô hình E6')
st.write('Tải ảnh JPEG hoặc PNG để dự đoán mã lớp Oxford (1–102).')
st.caption('Đây là mô hình cơ sở phục vụ nghiên cứu; E6 chưa được đánh giá trên tập kiểm thử chính thức và không cung cấp độ tin cậy đã hiệu chuẩn.')

@st.cache_resource
def get_model():
    return load_frozen_model(default_manifest_path())

upload = st.file_uploader('Ảnh hoa (tối đa 10 MiB)', type=['jpg', 'jpeg', 'png'])
if upload is not None:
    data = upload.getvalue()
    if len(data) > MAX_BYTES:
        st.error('Ảnh vượt quá giới hạn 10 MiB.')
    else:
        try:
            manifest, model = get_model()
            class_id = predict_image(data, manifest, model)
            st.image(data, caption='Ảnh đã tải lên')
            st.success(f'Mã lớp Oxford dự đoán: {class_id}')
        except (ValueError, OSError, KeyError) as exc:
            st.error('Không thể dự đoán. Hãy kiểm tra định dạng, kích thước ảnh, model tin cậy và cấu hình môi trường theo README.')
