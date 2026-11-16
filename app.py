"""Giao diện nhận diện hoa dựa trên mô hình E6 của nhóm."""
from html import escape
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from src.flower_catalog import display_name, load_catalog, load_reference_index, reference_images
from src.inference import MAX_BYTES, MAX_PIXELS, default_manifest_path, load_frozen_model, predict_top_k, predict_with_features

st.set_page_config(page_title='Vườn Hoa — Nhận diện hoa', page_icon='🌼', layout='wide', initial_sidebar_state='collapsed')

st.markdown("""
<style>
.stApp { background: #f8f8f2; color: #233b2d; }
.stApp, .stMarkdown, button, p, label { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 5rem; }
.brand { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #dfe6d9; padding-bottom:20px; margin-bottom:38px; }
.brand-name { font-family:Georgia, 'Times New Roman', serif; font-weight:700; font-size:25px; color:#193c2b; letter-spacing:-.03em; }
.brand-name span { display:inline-flex; width:35px; height:35px; border-radius:50%; background:#dcebd9; align-items:center; justify-content:center; margin-right:10px; font-family:sans-serif; font-size:19px; }
.brand-meta { color:#6d806f; font-size:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.hero-eyebrow { color:#6f9071; font-weight:700; font-size:12px; letter-spacing:.14em; text-transform:uppercase; margin-bottom:14px; }
.hero-title { font-family:Georgia, 'Times New Roman', serif; font-size:clamp(37px, 5vw, 62px); font-weight:600; line-height:1.12; letter-spacing:-.04em; color:#173d2b; margin:0 0 18px; }
.hero-title em { color:#6b9870; font-style:italic; }
.hero-copy { max-width:630px; font-size:16px; line-height:1.7; color:#647666; margin-bottom:24px; }
.chips { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:28px; }
.chip { color:#54735b; background:#e7efe3; border:1px solid #dce8d8; border-radius:999px; padding:7px 12px; font-size:12px; font-weight:600; }
.section-kicker { color:#749079; font-size:11px; font-weight:700; letter-spacing:.13em; text-transform:uppercase; margin-bottom:7px; }
.section-title { font-family:Georgia, 'Times New Roman', serif; color:#1f402c; font-size:26px; line-height:1.2; margin:0 0 10px; }
.small-copy { color:#718071; font-size:14px; line-height:1.6; }
.empty-card { min-height:270px; border-radius:20px; background:#eef3e9; border:1px solid #e0e9dc; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; padding:32px; }
.empty-icon { width:60px; height:60px; border-radius:50%; background:#dbead8; display:flex; align-items:center; justify-content:center; font-size:28px; margin-bottom:18px; }
.empty-title { font-family:Georgia, 'Times New Roman', serif; color:#345440; font-size:22px; font-weight:600; margin-bottom:7px; }
.empty-desc { color:#788b78; font-size:14px; line-height:1.55; max-width:320px; }
.result-card { background:#e8f1e4; border:1px solid #d5e5d1; border-radius:20px; padding:25px 26px; margin-bottom:16px; }
.result-label { color:#66856c; font-weight:700; letter-spacing:.12em; font-size:11px; text-transform:uppercase; }
.result-name { color:#173d2b; font-family:Georgia, 'Times New Roman', serif; font-size:clamp(30px,4vw,43px); font-weight:600; line-height:1.15; margin:12px 0 7px; }
.result-latin { color:#708773; font-size:15px; font-style:italic; }
.result-id { display:inline-block; margin-top:17px; border:1px solid #c8dbc5; background:#f5faf2; padding:6px 11px; border-radius:999px; font-size:12px; font-weight:700; color:#54735a; }
.notice { background:#fffaf0; border:1px solid #eee3c9; border-radius:15px; padding:16px 18px; color:#78694b; font-size:13px; line-height:1.55; margin:12px 0 22px; }
.ref-head { border-top:1px solid #dfe6d9; margin-top:32px; padding-top:30px; }
.ref-label { color:#738875; font-size:12px; line-height:1.5; }
.footer-note { border-top:1px solid #dfe6d9; margin-top:45px; padding-top:22px; color:#8a998a; font-size:12px; }
[data-testid="stFileUploader"] { background:#fff; border:1px solid #dfe8dc; border-radius:16px; padding:12px; }
[data-testid="stFileUploaderDropzone"] { background:#f7faf5; border:1px dashed #bfd3bb; border-radius:12px; }
[data-testid="stImage"] img { border-radius:16px; object-fit:contain; }
[data-testid="stAlert"] { border-radius:12px; }
@media(max-width:640px) { .brand-meta {display:none} .block-container {padding-top:1rem} .hero-title {font-size:38px} }
</style>
<div class="brand"><div class="brand-name"><span>✿</span>Vườn Hoa</div><div class="brand-meta">Nhóm 20 · Đề tài 7</div></div>
<div class="hero-eyebrow">NHẬN DIỆN HOA QUA ẢNH</div>
<h1 class="hero-title">Một bức ảnh,<br><em>một gợi ý về hoa.</em></h1>
<div class="hero-copy">Tải lên một bức ảnh để khám phá tên nhóm hoa mà mô hình dự đoán, rồi đối chiếu với những ảnh tham khảo trong bộ dữ liệu.</div>
<div class="chips"><span class="chip">✿ &nbsp;102 nhóm hoa</span><span class="chip">▧ &nbsp;JPEG hoặc PNG</span><span class="chip">⌁ &nbsp;Kết quả tham khảo</span></div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_frozen_model(default_manifest_path())


@st.cache_resource
def get_catalog():
    return load_catalog()


@st.cache_resource
def get_reference_index():
    manifest, payload = get_model()
    return load_reference_index(manifest, payload)


left, right = st.columns([1, 1.08], gap='large')
with left:
    st.markdown('<div class="section-kicker">BƯỚC 01</div><h2 class="section-title">Chọn ảnh của bạn</h2><p class="small-copy">Một bông hoa rõ nét, đủ sáng sẽ dễ đối chiếu hơn. Ảnh được xử lý tại ứng dụng đang chạy trên máy của bạn.</p>', unsafe_allow_html=True)
    upload = st.file_uploader('Ảnh hoa', type=['jpg', 'jpeg', 'png'], help='Chọn ảnh JPEG hoặc PNG, tối đa 10 MiB.', label_visibility='collapsed')
    if upload:
        data = upload.getvalue()
        if len(data) <= MAX_BYTES:
            try:
                with Image.open(BytesIO(data)) as preview:
                    if preview.width * preview.height > MAX_PIXELS:
                        raise ValueError('Ảnh vượt quá giới hạn 20 megapixel.')
                    preview.verify()
                st.image(data, caption='Ảnh bạn đã chọn', width='stretch')
            except (OSError, ValueError, Image.DecompressionBombError):
                st.warning('Không thể xem ảnh. Hãy chọn tệp JPEG hoặc PNG hợp lệ, tối đa 20 megapixel.')
        else:
            st.error('Ảnh vượt quá 10 MiB. Hãy chọn ảnh nhỏ hơn.')

with right:
    st.markdown('<div class="section-kicker">BƯỚC 02</div><h2 class="section-title">Kết quả nhận diện</h2>', unsafe_allow_html=True)
    if not upload:
        st.markdown('<div class="empty-card"><div class="empty-icon">✿</div><div class="empty-title">Chờ một bức ảnh</div><div class="empty-desc">Tải ảnh ở bên trái để xem tên nhóm hoa và ảnh tham khảo tương ứng.</div></div>', unsafe_allow_html=True)
    elif len(data) <= MAX_BYTES:
        try:
            with st.spinner('Đang quan sát bức ảnh...'):
                manifest, payload = get_model()
                top_classes, projected = predict_top_k(data, manifest, payload, k=3)
                class_id = top_classes[0]['class_id']
                item = get_catalog()[class_id]
            name = escape(display_name(item)); original = escape(item['name_en'])
            subtitle = f'<div class="result-latin">Tên gốc trong bộ dữ liệu: {original}</div>' if item.get('name_vi') else '<div class="result-latin">Tên gốc theo bộ dữ liệu (tiếng Anh)</div>'
            st.markdown(f'<div class="result-card"><div class="result-label">NHÓM HOA ĐƯỢC GỢI Ý (HÀNG ĐẦU)</div><div class="result-name">{name}</div>{subtitle}<span class="result-id">Mã lớp Oxford · {class_id:02d}</span></div>', unsafe_allow_html=True)

            top3_html = '<div style="background:#f0f6ee;border:1px solid #d5e5d1;border-radius:14px;padding:12px 16px;margin:12px 0;"><div style="font-size:11px;font-weight:700;color:#54735b;text-transform:uppercase;margin-bottom:8px;">Các lớp nhận phiếu từ K-NN</div>'
            for rank, c_info in enumerate(top_classes, start=1):
                c_item = get_catalog()[c_info['class_id']]
                c_name = escape(display_name(c_item))
                c_orig = escape(c_item['name_en'])
                top3_html += f'<div style="font-size:13px;padding:4px 0;"><strong>#{rank}</strong> {c_name} <span style="color:#788b78">({c_orig}) · Lớp {c_info["class_id"]:02d}</span> — {c_info["vote_share"]*100:.1f}% phiếu có trọng số</div>'
            top3_html += '<div style="font-size:11px;color:#788b78;margin-top:6px;">Tỉ lệ phiếu có trọng số, không phải xác suất đã hiệu chuẩn. Chỉ hiển thị những lớp có phiếu.</div></div>'
            st.markdown(top3_html, unsafe_allow_html=True)
            val_acc = float(manifest['selection']['validation_accuracy']) * 100
            st.markdown(f'<div class="notice"><strong>Hãy xem đây là một gợi ý.</strong> Model đang dùng được fit trên train; Accuracy validation của model này là {val_acc:.1f}%. Lần đánh giá test E6 đã dùng model fit lại trên train+validation và không phải cùng model đang hiển thị.</div>', unsafe_allow_html=True)
            from pathlib import Path
            import json
            test_file = Path(__file__).resolve().parent / 'results/final_test/metrics.json'
            if test_file.is_file():
                with st.expander('Lần đánh giá test đã ghi nhận'):
                    saved = json.loads(test_file.read_text())
                    measured = saved['E6_final_model']['full_test']['accuracy'] * 100
                    st.write(f'E6 16 bin sau khi fit lại trên train+validation đạt Accuracy {measured:.2f}% trên test. Đây là model khác với model train-only đang dùng trong giao diện; biến thể E6_bins32 dẫn đầu validation chưa được đánh giá test.')
            with Image.open(BytesIO(data)) as image:
                image.thumbnail((96, 96)); rgb = np.asarray(image.convert('RGB')).mean(axis=(0, 1)).astype(int)
            st.markdown(f'<div class="small-copy"><strong>Quan sát ảnh:</strong> màu trung bình của toàn ảnh <span style="display:inline-block;width:13px;height:13px;border-radius:50%;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});border:1px solid #becbb9;vertical-align:middle"></span> có tính cả nền và lá, không phải màu riêng của hoa.</div>', unsafe_allow_html=True)
        except (ValueError, OSError, KeyError) as exc:
            st.error(f'Chưa thể nhận diện ảnh: {exc}')
            st.caption('Hãy kiểm tra ảnh hoặc chạy `python scripts/chuan_bi_ui.py` theo README.')
            class_id = None
    else:
        class_id = None

if upload and len(data) <= MAX_BYTES and class_id is not None:
    st.markdown('<div class="ref-head"><div class="section-kicker">ĐỐI CHIẾU BẰNG MẮT</div><h2 class="section-title">Nhìn gần hơn vào nhóm hoa này</h2><p class="small-copy">Đây là tối đa ba ảnh thuộc lớp được dự đoán trong tập huấn luyện, được chọn theo khoảng cách đặc trưng. Hãy so sánh cánh hoa, màu sắc và hình dáng; ảnh tham khảo không xác nhận dự đoán là đúng.</p></div>', unsafe_allow_html=True)
    try:
        paths = reference_images(projected, class_id, payload, get_reference_index())
        if paths:
            cols = st.columns(3, gap='medium')
            for index, path in enumerate(paths):
                with cols[index]:
                    st.image(str(path), width='stretch')
                    st.markdown(f'<div class="ref-label">Ảnh tham khảo {index+1} · Tập huấn luyện</div>', unsafe_allow_html=True)
        else:
            st.info('Chưa tìm thấy ảnh tham khảo cục bộ cho lớp này.')
    except (ValueError, OSError, KeyError) as exc:
        st.info(f'Chưa thể tải ảnh tham khảo: {exc}')
    with st.expander('Mô hình đã dựa vào những gì?'):
        st.write('Mô hình kết hợp thông tin màu sắc và đường nét trong toàn ảnh, sau đó so sánh với ảnh huấn luyện. Lá, nền, ánh sáng và cách chụp cũng có thể ảnh hưởng đến kết quả. Thông tin này mô tả cách mô hình hoạt động, không phải đặc điểm riêng đã được xác minh của loài hoa.')

st.markdown('<div class="footer-note">VƯỜN HOA · Sản phẩm môn Học máy của Nhóm 20 &nbsp; / &nbsp; Tên lớp đối chiếu theo <a href="https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/oxford_flowers102/oxford_flowers102_dataset_builder.py" target="_blank">TensorFlow Datasets</a></div>', unsafe_allow_html=True)
