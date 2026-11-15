"""Tạo hình tiếng Việt từ kết quả đã lưu; không chạy lại model."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from src.utils import ROOT
from src.preprocessing import preprocess

OUT = ROOT/'reports/latex/figures'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(ROOT/'results/metrics/dataset_metadata.csv')
    fig, ax = plt.subplots(figsize=(12, 3)); metadata.groupby('class_id').size().plot.bar(ax=ax, width=1)
    ax.set(title='Phân bố số ảnh theo lớp', xlabel='Mã lớp chính thức', ylabel='Số ảnh'); ax.tick_params(axis='x', labelsize=5); fig.tight_layout(); fig.savefig(OUT/'phan_bo.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].scatter(metadata.width,metadata.height,s=2,alpha=.2); axes[0].set(xlabel='Chiều rộng (pixel)',ylabel='Chiều cao (pixel)',title='Kích thước ảnh'); (metadata.width/metadata.height).hist(ax=axes[1],bins=40);axes[1].set(xlabel='Tỉ lệ rộng/cao',ylabel='Số ảnh',title='Tỉ lệ ảnh');fig.tight_layout();fig.savefig(OUT/'kich_thuoc.png',dpi=150);plt.close(fig)
    sample=metadata.groupby('class_id').head(1).head(12);fig,axes=plt.subplots(3,4,figsize=(10,8))
    for ax,row in zip(axes.flat,sample.itertuples()):
        with Image.open(ROOT/row.path) as im: ax.imshow(im.convert('RGB'))
        ax.set_title(f'Lớp {row.class_id}');ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'anh_mau.png',dpi=150);plt.close(fig)
    comparison = json.loads((ROOT/'results/v2/metrics/comparison.json').read_text())['experiments']
    fig, ax = plt.subplots(figsize=(7, 3)); ax.bar([r['experiment'] for r in comparison], [r['val_macro_f1'] for r in comparison]); ax.set(title='So sánh trên tập validation', ylabel='F1 trung bình vĩ mô', xlabel='Thí nghiệm'); fig.tight_layout(); fig.savefig(OUT/'validation.png',dpi=150); plt.close(fig)
    cv = json.loads((ROOT/'results/v3/metrics/cv_summary.json').read_text())['summaries']; rows=[r for r in cv if r['pca_seed'] in (None,42)]
    fig, ax = plt.subplots(figsize=(7, 3)); ax.bar([r['experiment'] for r in rows],[r['mean_macro_f1'] for r in rows],yerr=[r['std_macro_f1_population'] for r in rows],capsize=4); ax.set(title='Kiểm định chéo năm fold trên train', xlabel='Thí nghiệm', ylabel='F1 trung bình vĩ mô'); fig.tight_layout(); fig.savefig(OUT/'kiem_dinh_cheo.png',dpi=150); plt.close(fig)
    cm=np.load(ROOT/'results/v2/metrics/E6_validation_confusion.npy'); fig, ax=plt.subplots(figsize=(8,7)); im=ax.imshow(cm,cmap='Blues',interpolation='nearest'); ticks=list(range(0,102,10)); ax.set_xticks(ticks,[str(t+1) for t in ticks]);ax.set_yticks(ticks,[str(t+1) for t in ticks]); ax.set(title='Ma trận nhầm lẫn E6 trên validation',xlabel='Mã lớp dự đoán',ylabel='Mã lớp thực tế');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/'ma_tran_e6.png',dpi=150);plt.close(fig)
    per=pd.read_csv(ROOT/'results/v3/metrics/E6_per_class.csv');fig,ax=plt.subplots(figsize=(12,3));ax.bar(per.class_id,per.f1);ax.set(title='F1 từng lớp của E6 trên validation',xlabel='Mã lớp chính thức',ylabel='F1');fig.tight_layout();fig.savefig(OUT/'f1_tung_lop.png',dpi=150);plt.close(fig)
    for exp in ('E4','E5','E6'):
        cm=np.load(ROOT/f'results/v2/metrics/{exp}_validation_confusion.npy')
        fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(cm,cmap='Blues',interpolation='nearest'); ticks=list(range(0,102,10)); ax.set_xticks(ticks,[str(t+1) for t in ticks]); ax.set_yticks(ticks,[str(t+1) for t in ticks]); ax.set(title=f'Ma trận nhầm lẫn {exp} trên validation',xlabel='Mã lớp dự đoán',ylabel='Mã lớp thực tế');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/f'{exp}_ma_tran.png',dpi=150);plt.close(fig)
        pred=pd.read_csv(ROOT/f'results/v2/metrics/{exp}_validation_predictions.csv'); wrong=pred[pred.class_id!=pred.predicted_class_id].head(12)
        fig,axes=plt.subplots(3,4,figsize=(12,9))
        for ax in axes.flat: ax.axis('off')
        for ax,row in zip(axes.flat,wrong.itertuples()):
            with Image.open(ROOT/'data/raw/jpg'/f'image_{row.image_id:05d}.jpg') as im: ax.imshow(im.convert('RGB'))
            ax.set_title(f'Thực tế {row.class_id}, dự đoán {row.predicted_class_id}')
        fig.suptitle(f'{exp}: ví dụ lỗi trên validation');fig.tight_layout();fig.savefig(OUT/f'{exp}_loi.png',dpi=120);plt.close(fig)
    examples=pd.read_csv(ROOT/'results/v3/metrics/E6_top_confusion_examples.csv')
    fig,axes=plt.subplots(4,3,figsize=(11,12))
    for ax in axes.flat: ax.axis('off')
    for ax,row in zip(axes.flat,examples.itertuples()):
        with Image.open(ROOT/'data/raw/jpg'/f'image_{row.image_id:05d}.jpg') as im: ax.imshow(im.convert('RGB'))
        ax.set_title(f'Ảnh {row.image_id}: {row.class_id} → {row.predicted_class_id}')
    fig.suptitle('E6: ví dụ các cặp nhầm thường gặp trên validation');fig.tight_layout();fig.savefig(OUT/'vi_du_nham.png',dpi=120);plt.close(fig)
    first=metadata.iloc[0]; path=ROOT/first['path']; fig,axes=plt.subplots(1,2,figsize=(8,4))
    with Image.open(path) as image: axes[0].imshow(image.convert('RGB'))
    axes[1].imshow(preprocess(path,64)); axes[0].set_title('Ảnh gốc'); axes[1].set_title('RGB sau resize và chuẩn hóa')
    for ax in axes: ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'tien_xu_ly.png',dpi=150);plt.close(fig)
    print('Đã tạo hình từ artifact đã lưu; không đánh giá model.')
if __name__=='__main__': main()
