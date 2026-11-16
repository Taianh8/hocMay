"""Tạo các hình ảnh và bảng phân tích lỗi cho báo cáo LaTeX."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data_loader import load_dataset
from src.flower_catalog import load_catalog
from src.utils import ROOT

FIG_DIR = ROOT / 'reports/latex/figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def create_eda_sample_grid():
    """Tạo lưới ảnh mẫu một số lớp tiêu biểu cho EDA (Chương 1)."""
    df = load_dataset()
    catalog = load_catalog()
    # Chọn 8 lớp tiêu biểu đại diện các hình thái và màu sắc khác nhau
    selected_classes = [1, 8, 15, 24, 58, 63, 77, 94]
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, cid in zip(axes.flat, selected_classes):
        # Lấy ảnh đầu tiên của lớp trong tập train
        row = df[(df.class_id == cid) & (df.split == 'train')].iloc[0]
        im_path = Path(row.path)
        with Image.open(im_path) as im:
            ax.imshow(im.convert('RGB'))
        name = catalog[cid].get('name_vi') or catalog[cid]['name_en'].capitalize()
        ax.set_title(f"Lớp {cid}: {name}\n({catalog[cid]['name_en']})", fontsize=10)
        ax.axis('off')
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'anh_mau_eda.png', dpi=180)
    plt.close(fig)
    print("Đã tạo reports/latex/figures/anh_mau_eda.png")


def create_top_confusion_heatmap():
    """Tạo heatmap top 15 lớp bị nhầm nhiều nhất trên validation (thay cho ma trận 102x102 khó đọc)."""
    cm = np.load(ROOT / 'results/v2/metrics/E6_validation_confusion.npy')
    catalog = load_catalog()
    
    # Tính tổng số lần nhầm lẫn ngoài đường chéo của từng lớp
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    confusions_per_class = off_diag.sum(axis=1) + off_diag.sum(axis=0)
    
    # Chọn top 15 lớp có nhiều nhầm lẫn nhất
    top_indices = np.argsort(confusions_per_class)[::-1][:15]
    sub_cm = cm[np.ix_(top_indices, top_indices)]
    
    labels = [f"L.{i+1} ({catalog[i+1]['name_en'][:10]})" for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(sub_cm, cmap='YlOrRd', interpolation='nearest')
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    
    # Ghi số đếm lên từng ô
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = sub_cm[i, j]
            color = 'white' if val > sub_cm.max() / 2 else 'black'
            ax.text(j, i, str(int(val)), ha='center', va='center', color=color, fontsize=8,
                    fontweight='bold' if i == j else 'normal')
            
    ax.set_title('Ma trận nhầm lẫn rút gọn: Top 15 lớp bị nhầm nhiều nhất (Validation E6)', fontsize=12, pad=15)
    ax.set_xlabel('Lớp dự đoán', fontsize=11, labelpad=10)
    ax.set_ylabel('Lớp thực tế', fontsize=11, labelpad=10)
    
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'ma_tran_e6_top15_v5.png', dpi=180)
    plt.close(fig)
    print("Đã tạo reports/latex/figures/ma_tran_e6_top15_v5.png (heatmap rút gọn)")


def create_confusion_side_by_side_grid():
    """Tạo lưới ảnh đối chiếu 3 cột: Ảnh dự đoán sai | Mẫu lớp đúng | Mẫu lớp nhầm."""
    df = load_dataset()
    catalog = load_catalog()
    
    # 4 cặp nhầm tiêu biểu từ validation
    cases = [
        (6814, 24, 11),  # red ginger nhầm thành snapdragon
        (7336, 94, 32),  # foxglove nhầm thành garden phlox
        (6413, 9, 29),   # monkshood nhầm thành artichoke
        (4260, 18, 30)   # peruvian lily nhầm thành sweet william
    ]
    
    fig, axes = plt.subplots(4, 3, figsize=(11, 13))
    
    for row_idx, (wrong_id, true_cid, pred_cid) in enumerate(cases):
        wrong_path = ROOT / 'data/raw/jpg' / f'image_{wrong_id:05d}.jpg'
        
        # Mẫu train của true_class
        true_train_row = df[(df.class_id == true_cid) & (df.split == 'train')].iloc[0]
        true_train_path = Path(true_train_row.path)
        
        # Mẫu train của pred_class
        pred_train_row = df[(df.class_id == pred_cid) & (df.split == 'train')].iloc[0]
        pred_train_path = Path(pred_train_row.path)
        
        # Cột 1: Ảnh sai
        with Image.open(wrong_path) as im:
            axes[row_idx, 0].imshow(im.convert('RGB'))
        axes[row_idx, 0].set_title(f"Ảnh sai #{wrong_id}\nThực tế: Lớp {true_cid} ({catalog[true_cid]['name_en']})\nDự đoán: Lớp {pred_cid} ({catalog[pred_cid]['name_en']})", fontsize=9, color='darkred')
        axes[row_idx, 0].axis('off')
        
        # Cột 2: Mẫu train lớp đúng
        with Image.open(true_train_path) as im:
            axes[row_idx, 1].imshow(im.convert('RGB'))
        axes[row_idx, 1].set_title(f"Mẫu huấn luyện lớp đúng {true_cid}\n({catalog[true_cid]['name_en']})", fontsize=9, color='darkgreen')
        axes[row_idx, 1].axis('off')
        
        # Cột 3: Mẫu train lớp bị nhầm
        with Image.open(pred_train_path) as im:
            axes[row_idx, 2].imshow(im.convert('RGB'))
        axes[row_idx, 2].set_title(f"Mẫu huấn luyện lớp bị nhầm {pred_cid}\n({catalog[pred_cid]['name_en']})", fontsize=9, color='darkblue')
        axes[row_idx, 2].axis('off')
        
    fig.suptitle("Phân tích lỗi E6: Ảnh sai trên Validation so với Mẫu huấn luyện tương ứng", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'vi_du_nham.png', dpi=180)
    plt.close(fig)
    print("Đã tạo reports/latex/figures/vi_du_nham.png")


if __name__ == '__main__':
    create_eda_sample_grid()
    create_top_confusion_heatmap()
    create_confusion_side_by_side_grid()
