"""Vẽ sơ đồ pipeline học máy không rò rỉ dữ liệu (No Data Leakage Pipeline)."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.utils import ROOT

FIG_DIR = ROOT / 'reports/latex/figures'


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Styles
    box_blue = dict(boxstyle="round,pad=0.5", fc="#e3f2fd", ec="#1e88e5", lw=1.5)
    box_green = dict(boxstyle="round,pad=0.5", fc="#e8f5e9", ec="#43a047", lw=1.5)
    box_orange = dict(boxstyle="round,pad=0.5", fc="#fff3e0", ec="#fb8c00", lw=1.5)
    box_purple = dict(boxstyle="round,pad=0.5", fc="#f3e5f5", ec="#8e24aa", lw=1.5)
    box_red = dict(boxstyle="round,pad=0.5", fc="#ffebee", ec="#e53935", lw=1.5)

    # 1. Dữ liệu gốc
    ax.text(12, 85, "1. DỮ LIỆU GỐC\nOxford 102 Flowers\n(8.189 ảnh, 102 lớp)", ha="center", va="center", bbox=box_blue, fontsize=10, fontweight="bold")
    
    # Mũi tên chia split
    ax.annotate("", xy=(30, 85), xytext=(22, 85), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 2. Phân chia Split chính thức
    ax.text(48, 85, "2. PHÂN CHIA SPLIT CHÍNH THỨC\nTrain: 1.020 ảnh (10 ảnh/lớp)\nValidation: 1.020 ảnh (10 ảnh/lớp)\nTest: 6.149 ảnh (Holdout độc lập)", ha="center", va="center", bbox=box_green, fontsize=9.5)

    # Khung ranh giới chống rò rỉ
    leak_rect = patches.FancyBboxPatch((2, 18), 70, 52, boxstyle="round,pad=0.8", fc="#fafafa", ec="#d32f2f", ls="--", lw=1.8)
    ax.add_patch(leak_rect)
    ax.text(37, 68, "RANH GIỚI CHỐNG RÒ RỈ (LEAK-FREE BOUNDARY)\nTuyệt đối không dùng Validation/Test để fit tham số", ha="center", va="center", color="#c62828", fontweight="bold", fontsize=10)

    # 3. Tiền xử lý & Trích xuất đặc trưng
    ax.text(18, 52, "Tiền xử lý độc lập\nRGB, Resize 64x64\nChuẩn hóa [0, 1]", ha="center", va="center", bbox=box_blue, fontsize=9)
    ax.annotate("", xy=(28, 52), xytext=(24, 52), arrowprops=dict(arrowstyle="->", lw=1.2))

    ax.text(42, 52, "Trích xuất đặc trưng\nHOG (1.764d) + HSV (48d)\nNối đặc trưng = 1.812d", ha="center", va="center", bbox=box_blue, fontsize=9)
    ax.annotate("", xy=(42, 42), xytext=(42, 47), arrowprops=dict(arrowstyle="->", lw=1.2))

    # 4. Học biến đổi tham số (PCA)
    ax.text(42, 34, "HỌC THAM SỐ: PCA (128 chiều)\nCHỈ FIT TRÊN TRAIN (hoặc train fold)\nValidation & Test chỉ dùng transform()", ha="center", va="center", bbox=box_orange, fontsize=9.5, fontweight="bold")

    # 5. Huấn luyện mô hình
    ax.text(42, 22, "Huấn luyện K-NN / Cây quyết định\nChỉ fit trên vector Train đã chiếu PCA", ha="center", va="center", bbox=box_purple, fontsize=9)

    # 6. Đánh giá chọn mô hình
    ax.annotate("", xy=(85, 52), xytext=(56, 34), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1565c0"))
    ax.text(86, 55, "3. ĐÁNH GIÁ & CHỌN MÔ HÌNH\nChỉ dùng Validation & 5-fold CV\nTiêu chí: Macro F1, hòa xét Accuracy\n-> Chọn Mô hình cuối E6", ha="center", va="center", bbox=box_green, fontsize=9.5, fontweight="bold")

    # 7. Đánh giá Test cuối duy nhất
    ax.annotate("", xy=(86, 25), xytext=(86, 45), arrowprops=dict(arrowstyle="->", lw=2, color="#d32f2f"))
    ax.text(86, 22, "4. ĐÁNH GIÁ TEST CUỐI (1 LẦN)\nĐóng băng E6 & Baseline E1\nFit lại trên Train+Val (2.040 ảnh)\nĐo trên toàn bộ Test (6.149 ảnh)\nKiểm toán độ nhạy ảnh trùng byte", ha="center", va="center", bbox=box_red, fontsize=9.5, fontweight="bold")

    # 8. Phân tích lỗi
    ax.text(50, 6, "5. PHÂN TÍCH LỖI & ĐỐI CHỨNG GIẢ THUYẾT\nTop cặp nhầm lẫn, Top lớp F1 cao/thấp, Heatmap rút gọn\nThí nghiệm đối chứng: image_size 128 và histogram bins 32", ha="center", va="center", bbox=box_blue, fontsize=9.5)
    ax.annotate("", xy=(50, 11), xytext=(72, 18), arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474f"))

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'pipeline_sodo.png', dpi=180)
    plt.close(fig)
    print("Đã tạo reports/latex/figures/pipeline_sodo.png")


if __name__ == '__main__':
    draw_pipeline()
