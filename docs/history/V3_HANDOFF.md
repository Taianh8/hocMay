# Bàn giao V3 — kiểm định chéo trên train và phân tích validation

**Cập nhật:** Các output thực nghiệm và cache đã được xóa theo yêu cầu. Tài liệu này ghi lại kết quả lịch sử; chạy lại theo README để tạo output mới.
V3 hoàn tất phân tích độ ổn định; không đánh giá test V2/V3. Cấu hình, model, dự đoán và điểm lịch sử V1/V2 được giữ. Cấu hình ở `configs/experiments_v3.yaml`; output, nhật ký notebook và đặc trưng train trung gian ở `results/v3/`. Một số CSV/JSON nhỏ có thể được theo dõi sau cleanup. `python -m scripts.check_v3_artifacts` đối chiếu split, F1 từ dự đoán, thống kê, chỉ số validation và nguồn gốc; chỉ dùng artifact tin cậy. Lệnh đầy đủ ở README.

E1 là pixel k=1; E4 HOG 8 + HSV, k=5, Euclidean có trọng số khoảng cách; E6 cùng đặc trưng với PCA randomized 128, k=3, cosine có trọng số khoảng cách. Đây là các cấu hình tốt nhất đã chọn bằng validation. Cùng năm fold train chính thức, xáo trộn seed 42: 816 fit, 204 holdout; mỗi fold holdout có hai ảnh/lớp. Validation và test không nằm trong fold. PCA fit riêng trên phần train từng fold. E6 thử seed PCA 42/7/123 trên cùng fold; E1/E4 không có bước fit ngẫu nhiên. Không tìm hyperparameter hoặc đánh giá test ở V3.

| Model/seed | F1 vĩ mô năm fold | Trung bình | Độ lệch chuẩn quần thể |
|---|---|---:|---:|
| E1 | 0,0814; 0,1042; 0,0842; 0,0712; 0,0975 | 0,0877 | 0,0118 |
| E4 | 0,1660; 0,1960; 0,2074; 0,1750; 0,1817 | 0,1852 | 0,0148 |
| E6 / 42 | 0,1584; 0,1906; 0,1998; 0,2178; 0,1992 | 0,1932 | 0,0195 |
| E6 / 7 | 0,1666; 0,1993; 0,2120; 0,2102; 0,1910 | 0,1958 | 0,0165 |
| E6 / 123 | 0,1593; 0,1995; 0,2081; 0,2194; 0,1987 | 0,1970 | 0,0203 |

Chênh lệch trung bình E6/E4 khoảng 0,0080 với seed 42 nhỏ hơn biến thiên fold, không phải khoảng tin cậy. Trung bình các seed PCA 0,1932–0,1970 trên cùng fold không chứng minh ổn định trên dữ liệu khác. CV không đánh giá toàn bộ quy trình chọn model vì tham số đã chọn trên validation; mỗi fold chỉ fit tám ảnh/lớp. Tìm 56 cấu hình và phân tích validation nhiều lần làm tăng rủi ro lạc quan.

Phân tích lỗi dùng dự đoán E4–E6 đã lưu và nhãn chính thức, đối chiếu với ma trận V2, xuất Precision, Recall, F1, support và số dự đoán theo lớp. E6 có 22 lớp F1=0, mỗi lớp có 10 ảnh validation. Cặp nhầm có hướng nhiều nhất 24→11 và 94→32, bốn ảnh mỗi cặp. Hình `results/v3/figures/E6_confusion_examples.png` cho thấy ảnh 6814, 6819, 6834 của lớp 24 có hoa đỏ/hồng giữa lá xanh; ảnh lớp 94 có tỉ lệ và nền khác nhau, 7336 chụp gần và 7386 cụm nhỏ trên nền tối. Ảnh 4260 và 4296 lớp 18 khác về ánh sáng và cách cắt. Màu, nền, kích thước, ánh sáng, hình dạng có thể ảnh hưởng HOG/histogram, nhưng chưa có đối chứng để khẳng định nguyên nhân. Chưa ánh xạ tên lớp.

Trung bình tài nguyên theo fold, seed 42: E1 12.288 chiều, fit 0,0082 s, dự đoán 0,0580 s, RSS đỉnh worker 343,8 MiB; E4 1.812 chiều, 0,0040 s, 0,0234 s, 214,1 MiB; E6 1.812→128 chiều, PCA fit/transform train và K-NN fit 0,0278 s, transform/dự đoán holdout 0,0026 s, 217,0 MiB. RSS bao gồm interpreter, thư viện, ma trận train và mảng tạm; không phải kích thước model. Thời gian không gồm tải dữ liệu và trích xuất. Giải mã/resize/chuẩn hóa 1.020 ảnh train mất 2,271 s; vector pixel 0,002 s; HOG+HSV 1,161 s. Các số phụ thuộc máy và cache.

Đề xuất đánh giá cuối: khóa E6 với seed 42, PCA/K-NN fit trên train chính thức; trước dự đoán test phải khóa mã, hash, chỉ số và được cho phép riêng. Báo cáo Accuracy, macro/weighted Precision, Recall, F1, ma trận và lỗi bất kể kết quả; không chọn E4 hay E6 bằng test. Train thêm validation là quy trình khác phải định trước. V1 đã xem test nên cần công bố và ưu tiên holdout độc lập mới. Test mất cân bằng và ảnh gần trùng chưa được định lượng lúc bàn giao V3; kết quả kiểm toán sau đó ở V4.

V4 được đề xuất: lấy holdout độc lập hoặc thiết kế nested selection, kiểm tra ảnh trùng/nguồn gốc, kiểm định giả thuyết lỗi có định trước; không phát triển theo phản hồi test chính thức.
