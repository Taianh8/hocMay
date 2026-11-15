# Báo cáo học thuật tiếng Việt

`main.tex` dùng Unicode, `fontspec` và phông Times New Roman có dấu tiếng Việt. Thông tin Trường Đại học Đại Nam, Nhóm 20, Đề tài 7 và hai thành viên đã được điền. Đã điền K18, chuyên ngành Phát triển phần mềm, giảng viên ThS. Lê Thị Thuỳ Trang, Hà Nội và năm 2026; nhóm cần rà soát lại thông tin trước khi nộp. Output thực nghiệm đã được xóa theo yêu cầu; các số trong PDF được rút từ lần chạy đã kiểm tra trước khi xóa. Kết quả test E6 chưa được đánh giá và được ghi là chưa có. Các hình trong `figures/` được tạo từ dữ liệu, ma trận và chỉ số đã lưu bằng `python -m scripts.tao_hinh_bao_cao`; script không huấn luyện hoặc dự đoán test. Output V2/V3 đã được xóa theo yêu cầu; cần chạy lại notebook 09–10 trước khi tạo lại hình từ output mới. Các hình đã tạo vẫn có thể được dùng để biên dịch báo cáo.

Từ thư mục này, nếu có XeLaTeX và Times New Roman:

```bash
xelatex main.tex
xelatex main.tex
```

Trên máy hiện tại không có lệnh `xelatex`; bản PDF đã được tạo bằng Tectonic dùng engine XeTeX và gói LaTeX. Có thể chạy `tectonic --keep-logs main.tex` khi có Tectonic và phông trên. PDF cần được tạo lại sau mỗi thay đổi nguồn; font trên hệ khác có thể cần cài đặt hoặc thay bằng font Unicode tương thích.

Báo cáo sử dụng trực tiếp `settings.tex`, `cover.tex`, logo và cấu trúc nhiều tệp từ [Template_BCTT_DATN](https://github.com/TrangLe1912/Template_BCTT_DATN), commit `0de06d4fd2af99690d0efe874136afc7d24800b1`. Thông tin sinh viên và giảng viên mặc định của mẫu đã được loại bỏ; trường và hai thành viên được điền theo thông tin người dùng, các mục chưa biết để placeholder. Logo Đại Nam của mẫu được giữ theo trường đã xác nhận. Nội dung và sáu chương được điều chỉnh cho môn Học máy; không áp dụng số trang hay cấu trúc chương thực tập. Repository mẫu không có tệp giấy phép rõ ràng tại thời điểm kiểm tra; nhóm cần xác nhận quyền sử dụng trước khi phát hành hoặc nộp công khai.

Đường học được tính thực tế từ 60 lần fit E1/E4/E6 trên năm fold train chính thức. Để chạy lại và tạo hình, xem `python -m scripts.duong_hoc`; không dùng validation chính thức hoặc test trong phép thử này. Điểm train là dự đoán trên chính mẫu dùng fit, nên K-NN có thể nhận lại chính mẫu đó. Sao lưu kết quả đường học trước khi chạy lại vì lệnh ghi đè output cùng tên.
