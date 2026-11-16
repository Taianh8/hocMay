# Công bố sử dụng AI

Trong quá trình thực hiện, OpenAI Codex đã hỗ trợ nhóm ở các phần sau:

- gợi ý và chỉnh sửa mã cho pipeline tiền xử lý, trích xuất đặc trưng, đánh giá, giao diện và các kiểm tra bất biến;
- hỗ trợ thiết kế và đọc các bảng thí nghiệm E1--E7, kiểm định chéo, đối chứng E3/E7 và các biến thể độ phân giải/bin màu;
- kiểm toán dữ liệu, split, ảnh trùng, manifest, checksum và sự nhất quán giữa dự đoán CSV với các chỉ số đã lưu;
- soạn thảo, biên tập và sắp xếp báo cáo LaTeX, README, hướng dẫn đóng góp, bàn giao và kiểm toán quy trình;
- thực hiện các lệnh thí nghiệm theo cấu hình đã có, tạo artifact và tính lại chỉ số từ artifact; không tự tạo điểm số hoặc thay thế artifact bằng giá trị suy đoán.

Nhóm đã đối chiếu nội dung báo cáo với các file trong `results/v5` và `results/final_test`, kiểm tra các công thức/tổng hợp bằng dữ liệu đã lưu, đọc lại mã nguồn và ghi rõ những điểm không thể kết luận. Đặc biệt, nhóm giữ nguyên sự kiện test E1/E6 đã chạy, không gán kết quả đó cho E6_bins32 và không chạy lại test để che lấp sai lệch lựa chọn. Nhóm chịu trách nhiệm cuối cùng về mã, số liệu, diễn giải, việc rà soát thủ công và tuân thủ quy định học phần.
