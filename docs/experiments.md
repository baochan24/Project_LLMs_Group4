# Thực nghiệm

Chưa có kết quả nghiên cứu đầy đủ. Smoke test chỉ kiểm tra phần mềm.

## Q1
Chạy train, xem outputs/metrics/training.csv và loss.png.
Các cột train_eval_loss/validation_loss đo ở eval mode trên prefix cố định có giới hạn eval_tokens.
Đánh giá đầy đủ validation bằng scripts.evaluate; không trộn kết quả full với prefix mà không ghi rõ.
Nonoverlapping target windows: mọi target sau token đầu được tính đúng một lần; cửa sổ sau dùng token biên trước đó làm context đầu; giữ phần đuôi. Cách này không cung cấp context dài ở đầu mỗi chunk.

## Q2
Runner thay temperature hoặc top-k hoặc top-p từng yếu tố; seeds 42,43,44.
Greedy là mẫu đối chiếu riêng. Lưu prompt/checkpoint/config cùng output.

## Q3
Bật --with-pretrained để chạy GPT-2 thật. Có thể cần mạng và đủ RAM.
Đối chiếu định tính cùng prompt tiếng Anh; tokenizer và quy mô khác nhau. Không so PPL chéo.

## Biểu mẫu báo cáo
Ghi hardware, phiên bản, checksum dataset, step/checkpoint, NLL/PPL, hình, mẫu sinh và giới hạn.
Không điền số liệu giả định vào bảng kết quả.
