# Kiểm chứng implementation

Đây là kiểm chứng phần mềm, **không phải báo cáo kết quả huấn luyện đầy đủ**.

## Môi trường thực sự đã chạy

- Python 3.12.14, Linux CPU.
- PyTorch 2.15.0+cpu.
- Transformers 4.57.6.
- Matplotlib 3.10.8; pytest 9.2.0.
- Chưa kiểm chứng trên Windows hoặc máy của từng thành viên.

## Kiểm tra tự động

`python -m pytest -q`: **6 passed**.

Bao phủ tokenizer roundtrip/UNK/save-load; split/sequence shifting và full-tail evaluation;
attention đối chiếu `scaled_dot_product_attention` (atol=1e-6, rtol=1e-5);
causality, logits shape, tied weights, parameter formula, finite backward/optimizer;
checkpoint logits; weighted evaluation; seeded generation và top-k;
pipeline tổng hợp và resume CPU đối chiếu với chạy liên tục trong test.
Không suy rộng bitwise resume CPU test sang mọi GPU/hardware.

## Smoke test trên Tiny Shakespeare thật

- Đã tải từ nguồn ghi trong data/README.md.
- SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- Train: 1,003,854 character tokens; validation: 111,540.
- Vocabulary: 66 (bao gồm UNK); parameter count: **421,632**, khớp công thức.
- Baseline chạy **3 optimizer steps**; checkpoint lưu/nạp và generation thành công.
- Prefix validation NLL tại step 0: khoảng 4.1906; step 3: khoảng 3.8037.
- Full validation sau smoke: 111,539 target tokens; NLL khoảng 3.802586; PPL khoảng 44.81693.
- Các con số chỉ xác nhận đường tính toán hoạt động. Không đủ để kết luận chất lượng/overfitting.
- Custom generation sau 3 steps chủ yếu lặp khoảng trắng; đây không phải output trình diễn cuối.

## Entry points đã chạy

1. `python -m scripts.prepare_data`
2. `python -m scripts.train --steps 3`
3. `python -m scripts.evaluate`
4. `python -m scripts.generate --strategy greedy --max-new-tokens 12`
5. `python -m scripts.run_experiments`
6. `python -m scripts.run_experiments --with-pretrained --local-files-only`

Q2 xuất 18 mẫu (6 thiết lập × 3 seed). Q3 xuất 3 cặp custom/GPT-2.
Pretrained GPT-2 đã tải và sinh văn bản thật; không chỉ mock hoặc khởi tạo ngẫu nhiên.
Các code cells của notebook được thực thi tuần tự thành công bằng Python; chưa kiểm tra giao diện Jupyter trên Windows.

## Việc nhóm tiếp tục làm

- Review và giải thích từng module theo vai trò; không xem code có sẵn là đã hiểu cơ chế.
- Cài môi trường trên máy nhóm, smoke test, đo tốc độ rồi khóa ngân sách steps chính thức.
- 1000 steps trong baseline là giá trị khởi đầu, chưa phải lựa chọn tối ưu đã được chứng minh.
- Train đủ, chạy lại Q1–Q3, lưu output và phân tích có dẫn chứng.
- Hoàn thiện nội dung nghiên cứu, 12 slide và diễn tập 20 phút.
- Checkpoint/dataset/output sinh cục bộ không được đẩy lên Git trong commit code này.
