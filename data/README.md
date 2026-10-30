# Dữ liệu — TV1/TV2

Nguồn: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

Chạy từ repository root: `python -m scripts.prepare_data`.
Nếu đã có file, dùng `--raw đường_dẫn/input.txt`. Không ghi đè raw file đã có.

Chia theo ký tự 90/10 trước khi tạo cửa sổ. Vocabulary chỉ từ train, UNK ID=0.
Không PAD/EOS, không lowercase/strip. `metadata.json` lưu SHA-256 và kích thước thật.
Raw text và processed tensors được sinh lúc chạy, không commit mặc định.
Không coi corpus tổng hợp trong tests là Tiny Shakespeare hoặc kết quả nghiên cứu.
