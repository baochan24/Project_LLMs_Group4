# Project LLMs — Group 4

Presentation môn Deep Learning: **decoder-only Transformer**, **12 slide / 20 phút**, **6 thành viên**.
Code combined gồm custom model nhỏ bằng PyTorch và pretrained GPT-2 inference.
Custom model không phải LLM quy mô lớn và không nạp trực tiếp GPT-2 weights.

## Tài liệu nhóm

- [Cấu trúc, phạm vi và phân công](docs/01_Project_Structure.md)
- [Research Outline](docs/research_outline.md)
- [Ghi chú nghiên cứu](docs/research_notes.md)
- [Thực nghiệm Q1–Q3](docs/experiments.md)
- [Kiểm chứng code và môi trường đã chạy](docs/validation.md)
- [Nguồn tham khảo](docs/references.md)

## Cài đặt trên Windows PowerShell

Chạy trong thư mục repository. Môi trường kiểm chứng cụ thể nằm trong validation.md; chưa kiểm chứng trên máy Windows của nhóm.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Có thể dùng Python 3.12 thay 3.11. Luôn chọn đúng interpreter/kernel của venv trong VS Code.
Lệnh Python bên dưới viết ngắn là `python`; nếu chưa activate venv, thay bằng `.\.venv\Scripts\python.exe`.
Trên Linux, tạo venv bằng `python3 -m venv .venv` và dùng `.venv/bin/python`.

Notebook tùy chọn: `python -m pip install -r requirements-notebook.txt`.

## Kiểm tra và chạy pipeline

Tất cả lệnh chạy từ repository root; dùng `-m` để import `src` nhất quán.

```bash
python -m pytest -q
python -m scripts.prepare_data
python -m scripts.train --config configs/baseline.json --steps 3
python -m scripts.evaluate --checkpoint checkpoints/best.pt
python -m scripts.generate --checkpoint checkpoints/best.pt --strategy greedy --max-new-tokens 40
python -m scripts.run_experiments --checkpoint checkpoints/best.pt
```

`--steps 3` là **smoke test phần mềm**, không đủ để kết luận mô hình đã học tốt.
Baseline có sẵn target 1000 steps như ngân sách khởi đầu, không phải số steps đã xác nhận tối ưu. Sau khi kiểm tra tốc độ máy nhóm, ghi cấu hình chạy chính thức và không thay giữa các thí nghiệm đối chiếu.

### Huấn luyện tiếp

```bash
python -m scripts.train --config configs/baseline.json --resume checkpoints/latest.pt --steps 1000
```

`--steps` là tổng optimizer steps muốn đạt, không phải số steps cộng thêm.
Resume yêu cầu cùng config, tokenizer và checksum dữ liệu. Lưu latest/best khi đánh giá và khi kết thúc target steps; Ctrl+C có thể mất phần chưa lưu sau checkpoint gần nhất. Không xóa checkpoint nếu muốn tiếp tục.

### Sampling và pretrained

```bash
python -m scripts.generate --temperature 0.7 --top-k 10 --top-p 1.0 --seed 42
python -m scripts.run_experiments --with-pretrained
```

Runner lưu Q2 từ custom model. `--with-pretrained` chạy thêm Q3, cần tải GPT-2 và đủ RAM. Có cache thì thêm `--local-files-only`. Dataset và pretrained weights tải khi gọi lệnh, không nằm trong Git.

## Kiến trúc đã hiện thực

- Character tokenizer: train-only vocabulary + UNK; Tiny Shakespeare chia 90/10 trước windowing.
- 2 blocks, 4 heads, d=128, FFN=512, context=128, learned position.
- Causal self-attention, Pre-LN, final LayerNorm, GELU chính xác (`approximate='none'`).
- Dropout 0.1 tại embedding, attention weights và hai nhánh residual.
- Tied token embedding/output head, tất cả attention/FFN Linear có bias; head không bias.
- AdamW, gradient clipping, FP32; không scheduler/gradient accumulation.
- Không PAD/EOS trong custom baseline; generation dừng bằng max_new_tokens.

`P_block = 12d² + 13d`; `P_total = Vd + Cd + L*P_block + 2d`.
Với V=66 của dữ liệu đã tải: **421,632 tham số duy nhất**. Vocabulary thực tế được ghi trong metadata.

## Đầu ra

- `data/processed/`: IDs, tokenizer và checksum/split metadata.
- `checkpoints/best.pt`, `latest.pt`: model/optimizer/config/tokenizer/step/RNG.
- `outputs/metrics/training.csv`: train eval loss, validation loss/PPL và batch loss.
- `outputs/metrics/validation.json`: đánh giá toàn bộ validation.
- `outputs/samples/`: mẫu trước train, generation, Q2/Q3.
- `outputs/figures/loss.png`: đồ thị NLL.
- `notebooks/presentation_demo.ipynb`: demo gọi lại src và hình attention; cần checkpoint có sẵn.

Mặc định dữ liệu, checkpoint và kết quả sinh không commit. Tài liệu validation ghi kết quả kiểm chứng thật; chưa có kết luận nghiên cứu Q1–Q3 hoặc slide hoàn chỉnh. Không so PPL giữa tokenizer khác nhau.
