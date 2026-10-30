# Cấu trúc và ghi chú triển khai — Project LLMs Group 4

> Bản chốt ngày 30/10/2026. Tài liệu mô tả cấu trúc dự kiến và tiêu chí triển khai; không khẳng định các module đã được viết hoặc kiểm thử.
> Repository: https://github.com/baochan24/Project_LLMs_Group4

## 1. Mục tiêu và phạm vi

- Presentation môn Deep Learning: 12 slide, 20 phút, 6 thành viên.
- Nghiên cứu mô hình ngôn ngữ autoregressive sử dụng Transformer decoder-only.
- Code combined: custom model nhỏ bằng PyTorch + inference với pretrained GPT-2.
- Custom model minh họa cơ chế nền tảng; không gọi mô hình nhỏ là LLM quy mô lớn.
- Custom model lấy cảm hứng từ GPT-2 nhưng KHÔNG tương thích trực tiếp checkpoint GPT-2.
- Không triển khai fine-tuning, LoRA, ablation bỏ mask/residual hoặc benchmark KV cache.
- Không so sánh trực tiếp perplexity giữa custom character tokenizer và GPT-2 tokenizer.

## 2. Cấu trúc dự kiến

Danh sách dưới đây là các đường dẫn cần triển khai. Việc có tên trong tài liệu không có nghĩa file đã tồn tại.

| Đường dẫn | Chức năng | Phụ trách chính |
|---|---|---|
| README.md | Giới thiệu, cài đặt, môi trường và lệnh chạy đã kiểm chứng | TV6 |
| requirements.txt | Khóa phiên bản dependencies sau smoke test | TV6 |
| .gitignore | Bỏ qua venv, cache, dataset tải về và checkpoint lớn | TV6 |
| configs/baseline.json | Cấu hình model/train/evaluation/generation | TV6 |
| data/README.md | Nguồn, cách chuẩn bị, chia dữ liệu, checksum | TV1 |
| data/raw/input.txt | Tiny Shakespeare tải từ nguồn; không sửa văn bản | TV2 |
| data/processed/ | Train/validation IDs, tokenizer mapping và metadata sinh tự động | TV2 |
| src/__init__.py | Đánh dấu package | TV6 |
| src/tokenizer.py | Character tokenizer, UNK, encode/decode, save/load | TV2 |
| src/data.py | Chia dữ liệu, tạo input–target và batches | TV2 |
| src/attention.py | QKV, heads, causal mask, attention dropout, output projection | TV3 |
| src/model.py | Embedding, FFN, blocks, final LN, tied output head, parameter count | TV4 |
| src/train.py | Train, AdamW groups, clipping, validation, checkpoint/resume | TV4 |
| src/evaluate.py | Token-weighted NLL/loss, PPL và metrics | TV5 |
| src/generate.py | Greedy, sampling, temperature, top-k/top-p, context cropping | TV5 |
| src/pretrained.py | Nạp GPT-2 và tokenizer gốc, inference, xuất mẫu | TV1 |
| src/visualize.py | Biểu đồ loss và hình attention | TV5 |
| src/utils.py | Seed, device, config, lưu JSON và metadata | TV6 |
| scripts/prepare_data.py | Entry point chuẩn bị dữ liệu | TV2 |
| scripts/train.py | Entry point train/resume | TV4 |
| scripts/evaluate.py | Entry point đánh giá checkpoint | TV5 |
| scripts/generate.py | Entry point sinh văn bản custom model | TV5 |
| scripts/run_experiments.py | Điều phối Q1–Q3 và lưu thiết lập/kết quả | TV6 |
| notebooks/presentation_demo.ipynb | Gọi lại src để trình diễn, không viết lại model | TV6 |
| tests/test_data.py | Tokenizer, UNK, sequence shifting, ranh giới tập | TV2 |
| tests/test_model.py | Shapes, attention reference, causality, count, checkpoint | TV3 + TV4 |
| outputs/metrics/ | CSV/JSON loss, PPL và thiết lập chạy | TV5 |
| outputs/figures/ | Hình loss, attention và hình dùng trong slide | TV5 |
| outputs/samples/ | Prompt, cấu hình decoding, seed và văn bản sinh | TV1 + TV5 |
| checkpoints/ | Checkpoint cục bộ; không commit mặc định | TV4 |
| docs/01_Project_Structure.md | Tài liệu cấu trúc và hợp đồng triển khai này | TV6 |
| docs/research_outline.md | Research Outline chính thức: 7 phần | Cả nhóm |
| docs/research_notes.md | Giải thích, phép tính và dẫn nguồn theo từng phần | Cả nhóm |
| docs/experiments.md | Thiết lập, số liệu, phân tích Q1–Q3 và giới hạn | Cả nhóm |
| docs/references.md | Danh mục tài liệu thực sự sử dụng | Cả nhóm |
| slides/ | Slide thành viên và bản tổng hợp cuối | Cả nhóm |

Các thư mục dữ liệu/kết quả có thể được tạo lúc chạy. Không tạo các file Python rỗng rồi xem đó là code hoàn chỉnh.

## 3. Luồng thực thi

1. prepare_data: đọc raw text → chia train/validation → xây vocabulary từ train → mã hóa riêng hai tập → lưu metadata.
2. train: nạp config + dữ liệu → khởi tạo model → training/validation → ghi metrics → lưu best/latest checkpoint.
3. evaluate: nạp checkpoint và tokenizer đi kèm → đánh giá validation → xuất NLL và PPL.
4. generate: nạp cùng checkpoint/tokenizer → prompt → sinh token → lưu mẫu và cấu hình.
5. run_experiments: thu kết quả Q1–Q3; pretrained dùng tokenizer riêng.
6. visualize/notebook: sử dụng kết quả và module đã có để tạo hình/demo.

Các scripts chỉ điều phối. Logic dùng chung nằm trong src. Notebook không chứa bản model thứ hai.

## 4. Dataset và tokenizer đã chốt

- Tiny Shakespeare: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
- Giữ nguyên chữ hoa/thường, dấu câu và xuống dòng; không strip/lowercase dữ liệu.
- Chia theo vị trí ký tự: n_train = floor(0.9 * len(text)); phần còn lại là validation.
- Chia trước khi tạo cửa sổ; mỗi cửa sổ T+1 token chỉ thuộc một tập.
- Vocabulary: các ký tự duy nhất trong train, sắp thứ tự xác định; UNK là một ID đặc biệt riêng.
- Không thêm PAD/EOS trong baseline.
- Lưu đầy đủ mapping, UNK ID và vocabulary size; không xây lại mapping khi inference.
- UNK ID không phải chuỗi ký tự được tách tiếp; khi decode hiển thị ký hiệu <UNK>.
- Mỗi mẫu: input = chunk[:-1], target = chunk[1:]; hai tensor có shape [T], dtype long.
- Batch có shape [B,T]; T = 128 trong baseline.
- Lưu SHA-256 raw file, số ký tự/token từng tập, V và ranh giới chia; không điền số liệu khi chưa đo.

## 5. Cấu hình baseline

| Thuộc tính | Giá trị |
|---|---|
| Framework | PyTorch |
| Architecture | Decoder-only; không cross-attention |
| Layers L | 2 |
| Hidden dimension d | 128 |
| Heads h | 4 |
| Head dimension | 32 |
| FFN dimension m | 512 = 4d |
| Context tối đa C | 128 |
| Batch size | 8 |
| Tokenizer | Character-level + UNK |
| Position | Learned positional embedding |
| Block | Pre-LayerNorm |
| Final normalization | LayerNorm |
| LayerNorm | Có affine scale và bias; epsilon 1e-5 |
| Activation | GELU, dùng cùng biến thể trong code và tài liệu |
| Linear projections | Có bias trong attention và FFN |
| Output head | Không bias, share weight với token embedding |
| Embedding/attention/residual dropout | 0.1 |
| Precision | FP32 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Betas / epsilon | Ghi rõ giá trị thực dùng trong config; không để ẩn khi báo cáo |
| Weight decay | 0.01 cho matrix weights; 0 cho bias và LayerNorm |
| Gradient clipping | Max norm 1.0 |
| Seed | 42 |
| Scheduler / accumulation / mixed precision | Không dùng trong baseline |

Số training steps, chu kỳ validation và phiên bản thư viện được khóa sau smoke test. Phải ghi phần cứng và thời gian đo thực tế; chưa khẳng định chạy nhanh hoặc đủ chất lượng.

Weight tying phải được thiết lập trước khi tạo optimizer; mỗi Parameter chỉ xuất hiện trong một optimizer group.

## 6. Kiến trúc và số tham số

Embedding đầu vào:
X = Dropout(TokenEmbedding(ids) + PositionEmbedding(position_ids))

Mỗi block:
U = X + Dropout(MHA(LN1(X)))
Y = U + Dropout(FFN(LN2(U)))

Attention:
A = softmax(QK^T / sqrt(d_k) + causal_mask)
H = Dropout(A) V

Dropout trên attention weights nằm bên trong MHA; dropout sau attention output projection nằm trên nhánh residual. FFN dùng Linear → GELU → Linear, rồi dropout trước cộng residual.

Đầu ra: L blocks → final LayerNorm → tied output head → logits [B,T,V].

### Công thức

Với d là hidden dimension, m là FFN dimension:
- QKV: 3d² + 3d.
- Attention output projection: d² + d.
- FFN: 2dm + m + d.
- Hai LayerNorm: 4d.

**P_block = 4d² + 2dm + 9d + m.**

Khi m = 4d:
**P_block = 12d² + 13d.**

Với token embedding và output head tied, final LayerNorm có affine:
**P_total = Vd + Cd + L(12d² + 13d) + 2d.**

Cấu hình baseline:
- P_block = 198,272.
- P_total = 128V + 413,184.
- V lấy từ tokenizer thực tế.
- Dropout, activation và causal-mask buffer không có tham số học.
- Nếu bỏ weight tying hoặc thay bias/norm, công thức trên không còn áp dụng nguyên trạng.

Đếm tham số duy nhất bằng model.parameters(); không cộng hai lần cùng weight đã tied.

## 7. Hợp đồng giữa các module

| Thành phần | Đầu vào | Đầu ra/yêu cầu |
|---|---|---|
| Tokenizer | Chuỗi text | IDs; mapping lưu/nạp nhất quán |
| Data | IDs của một tập, T, B | x và y long [B,T], không vượt ranh giới |
| Attention | Float [B,T,d] | Float [B,T,d], chỉ nhìn hiện tại/quá khứ |
| Language model | IDs long [B,T], 1 <= T <= C | Raw logits [B,T,V], không softmax trước cross-entropy |
| Loss | Logits và targets [B,T] | CE/NLL trung bình theo token |
| Generation | Prompt + checkpoint + decoding config | Văn bản/IDs sinh và metadata |
| Checkpoint | Model/config/tokenizer/optimizer/step | Nạp lại để đánh giá hoặc tiếp tục train |

- Không dùng nn.TransformerDecoder mặc định rồi vô tình giữ cross-attention.
- Quy ước mask phải được kiểm tra theo API PyTorch sử dụng; không mặc định mọi API có cùng ý nghĩa boolean mask.
- Khi sampling, lọc top-k/top-p trên logits rồi chuẩn hóa lại; tránh cấu hình loại bỏ toàn bộ token.
- Greedy được triển khai riêng; temperature cho sampling phải > 0.
- Mọi tensor/mask phải phù hợp device và dtype.
- Pretrained GPT-2 dùng tokenizer/model riêng; không truyền character IDs vào GPT-2.

## 8. Huấn luyện, đánh giá và checkpoint

- Training: model.train(); zero_grad → forward → CE → backward → clip → optimizer.step.
- Validation/generation: model.eval() và no_grad(); no_grad không tự tắt dropout.
- Train loss trên đồ thị học có thể là loss quan sát khi train; phải ghi rõ định nghĩa.
- Khi so sánh train và validation để phân tích generalization, đo thêm loss trên mẫu train cố định ở eval mode, cùng cách tính.
- Validation dùng các cửa sổ cố định, không xáo trộn lại giữa các lần đánh giá; mỗi target trong vùng đánh giá được tính một lần, không đếm lặp do cửa sổ chồng lấn.
- Tính tổng NLL / tổng số token mục tiêu, sau đó PPL = exp(mean_NLL).
- Ghi rõ chính sách cửa sổ, phần đuôi và các token không được tính do thiếu ngữ cảnh.
- Best checkpoint chọn theo validation loss; latest dùng tiếp tục công việc.
- Checkpoint chứa model state, optimizer state, step, config, tokenizer mapping và best validation loss.
- Muốn resume tái lập chính xác cần lưu thêm RNG states/sampler state; không tuyên bố exact resume nếu chưa làm.
- Lưu checkpoint tại điểm hoàn tất optimizer step; không hứa Ctrl+C luôn giữ được trạng thái chưa lưu.

## 9. Generation và pretrained

- Custom: lấy logits vị trí cuối → chọn token → nối chuỗi → lặp.
- Giới hạn đầu vào bằng C token cuối; positional IDs bắt đầu lại từ 0 trong cửa sổ.
- Dừng bằng max_new_tokens; không chờ EOS vì baseline không huấn luyện EOS.
- Greedy và sampling (temperature, top-k, top-p).
- Mỗi mẫu lưu prompt, seed, checkpoint, cấu hình và output.
- GPT-2: openai-community/gpt2; chỉ inference, giữ tokenizer gốc.
- Nếu thư viện batching yêu cầu padding cho GPT-2, xử lý riêng và truyền attention mask; không thay đổi tokenizer baseline custom.
- KV cache chỉ giải thích lý thuyết, không có benchmark custom.

## 10. Kiểm chứng bắt buộc

| Kiểm tra | Điều kiện đạt |
|---|---|
| Tokenizer round trip | Chuỗi chứa ký tự đã biết được khôi phục đúng |
| UNK | Ký tự chưa biết ánh xạ đúng ID |
| Shifting | y[t] là token kế tiếp của x[t] trong chunk |
| Split | Không có cửa sổ train/validation vượt ranh giới |
| Shapes | Attention [B,T,d], logits [B,T,V] |
| Reference attention | Cùng Q/K/V, mask, scale, dropout=0; torch.testing.assert_close với tolerance ghi rõ |
| Causality | Thay token sau vị trí t không đổi logits đến t ở eval mode |
| Parameter count | Khớp công thức, không double count tied weight |
| Backward | Loss hữu hạn, gradient hữu hạn, optimizer cập nhật trọng số |
| Checkpoint | Save/load giữ logits trong tolerance ở eval mode |
| Pipeline | Prepare → train smoke → evaluate → reload → generate chạy được |

Đối chiếu numerical reference không phải so logits custom với pretrained GPT-2 khác trọng số.

## 11. Câu hỏi thực nghiệm và đầu ra

### Q1 — Quá trình học
- Train/validation loss theo steps, validation PPL.
- Mẫu sinh trước và sau train.
- Chỉ kết luận overfitting khi có bằng chứng; ghi cách đo loss.
- Đầu ra: metrics CSV/JSON, loss curve, checkpoint và samples.

### Q2 — Decoding
- Cùng prompt và checkpoint; thay từng yếu tố temperature/top-k/top-p.
- Cố định các yếu tố còn lại, lưu các seed.
- Phân tích mẫu quan sát về lặp, đa dạng và mạch văn; không kết luận phổ quát.
- Đầu ra: bảng cấu hình và mẫu sinh.

### Q3 — Custom và GPT-2
- Cùng prompt văn bản tiếng Anh, mỗi model dùng tokenizer riêng.
- Chỉ đối chiếu định tính; nêu chênh lệch quy mô, dữ liệu và tokenizer.
- Không lấy PPL khác tokenizer làm kết luận mô hình tốt hơn.
- Đầu ra: mẫu sinh và nhận xét có giới hạn.

Không điền kết quả minh họa thành số liệu thực nghiệm.

## 12. Research Outline và slide

Research Outline gồm:
I. LLM như mô hình Deep Learning.
II. Từ văn bản đến tensor.
III. Transformer decoder-only.
IV. Huấn luyện.
V. Inference và sinh văn bản.
VI. Đánh giá và giới hạn.
VII. Hiện thực và thực nghiệm.

| Slide | Nội dung | Thành viên | Thời gian |
|---|---|---|---|
| 1 | LLM học điều gì? | TV1 | 1:00 |
| 2 | Phạm vi decoder-only, chain rule, combined | TV1 | 1:30 |
| 3 | Text, tokenizer, input–target | TV2 | 1:30 |
| 4 | Embedding, position và shapes | TV2 | 1:30 |
| 5 | QKV, scaled attention, causal mask; ví dụ 3 token | TV3 | 2:30 |
| 6 | Multi-head, FFN, Pre-LN, residual, dropout | TV3 | 2:00 |
| 7 | Blocks → logits, weight tying, parameter count | TV4 | 1:30 |
| 8 | CE và training step | TV4 | 2:00 |
| 9 | Generation, greedy và sampling | TV5 | 1:30 |
| 10 | Q1: loss, PPL và giới hạn | TV5 | 1:30 |
| 11 | Q2–Q3: demo combined | TV6 | 2:00 |
| 12 | Kết luận và nguồn chính | TV6 | 1:00 |
| Tổng | 12 slide | 6 thành viên | 20:00 |

TV6 là Bảo, phụ trách tích hợp. Mỗi người bàn giao ghi chú có nguồn, module code tương ứng, kiểm chứng và hai slide. Các file dùng chung được phối hợp qua PR; không đẩy chồng thay đổi của nhau.

## 13. Thứ tự triển khai và Git

1. TV6 thiết lập config, quy ước import, môi trường và tài liệu.
2. TV2 chuẩn bị data/tokenizer; TV1 chuẩn bị pretrained inference.
3. TV3 và TV4 thống nhất attention/block interface, hoàn thiện model và tests.
4. TV4 nối training; TV5 nối evaluation/generation/visualization.
5. TV6 kiểm tra pipeline; chạy Q1–Q3 sau khi kiểm chứng đạt.
6. Cả nhóm điền kết quả thật vào báo cáo và slides; diễn tập 20 phút.

Nhánh công việc dự kiến: feature/overview-pretrained, feature/data, feature/attention, feature/model-training, feature/evaluation-generation, feature/integration.
Đây chỉ là quy ước tên; tài liệu không tự tạo các nhánh này.

- Trước khi sửa: cập nhật main, tạo/đổi sang nhánh nhiệm vụ.
- Commit theo phần việc; mở PR về main và ghi cách kiểm chứng.
- Không commit venv, secrets, cache hoặc checkpoint lớn.
- Outputs nhỏ có giá trị cho báo cáo có thể commit; checkpoint lưu bằng phương thức thống nhất riêng.
- README chỉ công bố lệnh đã chạy thành công; command entry points chưa hiện thực không được đánh dấu hoàn tất.

## 14. Điều kiện hoàn thành

- [ ] Cấu hình và môi trường được ghi nhận sau smoke test.
- [ ] Data/tokenizer đúng quy ước và có metadata.
- [ ] Model đúng kiến trúc và parameter count.
- [ ] Kiểm chứng numerical attention và causality đạt.
- [ ] Train/evaluate/checkpoint/generate chạy end-to-end.
- [ ] Q1–Q3 có đầu ra thật và nhận xét đúng phạm vi.
- [ ] GPT-2 inference chạy được hoặc có bản demo đã chạy lưu sẵn.
- [ ] Notebook gọi lại src; không có implementation trùng.
- [ ] 12 slide có nguồn và diễn tập đủ 20 phút.
- [ ] README có hướng dẫn tái lập; không công bố số liệu chưa đo.

## 15. Tài liệu gốc để nghiên cứu

- Transformer paper: https://arxiv.org/abs/1706.03762
- GPT-2 report: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- GPT-2 documentation: https://huggingface.co/docs/transformers/en/model_doc/gpt2
- GPT-2 model card: https://huggingface.co/openai-community/gpt2
- PyTorch documentation: https://docs.pytorch.org/docs/stable/
- Tiny Shakespeare: https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare

Nhóm phải đọc và dẫn đúng phần nguồn thực sự sử dụng; danh sách này không thay cho việc xác minh nội dung.
