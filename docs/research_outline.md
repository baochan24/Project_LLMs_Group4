# Research Outline đã chốt

Phạm vi, cấu hình và phân công chi tiết: [01_Project_Structure.md](01_Project_Structure.md).

1. **LLM như mô hình Deep Learning:** language modeling, chain rule, next-token prediction; quy mô; bảng ba họ kiến trúc và phạm vi decoder-only.
2. **Từ văn bản đến tensor:** corpus, chia train/validation, character tokenizer, vocabulary/UNK, sequence shifting, learned position và shapes.
3. **Transformer decoder-only:** QKV; scaled dot-product attention; ví dụ 3 token; causal mask; multi-head; FFN/GELU; Pre-LN/residual/dropout; tied head; số tham số.
4. **Huấn luyện:** self-supervised objective; CE/NLL; forward/backward; AdamW; clipping; train/eval; checkpoint. Post-training chỉ tổng quan.
5. **Inference:** vòng autoregressive, greedy/sampling/temperature/top-k/top-p, context cropping. KV cache chỉ lý thuyết.
6. **Đánh giá:** train/validation NLL và PPL; điều kiện so sánh; giới hạn dữ liệu/context/quy mô và diễn giải mẫu sinh.
7. **Thực nghiệm:** Q1 loss/PPL qua steps; Q2 decoding; Q3 đối chiếu định tính custom/GPT-2. Không benchmark PPL khác tokenizer.

Custom model không phải pretrained GPT-2 và không nạp checkpoint GPT-2. Các mục nghiên cứu được chọn lọc thành 12 slide, không trình bày hết trong 20 phút.
