# Lab 21 — Evaluation Report

**Họ tên**: Trần Hữu Đức  **MSSV**: 2A202602459  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng)`

> Mọi con số dưới đây được đo thực tế trên môi trường Colab GPU T4 và khớp chính xác 100% với các tệp artefact trong thư mục `results/`.

---

## 1. Setup

| Thông số | Giá trị thực nghiệm |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 mẫu / 25 mẫu (cố định split seed 42) |
| `max_length` | 1024 (p95 đo được thực tế là 98 tokens, gợi ý 256 tokens; giữ 1024 theo cấu hình chuẩn của tier T4 để đảm bảo an toàn bộ nhớ và khả năng mở rộng) |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 epochs / 30 optimizer steps (gradient accumulation = 16, batch size = 1) |

**Template có giữ khối `<think>` không?** Có (`VERDICT: reasoning preserved — safe to train on traces` trong `results/template_check.json`). Chuỗi render chat template của Qwen3.5 giữ nguyên cặp thẻ `<think>...</think>`, đảm bảo không bị nuốt mất reasoning block.

---

## 2. Mask proof (NB1)

| Tiêu chí | Kết quả |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng số tokens được tính loss) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss (trích xuất từ `results/mask_proof.json`):

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3228.6 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1013.3 |
| (c) LoRA fine-tune | 0.980 | 0.744 | 1.000 | 1429.3 |

**Prompt (b) có thật sự mạnh hơn (a) không?** Có, vượt trội hoàn toàn: target tăng từ 0.000 lên 0.765, format đạt 1.000 (so với 0.000 của prompt ngây thơ vì prompt ngây thơ không ép được cấu trúc JSON). Prompt (b) không bị chỉnh sửa hay làm yếu đi, giữ nguyên mã SHA `719e74d3b6232053` như repo mặc định để đảm bảo tính liêm chính khoa học của phép so sánh.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6276 | 0.9800 | 405.3 | 8.78 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 0.0001 | 0.5388 | 0.9700 | 273.9 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5702 | 0.0000 | 406.3 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.9400 | 469.1 | 3.86 |

### Trả lời ba câu hỏi giải phẫu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Trên tập đánh giá target mục tiêu, `correct` đã **chiến thắng** `attn_only` (0.9800 so với 0.9700). Tuy nhiên, nếu chỉ nhìn vào train loss ở NB4, `attn_only` lại có loss thấp hơn đáng kể (0.5388 so với 0.6276 của `correct`). Sự đảo ngược thứ tự giữa train loss và target accuracy là bằng chứng thực nghiệm rõ ràng nhất cho thấy: rank cực cao ($r=283$) trong attention-only dễ dàng ép loss huấn luyện xuống sâu nhờ năng lực ghi nhớ (memorization), nhưng không mang lại sự tổng quát hóa tốt bằng việc rải đều adapter ở toàn bộ các lớp tuyến tính (`text-linear`) với rank vừa phải ($r=16$). Vị trí đặt adapter chính là đòn bẩy cấu trúc quan trọng hơn rank thuần túy.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Chỉ giảm learning rate đi 10 lần (từ 1e-4 xuống 1e-5 theo thang của full fine-tune), đường loss của `wrong_lr` gần như phẳng lì suốt 30 steps và kết thúc ở mức cao 1.5702, trong khi bản chuẩn hội tụ xuống 0.6276. Hậu quả là mô hình hoàn toàn thất bại trên tác vụ đích với điểm target và format đều bằng 0.0000. Nếu một kỹ sư chỉ nhìn vào đồ thị loss đi ngang mà không nắm rõ đặc thù của LoRA, họ sẽ dễ kết luận sai lầm rằng "dữ liệu quá phức tạp", "mô hình 4B không đủ năng lực", hoặc "LoRA không học nổi định dạng JSON", trong khi nguyên nhân thực sự đơn giản chỉ là learning rate quá nhỏ không đủ để dịch chuyển các ma trận adapter $A$ và $B$ khởi tạo gần 0.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
Cấu hình `qlora` tiết kiệm được gần 5 GB VRAM (chỉ chiếm đỉnh 3.86 GB VRAM so với 8.78 GB của bản 16-bit), cho phép chạy thoải mái trên các GPU cấu hình thấp. Tuy nhiên, cái giá phải trả là độ chính xác target bị tụt từ 0.9800 xuống 0.9400, đồng thời thời gian huấn luyện kéo dài thêm ~16% (469.1 giây so với 405.3 giây) do phụ phí giải lượng tử hóa (dequantization overhead) ở mỗi forward/backward pass. Kết quả thực nghiệm này hoàn toàn ủng hộ khuyến nghị của vendor trong deck §13: trên GPU T4 16GB, ta hoàn toàn đủ VRAM để huấn luyện fp16/bf16 LoRA trực tiếp mà không cần đánh đổi độ chính xác và tốc độ bằng QLoRA 4-bit.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.2150` · `regression Δ = -0.0467` · `valid_trace_rate = 0.0`

### Diễn giải phán quyết:
Cổng hồi quy ra phán quyết `FAILED` bởi vì chỉ số năng lực tổng quát (`regression`) bị tụt giảm 0.0467 điểm (từ 0.7911 xuống 0.7444), vượt quá ngưỡng dung sai cho phép là 0.020. Mặc dù trên tác vụ mục tiêu CSKH, bản LoRA fine-tune đã thể hiện sự vượt trội áp đảo khi nâng điểm target từ 0.765 lên 0.980 (tăng +21.5%) và định dạng chuẩn xác 100%, nhưng việc huấn luyện chuyên biệt hóa 2 epochs trên 225 mẫu JSON thuần túy đã gây ra hiện tượng quên thảm họa nhẹ (catastrophic forgetting). Mô hình bắt đầu mất dần một phần khả năng trả lời chính xác các câu hỏi kiến thức phổ thông ngoài miền. Theo đúng khuyến nghị của bài học (deck §6.3), đây là một phát hiện khoa học bình thường và có giá trị: để triển khai production một cách an toàn, đội ngũ phát triển cần bổ sung từ 1% đến 5% dữ liệu đàm thoại/kiến thức tổng quát (replay buffer) vào tập huấn luyện để duy trì năng lực nền của mô hình.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper không gọi. Hỏi cho biết thôi. Shop hỗ trợ tốt. | `{"intent": "van_chuyen", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment": "tich_cuc"}` | Đúng 3/4 trường (sai urgency) | `{"intent": "van_chuyen", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment": "tich_cuc"}` | ✅ FT thắng: Nhận diện chính xác ngữ cảnh không vội ("hỏi cho biết thôi") |
| 2 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. Mong shop phản hồi. Nhờ shop kiểm tra. | `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | Trả về thừa giải thích text | `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | ✅ FT thắng: Bắt trọn JSON gọn gàng, đúng nhãn 4 trường |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | `urgency: thap` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | ❌ **FT thua**: Sai trường `urgency` (dự đoán `trung_binh` thay vì `thap`) |
| 4 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều. | `{"intent": "san_pham_loi", "urgency": "thap", "product": "áo khoác gió", "sentiment": "tich_cuc"}` | `urgency: thap` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}` | ❌ **FT thua**: Sai trường `urgency` do thiên kiến cụm "khi nào tiện" |
| 5 | Chào shop, mình đặt nồi chiên không dầu mã đơn VN949966. Hoàn tiền. Khi nào tiện. Shop xem giúp. | `{"intent": "hoan_tien", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | `urgency: thap` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | ❌ **FT thua**: Tiếp tục đoán sai trường `urgency` thành `trung_binh` |

### Có mẫu chung nào ở các ca FT thua không?
Có một mẫu chung rất rõ ràng: ở tất cả các ca fine-tune bị trừ điểm (đạt score 0.75 thay vì 1.0), mô hình đều phân loại đúng 3 trường `intent`, `product` và `sentiment`, nhưng luôn đoán sai trường `urgency` thành `trung_binh` khi gặp cụm từ *"Khi nào tiện"*. Trong khi đó, nhãn chuẩn của bài toán quy định đây là mức độ `thap`. Mô hình fine-tune dường như đã hình thành một thiên kiến ưu tiên nhãn đa số (`trung_binh`) cho trường độ khẩn cấp khi người dùng dùng câu từ lịch sự mang tính thăm dò.

---

## 7. Kết luận & điều tôi học được

**Kết luận:**  
Bản fine-tune LoRA (`correct`) đã chứng minh được sự vượt trội vượt bậc trên bài toán nghiệp vụ cụ thể: tăng độ chính xác phân loại từ 76.5% (Baseline b) lên 98.0%, định dạng JSON đạt chuẩn 100% và giảm kích thước prompt đầu vào đáng kể mà không cần phải dựa dẫm vào các prompt hướng dẫn dài dòng. Tuy nhiên, nếu hỏi có nên deploy ngay lập tức bản checkpoint này lên hệ thống production hay không, câu trả lời là **chưa nên**. Lý do là kết quả cổng hồi quy chỉ ra mô hình bị suy thoái năng lực chung (regression drop -0.0467), có thể dẫn đến việc xử lý kém nếu khách hàng hỏi các câu nằm ngoài phân phối CSKH hẹp. Đòn bẩy quyết định thành công trong bài lab này không nằm ở rank cao hay các kỹ thuật lượng tử hoá phức tạp, mà nằm ở: (1) Tính đúng đắn tuyệt đối của Loss Mask (chỉ tính loss trên output), (2) Đặt adapter đồng đều trên toàn bộ text-linear, và (3) Thang đo learning rate phù hợp (~10x so với full fine-tune).

**Ba điều tôi học được:**
1. **Loss mask là điều kiện tiên quyết số một:** Nếu che mask sai (như chế độ `everything`), mô hình sẽ học cách lặp lại câu hỏi của người dùng và phá hỏng toàn bộ pipeline huấn luyện.
2. **Rank không phải là đòn bẩy thần kỳ:** Thí nghiệm matched rank đã chứng minh rằng ép rank lên thật cao ($r=283$) chỉ giúp giảm loss huấn luyện trên tập train nhờ ghi nhớ, nhưng trên tập test thực tế thì cấu hình đặt adapter toàn diện ở rank chuẩn ($r=16$) vẫn giành chiến thắng (0.98 so với 0.97).
3. **Giá trị của việc đo baseline trước khi train:** Đóng băng Baseline (b) giúp chúng ta có một cái nhìn tỉnh táo, tránh tự lừa dối bản thân bằng những chỉ số ảo như Perplexity, và dũng cảm chấp nhận phán quyết hồi quy để cải thiện giải pháp.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**  
Tôi sẽ tạo một tập dữ liệu hỗn hợp bằng cách trộn thêm 3% dữ liệu hội thoại tiếng Việt thông thường (general domain replay) vào 225 mẫu CSKH và huấn luyện lại cấu hình `correct`, với mục tiêu vượt qua cổng hồi quy (regression gate) để đạt trạng thái sẵn sàng triển khai thực tế.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap: Đã kiểm chứng merge model thành công, độ chính xác trước và sau merge không đổi (`target = 0.9800`, sai lệch $\Delta = 0.0000 \ge -0.01$, file `results/merge_check.json`), đồng thời kiểm chứng khả năng nạp và hoán đổi đa adapter linh hoạt.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
