# Phân tích tiếp theo: sai lệch phân phối khi nhãn đa số đã đúng

Ngày chuẩn bị: 17/11/2026. Trạng thái: đã viết chương trình độc lập và kiểm tra bằng dữ liệu tổng hợp; CHƯA chạy trên dự đoán của dự án. Chưa có repository, bảng OOF hoặc schema gốc trong phiên này. Không có kết quả nghiên cứu mới trong gói này.

## 1. Câu hỏi nghiên cứu và quyết định giữ nguyên

Mục tiêu: thu hẹp khoảng cách giữa MLLM và phân phối những cách con người diễn giải biểu cảm, gồm mức bất đồng và những lựa chọn cảm xúc, qua A/V/AV. V là điều kiện nghiên cứu có giá trị riêng. Không bắt AV là endpoint duy nhất.

Nhánh merger đã đóng theo báo cáo của người dùng tại commit bd660b9. Delta_within chính +0.015 [-0.011, 0.040]; mean+std +0.007 [-0.022, 0.033]; không PCA -0.007 [-0.018, 0.005], nhánh cuối chưa hội tụ. Không phát hiện suy giảm đáng tin; không phải chứng minh tương đương. Gói này không mở lại merger, không train loss bảo toàn và không mặc định encoder là nguyên nhân.

Bước 0 (người dùng báo, chưa chạy lại): HSE MLP - cont có delta E_ref=-0.052 và delta E_alt=-0.013. Teacher - cont lần lượt -0.058 và -0.018. s1_noI có E_alt=0.218, gần teacher 0.220. Những con số này làm yếu giả thuyết thiếu hụt chủ yếu nằm ở lựa chọn thay thế, nhưng chưa cho biết lỗi mức đồng thuận còn bao nhiêu trên clip đã nhận đúng lớp đa số.

## 2. Phép phân tích bổ sung duy nhất

**Trên cùng các clip mà Qwen và nguồn đối chiếu đều chọn đúng lớp đa số của người, khoảng cách về phân phối phán đoán và mức đồng thuận còn bao nhiêu?**

Không lặp lại temperature fitting, probe hidden state, token injection hay phân rã KL toàn tập. Không huấn luyện mới. Đây là phân tích mô tả hậu nghiệm để cụ thể hóa failure cần giải thích, không phải phép định vị nguyên nhân trong mạng hoặc một method mới.

Ba nguồn ưu tiên cho một lần chạy V: cont (đúng bản calibration đã dùng ở bước 0), HSE MLP, teacher ensemble. Chạy s1_noI như baseline thứ hai với đúng hai nguồn đối chiếu trên. Không tự suy ra calibration tạo ra khác biệt giữa hai checkpoint; nếu muốn so calibration, phải dùng cùng checkpoint trước/sau. A/AV chỉ phân tích nếu có dự đoán phù hợp, không phát sinh lượt GPU.

Không coi ensemble là người thật, ground truth hoặc nguồn độc lập với HSE. Đọc thành phần ensemble trước khi diễn giải đối chứng. Kết quả HSE/teacher tốt hơn là đối chứng thực nghiệm, không chứng minh nút thắt sinh học/nhận thức.

## 3. Tập đánh giá cố định và tránh so các tập khác nhau

- Luôn báo toàn bộ clip trước. Không bỏ clip khó để cải thiện bảng chính.
- Với từng cặp baseline/comparator, báo tập baseline đúng và tập cả hai cùng đúng; mọi delta được tính trên CHÍNH cùng một tập clip.
- Trên clip có đa số duy nhất, chia thêm bốn ô: cả hai đúng, chỉ baseline đúng, chỉ comparator đúng, cả hai không có dự đoán đúng duy nhất. Bốn ô này giúp mô tả nơi lợi thế tập trung, không là phân rã nhân quả.
- Ties trong phiếu người ở một nhóm riêng và vẫn nằm trong toàn tập. Không chọn tùy ý lớp tham chiếu cho tập 'cùng đúng'. Nếu model có nhiều lớp cùng xác suất lớn nhất, không gọi đó là dự đoán đúng duy nhất. Báo coverage và số actor từng nhóm.
- Với số đo xác suất lớp tham chiếu trên toàn tập, lấy trung bình qua các lớp đồng hạng của người.
- So sánh mức chênh giữa các nhóm không phải tác động độc lập của nhận diện nhãn: các nhóm khác độ khó, số clip và thành phần cảm xúc. Không lấy chênh lệch này làm tỷ lệ nguyên nhân.
- Tập 'cùng đúng' được chọn bằng dự đoán hai mô hình và nhãn quan sát. Đây là tập con có điều kiện, không thay thế ước lượng hiệu quả ngoài mẫu toàn tập. Không chọn method/hyperparameter bằng nhóm này rồi tuyên bố xác nhận độc lập trên chính nó.

## 4. Số đo

- JSD(p,q), log tự nhiên, không epsilon smoothing: đo toàn bộ phân phối và xử lý xác suất 0 theo giới hạn 0 log 0=0.
- Sai lệch có dấu xác suất lớp đa số: q_y-p_y. Dương = dành quá nhiều khối lượng cho lớp tham chiếu; âm = quá ít. Đây không tự động là calibration error theo correctness nhị phân.
- Sai lệch tuyệt đối |q_y-p_y| để tránh triệt tiêu sai số trái dấu.
- H(q)-H(p) và |H(q)-H(p)|, entropy chuẩn hóa log(6). Bias dương = dự đoán entropy cao hơn phiếu quan sát. Không suy ra bất đồng quần thể từ dấu này vì ít phiếu gây nhiễu và bias entropy.
- Báo Spearman entropy từng cảm xúc, điểm ước lượng mô tả. Script KHÔNG tự định nghĩa lại rho_within của dự án và không cấp CI cho Spearman. Khi tích hợp repository, gọi lại đúng evaluator đang dùng cho rho_within/partial và bootstrap của nó.
- Không thêm KL trong script: bước 0 đã làm; tránh tạo một quy tắc smoothing mới không tương thích.

Đơn vị chính: clip. CI 95% percentile, 2000 bootstrap actor, cùng mẫu bootstrap cho hai nguồn và mọi chỉ số trong một tập con. CI này điều kiện trên model đã train, dự đoán đã lưu, và phiếu đã quan sát; không gồm bất định seed/model selection/nhóm người chấm. Nhiều phân tích nhóm là mô tả thăm dò, không chọn ô có CI đẹp để tuyên bố phát hiện đã xác nhận. Nhóm một actor không được cấp CI.

## 5. Cách đọc để quyết định, không chuyển module tự động

1. Nếu HSE/teacher tốt hơn về phân phối hoặc mức đồng thuận ngay trong tập cả hai đúng, ta có một failure cụ thể: lợi thế không chỉ xuất hiện ở các clip sửa được nhãn top-1. Cần nêu đúng metric cải thiện; không đánh đồng JSD với xếp hạng entropy.
2. Nếu lợi thế tập trung ở những clip nguồn đối chiếu sửa được nhãn, động lực dùng lợi thế HSE làm bằng chứng cho một cơ chế đặc hiệu về ambiguity yếu đi. Không suy ra 'mô hình đã hiểu ambiguity'.
3. Nếu khác biệt chỉ ở bias trung bình nhưng chưa ở sai số tuyệt đối/xếp hạng, không gọi đó là khôi phục cấu trúc bất đồng.
4. Nếu s1_noI và cont cho các mẫu sai khác nhau, báo tính phụ thuộc checkpoint; không chọn một checkpoint để tạo câu chuyện nhất quán giả tạo.
5. Dù kết quả nào, gói này không chỉ ra encoder/LLM/head là nguyên nhân. Không khởi động một module mới chỉ vì có CI loại 0.

Sai số so phiếu hữu hạn có thể tồn tại ngay với predictor quần thể đúng. Vì vậy chỉ việc q khác p hoặc CI sai số >0 không chứng minh model kém hơn mức lấy mẫu. Mốc người mô phỏng cũ có thể bổ sung khi có cùng generator/smoothing, nhưng không thay bằng một mô phỏng mới hoặc coi là trần.

## 6. Hợp đồng đầu vào độc lập (chưa phải schema repository)

JSONL, một dòng cho mỗi (clip_id, condition, model):

```json
{"clip_id":"example","actor_id":"1001","fold":"0","condition":"V","model":"cont","classes":["Anger","Disgust","Fear","Happy","Neutral","Sad"],"counts":[6,4,0,0,0,0],"probs":[0.9,0.1,0,0,0,0]}
```

Đây chỉ là ví dụ schema; thứ tự classes phải được lấy từ exporter thật, không giả định theo ví dụ. Các model phải có cùng thứ tự lớp, tập clip, số phiếu, actor và outer fold. Script từ chối thiếu clip, duplicate, actor xuất hiện ở nhiều outer test folds, count không nguyên, xác suất không hợp lệ. Không âm thầm lấy giao hoặc tự đổi tên cảm xúc.

Điều kiện phải xác minh bên ngoài script: dự đoán OOF; mọi model/head/teacher không được huấn luyện trên test actors; calibration/PCA/hyperparameter không được fit trên test; phiếu đúng condition; mọi nguồn dùng đúng split dự định. Có fold_id không tự chứng minh không leakage. Cần cung cấp provenance các checkpoint, calibration và thành phần ensemble.

```bash
python -m pip install numpy scipy
python -m unittest -v test_audit.py
python audit_predictions.py --input predictions.jsonl --output audit_cont --baseline cont --comparators hse_mlp teacher --condition V
python audit_predictions.py --input predictions.jsonl --output audit_s1 --baseline s1_noI --comparators hse_mlp teacher --condition V
```

Đầu ra: audit.json; paired.csv (delta=comparator-baseline và CI); strata.csv (coverage); ranks_descriptive.csv (Spearman mô tả, không CI); per_clip.csv để kiểm tra lỗi. Không ghi đè thư mục kết quả đã có. Input SHA256, seed và số lần bootstrap được lưu. JSD/MAE âm nghĩa comparator tốt hơn; với bias có dấu không diễn giải delta âm tự động là tốt hơn.

## 7. Việc còn thiếu để chạy dữ liệu thật

Cần URL/checkout repository và đường dẫn OOF, hoặc một export JSONL theo schema trên. Khi có dữ liệu, việc đầu tiên là nối schema thật, kiểm tra provenance và tái tạo số clip/metric toàn tập; sau đó mới đọc nhóm cùng đúng. Chưa có file thật nên chưa thể xác nhận script tích hợp với analysis/ hoặc các commit của dự án.

Nguồn phương pháp liên quan: Baan et al., 2022, Stop Measuring Calibration When Humans Disagree, https://aclanthology.org/2022.emnlp-main.124/ — phân biệt calibration theo nhãn đa số với khớp các thuộc tính của phán đoán người. Protocol này là đề xuất phân tích của phiên hiện tại, không gán cho bài báo.
