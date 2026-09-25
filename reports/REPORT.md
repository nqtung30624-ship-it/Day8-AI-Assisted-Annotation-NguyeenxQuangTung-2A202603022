# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyen Quang Tung

Công cụ gán nhãn đã dùng: CVAT

## 1. Dữ liệu và cách chia tập

Tập pool và tập test được chia theo trục thời gian và có vùng đệm ở giữa để hạn chế việc các khung hình rất giống nhau xuất hiện đồng thời ở hai tập. Dữ liệu có 268 ảnh trong pool và 20 ảnh test. Giữa các vùng pool và test có các frame `buffer`.

Cách chia này phù hợp với dữ liệu video vì các frame liên tiếp thường có nội dung rất giống nhau. Nếu chia ngẫu nhiên, các frame gần nhau về thời gian có thể bị đưa vào cả pool và test. Khi đó mô hình có thể được huấn luyện trên một cảnh gần như giống với cảnh dùng để đánh giá, làm số đo trên test có xu hướng cao hơn và không phản ánh tốt khả năng tổng quát sang một đoạn thời gian khác.

Trong bộ dữ liệu này, test gồm 20 ảnh và được tách khỏi pool bằng các vùng đệm theo thời gian. Vì vậy việc đánh giá được thực hiện trên những đoạn thời gian khác với dữ liệu được chọn để gán nhãn.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0:

> Round 0 — YOLOv8n cold start, 0 ảnh train, 0 box train, AP50 = 0.7714, precision = 0.9249, recall = 0.4888, F1 = 0.6396, TP = 197, FP = 16, FN = 206.

Mô hình khởi đầu sử dụng `yolov8n.pt` với kích thước ảnh 960 và được đánh giá trên 20 ảnh test. Bộ tham chiếu có 403 box, trong đó 14 box rất nhỏ được bỏ qua khi đánh giá.

Kết quả cho thấy precision khá cao (0.9249) nhưng recall chỉ đạt 0.4888. Điều này có nghĩa là những box mà mô hình dự đoán thường khá chính xác, nhưng mô hình vẫn bỏ sót nhiều xe.

Theo kích thước xe, recall khác nhau rõ rệt:

| Kích thước | Recall | Số box tham chiếu |
|---|---:|---:|
| Small | 0.1818 | 66 |
| Medium | 0.5473 | 296 |
| Large | 0.5610 | 41 |

Recall của xe nhỏ chỉ 0.1818, thấp hơn nhiều so với xe medium và large. Điều này cho thấy mô hình cold start gặp khó khăn rõ rệt với những xe có kích thước nhỏ trong ảnh.

Tuy nhiên, không nên kết luận mọi trường hợp không có box đều là lỗi của mô hình. Nhãn tham chiếu của tập test được tạo bởi mô hình và chưa được người rà thủ công. Vì vậy, với những trường hợp xe rất nhỏ, chỉ còn hai điểm sáng hoặc nằm trong vùng khó quan sát, cần người kiểm tra lại nhãn tham chiếu trước khi kết luận rằng model đã dự đoán sai. Guideline cũng quy định box cao dưới khoảng 16 pixel có thể được bỏ qua khi chấm điểm.

## 3. Chiến lược chọn mẫu

Chiến lược sử dụng là `uncertainty`, chọn 12 ảnh từ 268 ảnh trong pool.

Điểm chọn mẫu được tính theo:

`score = W_U·U + W_A·A + W_D·D`

Trong đó:

- `U` thể hiện mức độ bất định của model đối với ảnh.
- `A` thể hiện mức độ ảnh có những đối tượng khó xử lý/cần rà nhãn.
- `D` thể hiện mức độ khác biệt hoặc tránh chọn những ảnh quá gần với các ảnh đã xét.
- `W_U`, `W_A`, `W_D` là trọng số của từng thành phần.

`MIN_GAP_S = 2.0` được dùng để tránh chọn các frame quá gần nhau về thời gian. Điều này quan trọng đối với video vì hai frame liên tiếp có thể gần như cùng một cảnh; nếu chọn cả hai thì chi phí gán nhãn tăng nhưng lượng thông tin mới có thể không tăng tương ứng.

Ba frame tiêu biểu trong lô được chọn:

- **frame_0182**: rank 1, score `0.9591`, `U = 0.9182`, `A = 1.0`, `D = 1.0`. Đây là frame có điểm tổng cao nhất trong các ứng viên được chọn, đồng thời có 18 box ambiguous.
- **frame_0369**: rank 2, score `0.9324`, `U = 0.9315`, `A = 0.8889`, `D = 1.0`. Mức bất định cao nên frame này được ưu tiên rà nhãn.
- **frame_0380**: rank 3, score `0.9170`, `U = 0.9340`, `A = 0.8333`, `D = 1.0`. Frame này cũng có mức bất định cao và có nhiều box cần xem xét.

Một frame khác là **frame_0392**: rank 15, nhưng vẫn được chọn với `U = 0.9747`, là mức bất định cao hơn cả ba frame trên. Tuy nhiên `A = 0.6667`, thấp hơn các frame tiêu biểu trên, nên điểm tổng chỉ là `0.8874`.

Điều này cho thấy điểm bất định không phải là bằng chứng rằng một ảnh chắc chắn sẽ giúp model cải thiện sau fine-tune. Nó chỉ cho biết model đang không chắc chắn ở ảnh đó. Hiệu quả thực tế còn phụ thuộc vào việc nguyên nhân của sự bất định có được sửa bằng nhãn chất lượng hay không và ảnh đó có cung cấp thông tin khác với các ảnh đã chọn hay không.

## 4. Các vòng học chủ động (active learning)

### Bảng kết quả

Hiện dữ liệu đã có kết quả **round 0** và nhãn đã sửa cho **round 1**, nhưng chưa có `metrics_round1.json` và `rounds_table.md` trong gói dữ liệu hiện tại. Vì vậy chưa thể đưa ra AP50 sau fine-tune hoặc kết luận định lượng rằng AP50 tăng hay giảm mà không tự tạo số liệu.

| Vòng | Ảnh train | Box train | AP50 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Round 0 - cold start | 0 | 0 | 0.7714 | 0.9249 | 0.4888 |
| Round 1 - sau fine-tune | 12 | 328 | Chưa có trong dữ liệu hiện tại | Chưa có | Chưa có |

### Mức độ sửa nhãn ở round 1

Round 1 gồm 12 ảnh:

`frame_0099, frame_0107, frame_0182, frame_0187, frame_0227, frame_0270, frame_0312, frame_0326, frame_0331, frame_0369, frame_0380, frame_0392`.

Sau khi rà và sửa trên CVAT, tổng số box trong nhãn cuối là **328 box**.

Theo kết quả đóng gói nhãn của round 1:

- Box được giữ nguyên (`accepted`): **80**
- Box được chỉnh sửa (`edited`): **66**
- Box bị xoá (`deleted`): **23**
- Box được thêm mới (`added`): **182**

Như vậy phần nhãn người rà đã có nhiều thay đổi so với pre-label của model. Đặc biệt, số box được thêm mới lớn, cho thấy pre-label đã bỏ sót nhiều xe trong các ảnh được chọn.

### Phân biệt ba loại bằng chứng

Quan sát độc lập trước khi xem pre-label phải được phân biệt với nhãn AI và kết quả model sau train:

1. **BLIND_SCAN** là quan sát của người trước khi xem nhãn AI.
2. **REVIEW_LOG / round1_diff** phản ánh những thay đổi đối với pre-label, ví dụ giữ nguyên, sửa, xoá hoặc thêm box.
3. **metrics_round1 / compare_round1** mới phản ánh model sau khi fine-tune.

Vì vậy việc một box được người sửa không đồng nghĩa với việc model sau fine-tune chắc chắn sẽ dự đoán tốt hơn ở test.

### Một ca khó theo guideline

Một tình huống khó là xe ở xa, thân xe tối và chủ yếu nhìn thấy cụm đèn. Guideline quy định nếu vẫn có thể suy ra đường viền thân xe thì phải vẽ box quanh phần thân xe có thể xác định, không chỉ khoanh hai điểm sáng. Ngược lại, nếu xe quá nhỏ và box cao dưới khoảng 16 pixel thì có thể gán hoặc không gán vì loại box này được bỏ qua khi chấm điểm.

Với xe bị che một phần, box cũng chỉ bao quanh phần nhìn thấy. Các xe đứng sát nhau phải được vẽ thành hai box riêng, không gộp thành một box.

Do chưa có `compare_round1.jpg` trong dữ liệu hiện tại, chưa thể xác định bằng chứng cụ thể từ ảnh rằng một ca nào đã tốt lên hoặc xấu đi sau fine-tune.

## 5. Kết luận và giới hạn

Ở cold start, model đạt AP50 **0.7714**, precision **0.9249** và recall **0.4888** trên 20 ảnh test. Điểm đáng chú ý nhất là recall của xe nhỏ chỉ **0.1818**, thấp hơn nhiều so với xe medium (**0.5473**) và large (**0.5610**).

Round 1 đã chọn 12 ảnh bằng chiến lược uncertainty và sau khi rà nhãn có tổng cộng 328 box. Trong quá trình sửa, có 80 box được giữ nguyên, 66 box chỉnh sửa, 23 box xoá và 182 box được thêm mới. Điều này cho thấy bước rà nhãn của người có vai trò đáng kể trong việc bổ sung và sửa các pre-label của model.

Tuy nhiên, chưa thể kết luận round 1 tốt hơn hay xấu hơn cold start về AP50 vì dữ liệu hiện tại chưa có `metrics_round1.json`. Do đó không nên tự ghi một mức AP50 round 1 hoặc tự kết luận AP50 tăng/giảm.

Nếu tiếp tục vòng sau, hai nhóm ca đáng ưu tiên là:

1. **Các xe nhỏ/khó nhìn**, vì recall small của cold start chỉ đạt 0.1818.
2. **Các frame có độ bất định cao nhưng điểm đa dạng thấp**, chẳng hạn những frame có nhiều đối tượng khó nhưng có nguy cơ gần với các cảnh đã chọn. Cần cân nhắc chi phí rà nhãn với lượng thông tin mới thu được.

Việc chọn mẫu cũng cần tránh quá nhiều frame gần nhau trong video. `MIN_GAP_S = 2.0` giúp giảm nguy cơ dành công sức cho những ảnh gần trùng nhau.

Có ba giới hạn quan trọng khi diễn giải kết quả:

- Tập test chỉ có **20 ảnh**, nên AP50 có thể dao động đáng kể.
- Các box rất nhỏ có quy tắc bỏ qua khi đánh giá, nên kết quả không phản ánh đầy đủ mọi xe rất xa.
- Nhãn tham chiếu của test được tạo bởi model và chưa được người rà thủ công. Vì vậy AP50 ở đây đo mức khớp với bộ tham chiếu đó, chứ không phải bằng chứng tuyệt đối về chất lượng phát hiện xe ngoài thực tế.

Nếu AP50 sau fine-tune giảm, trước khi train thêm cần kiểm tra **nhãn round 1**, đặc biệt các box được thêm/sửa/xoá; kiểm tra các ca khó theo guideline; kiểm tra xem có box sai hoặc không nhất quán giữa các ảnh hay không; sau đó đối chiếu `round1_diff.md`, `REVIEW_LOG.csv` và `compare_round1.jpg` để xác định vấn đề nằm ở nhãn, dữ liệu hay kết quả model. Không nên sửa nhãn test hoặc chỉnh trực tiếp số đo để làm AP50 tăng.