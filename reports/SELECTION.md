# Vì sao chọn lô này?

## 1. Năm frame ưu tiên nếu chỉ có ngân sách rà năm ảnh

Dựa trên các frame trong lô 12 ảnh được chiến lược `uncertainty` chọn, nếu chỉ có ngân sách rà 5 ảnh, tôi ưu tiên:

| Ưu tiên | Frame | Điểm | Thời điểm (s) | Thứ tự (rank) | Lý do |
|---|---|---:|---:|---:|---|
| 1 | `frame_0182.jpg` | 0.9591 | 72.8 | 1 | Điểm cao nhất, độ bất định `U=0.9182`, có 18 box ambiguous và `A=D=1.0`, nên có nhiều trường hợp cần người kiểm tra. |
| 2 | `frame_0369.jpg` | 0.9324 | 147.6 | 2 | Điểm rất cao, `U=0.9315`, có 16 box ambiguous; model có mức bất định lớn. |
| 3 | `frame_0380.jpg` | 0.9170 | 152.0 | 3 | `U=0.9340` cao và có 15 box ambiguous; vẫn có giá trị rà nhãn dù thời điểm khá gần `frame_0369`. |
| 4 | `frame_0312.jpg` | 0.9100 | 124.8 | 7 | `A=1.0`, `D=1.0`, có 18 box ambiguous; số trường hợp cần kiểm tra nhiều. |
| 5 | `frame_0392.jpg` | 0.8874 | 156.8 | 15 | Điểm tổng thấp hơn nhưng `U=0.9747` là rất cao, cao nhất trong các frame được chọn; vì vậy không nên bỏ qua chỉ vì rank thấp. |

### Xét ảnh gần trùng

`frame_0369.jpg` ở 147.6 giây và `frame_0380.jpg` ở 152.0 giây cách nhau **4.4 giây**, vì vậy có khả năng nội dung cảnh gần nhau hơn so với các frame cách xa về thời gian. Tuy nhiên cả hai đều có độ bất định cao (`U=0.9315` và `U=0.9340`), nên vẫn có lý do để rà cả hai nếu mục tiêu là tìm các trường hợp model chưa chắc chắn.

Đồng thời, chiến lược sử dụng `MIN_GAP_S = 2.0` để hạn chế việc chọn những frame quá gần nhau. Vì khoảng cách giữa hai frame này vẫn lớn hơn 2 giây nên chúng không bị loại chỉ vì điều kiện khoảng cách tối thiểu.

## 2. Ba frame thuộc lô 12 ảnh model chọn

Ba frame tiêu biểu là:

### `frame_0182.jpg`

- Rank: **1**
- Score: **0.9591**
- Thời điểm: **72.8 s**
- `U = 0.9182`
- `A = 1.0`
- `D = 1.0`
- Có **28 box**
- Có **18 box ambiguous**

Đây là frame có điểm cao nhất trong lô 12 ảnh, đồng thời có nhiều box ambiguous. Vì vậy đây là một ứng viên rõ ràng để người gán nhãn kiểm tra.

### `frame_0369.jpg`

- Rank: **2**
- Score: **0.9324**
- Thời điểm: **147.6 s**
- `U = 0.9315`
- `A = 0.8889`
- `D = 1.0`
- Có **43 box**
- Có **16 box ambiguous**

Frame này có nhiều box và độ bất định cao. Việc rà nhãn có thể giúp phát hiện các trường hợp model chưa chắc chắn.

### `frame_0392.jpg`

- Rank: **15**
- Score: **0.8874**
- Thời điểm: **156.8 s**
- `U = 0.9747`
- `A = 0.6667`
- `D = 1.0`
- Có **35 box**
- Có **12 box ambiguous**

Đây là một ví dụ đáng chú ý vì rank thấp hơn nhưng độ bất định `U` lại cao nhất trong các frame được chọn. Điều này cho thấy không nên chỉ nhìn vào một thành phần của điểm để quyết định có rà nhãn hay không.

## 3. Một frame có điểm thấp nhưng vẫn nên xem

Tôi chọn **`frame_0392.jpg`**.

Mặc dù score chỉ **0.8874**, thấp hơn nhiều frame khác trong lô, `U=0.9747` lại rất cao. Điều này cho thấy model đang rất không chắc chắn đối với frame này.

Vì vậy, nếu chỉ nhìn score tổng thì có thể bỏ qua frame 0392, nhưng khi xem từng thành phần của score thì đây vẫn là một ứng viên đáng kiểm tra.

Ngược lại, dữ liệu hiện có trong `batch.json` chỉ chứa 12 frame đã được chọn, không chứa đầy đủ 50 dòng đầu của `outputs/selection_round1.csv`. Vì vậy chưa thể xác định chính xác một frame **điểm cao nhưng không được chọn** trong top 50 mà không tự suy đoán thêm dữ liệu.

## 4. Điều phép chọn này chưa chứng minh về chất lượng mô hình

Điểm selection chỉ cho biết frame nào được ưu tiên theo tiêu chí của chiến lược active learning. Nó **không chứng minh rằng những frame đó chắc chắn sẽ làm mô hình tốt hơn sau khi train**.

Một frame có `U` cao nghĩa là model không chắc chắn về dự đoán của mình, nhưng nguyên nhân có thể là:

- ảnh khó nhìn;
- xe bị che khuất;
- xe rất nhỏ;
- nhiều xe nằm sát nhau;
- pre-label có nhiều box sai hoặc thiếu;
- hoặc frame có nội dung khác với dữ liệu model đã thấy.

Chỉ sau khi người rà nhãn sửa các trường hợp cần thiết, train lại model và đánh giá trên **cùng tập test**, mới có thể kiểm tra xem việc chọn những frame này có thực sự làm AP50, recall hoặc các chỉ số khác thay đổi hay không.

Vì vậy, **selection score là tiêu chí để phân bổ ngân sách gán nhãn, không phải thước đo trực tiếp về chất lượng cuối cùng của mô hình**.
