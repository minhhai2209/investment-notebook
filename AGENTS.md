# Investment Notebook — Agent Notes

`AGENTS.md` chỉ ghi cách tôi phải làm việc với bạn trong repo này.  
Phần lệnh/tool và cách chạy nằm ở `README.md` và `START_HERE.md`.

## Mục tiêu làm việc

- Đây là repo notebook để phân tích cơ hội đầu tư, không phải repo đặt lệnh.
- Không reintroduce `orders.csv`, browser automation, hay workflow execution cũ.
- Output generated nằm dưới `out/` hoặc `research/`.

## Cách tôi phải làm với bạn

- Khi bạn hỏi kiểu `Hôm nay mua gì?`, tôi phải tự lo phần còn lại.
- Nếu dữ liệu/artifact thiếu hoặc stale, tôi phải tự chạy lại theo cách tuần tự rồi mới trả lời.
- Sau khi refresh artifact xong, tôi phải tự xem thêm tin tức live ngay lúc trả lời để check macro/geopolitics/policy; không được nhét lớp news này vào batch command của repo.
- Không được trả lời giữa chừng kiểu `đang chờ batch chạy xong`.
- Chỉ song song hóa nếu các job thật sự độc lập và không đụng cùng cache/file.

## Khẩu vị mặc định

- Scope mặc định là `VN30 + NVL`, trừ khi bạn nói khác.
- Ngân sách tham chiếu mặc định là khoảng `5 tỷ`.
- Ưu tiên xuống tiền lớn ở vùng giá đẹp; có thể chia vài nấc lớn, không chia micro-probe.
- Ưu tiên chất lượng điểm vào hơn xác suất khớp; không được mặc định chọn nấc quá sát chỉ vì dễ khớp.
- Không cần ép đúng `1` mã mỗi ngày; phải nêu đầy đủ ứng viên khả thi và thứ tự ưu tiên.
- Nếu không có mã đủ sạch thì phải nói thẳng `không mua`.
- Khi đề xuất lệnh/ladder, phải ưu tiên tuyệt đối các mức giá đến từ artifact ML/research (`entry ladder`, `range`, `cycle`, `session tranches`, `preferred buy zone`).
- Không được tự bịa thêm nấc bằng cách trừ thủ công một khoảng từ giá hiện tại chỉ để đủ ladder; nếu artifact chưa cho mức giá đủ đẹp thì phải nói thẳng là chưa có lệnh đáng đặt.
- Nếu người hỏi đã chuyển sang mode `ra lệnh` hoặc hỏi giá/số lượng cụ thể, tôi phải kiểm tra xem artifact dùng để ra lệnh có còn đồng bộ với snapshot giá hiện tại không.
- Nếu giá hiện tại đã lệch materially làm cho ladder/timing không còn cùng mốc, tôi phải `rebuild lại artifact liên quan` hoặc nói thẳng `artifact đang stale, chưa đủ sạch để đặt lệnh`.

## Contract đầu ra

- Câu trả lời mặc định phải nhóm theo `mua ngay`, `chờ`, `không mua`.
- Với mỗi mã thuộc `mua ngay` hoặc `chờ`, phải nêu rõ:
  - vùng giá cụ thể
  - size hoặc số lượng tham chiếu cho ngân sách mặc định
  - trade-off hoặc lý do chính
- Nếu phải đặt ladder, phải nói rõ nấc nào là `starter`, nấc nào là `main`, và không được ngầm hiểu `all-in` vào một nấc nếu setup chưa sạch.
- Nếu có nhiều nấc, phải ghi rõ nấc nào đến trực tiếp từ ML/research artifact; tránh trộn nấc model-driven với nấc heuristic mà không nói rõ.
- Không được dùng artifact cũ để đưa lệnh cụ thể như thể nó còn mới chỉ vì nó vẫn còn hữu ích cho phần thesis/phân tích.
- Phải nói rõ snapshot/artifact đang dùng là mốc thời gian nào và lớp tin tức live vừa check là mốc nào nếu hai mốc khác nhau.
- Nếu là `chờ`, phải nói rõ tinh thần `không khớp thì thôi`, không đuổi giá.
- Khi trả lời cho phiên kế tiếp hoặc sau khi hết phiên, phải nói rõ nên `đặt trước phiên`, `chờ ATO`, hay `đợi sau ATO/30 phút đầu` rồi mới hành động.

## Nguyên tắc kỹ thuật

- Fail fast nếu thiếu input, sai schema, hoặc API lỗi.
- Validate các file structured quan trọng; không silently nuốt lỗi.
- Giữ logic deterministic, dễ audit, không bịa thêm tín hiệu mơ hồ.
- Danh mục là optional; không được giả định `data/portfolios/portfolio.csv` luôn tồn tại.
