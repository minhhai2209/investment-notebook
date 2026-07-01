# Investment Notebook — Agent Notes

`AGENTS.md` chỉ ghi cách tôi phải làm việc với bạn trong repo này.  
Phần lệnh/tool và cách chạy nằm ở `README.md` và `START_HERE.md`.

## Mục tiêu làm việc

- Đây là repo notebook để dự báo thị trường/mã theo dữ liệu, không phải repo khuyến nghị giao dịch.
- Scope phân tích mặc định hiện là `VIC`; nếu bạn muốn đổi scope, phải cập nhật contract/config trước rồi chạy lại pipeline liên quan.
- Không reintroduce `orders.csv`, browser automation, order placement, portfolio workflow, budget sizing, hoặc execution workflow cũ.
- Output generated nằm dưới `out/`, `research/`, hoặc `reports/`.

## Cách tôi phải làm với bạn

- Khi bạn hỏi `hiện tại`, `phiên tới`, `dự báo`, hoặc câu tương tự, tôi phải tự kiểm tra artifact có stale không; nếu stale thì refresh artifact liên quan trước khi phân tích.
- Khi rebuild/fetch/full ML đang cần để có forecast sạch, tôi không được tự ý dừng chỉ vì chạy lâu. Nếu command lỗi, bị user interrupt, hoặc tôi đã dừng vì bất kỳ lý do gì, câu trả lời phải nói rõ `đã dừng/chưa có artifact đủ sạch` và không được kết luận từ phần chạy dở.
- Không trả lời giữa chừng kiểu `đang chờ batch chạy xong` nếu forecast chính vẫn đang chạy.
- Chỉ song song hóa nếu các job thật sự độc lập và không đụng cùng cache/file.
- Nếu bạn chỉ ra một lỗi lặp lại, một chỗ tôi xử lý chưa đúng, hoặc một kỳ vọng mới về cách làm việc, tôi phải cập nhật `way of working` của repo ngay khi hợp lý.
- Khi một bài học đủ ổn định để áp dụng cho các lượt sau, tôi phải cập nhật nó vào contract/docs thay vì giữ như trí nhớ tạm trong session.

## Prediction-Only Contract

- Repo này không còn dùng ngân sách mặc định, không tính số lượng cổ phiếu, không nói size, không dựng ladder, không đọc/gộp danh mục mặc định.
- Tôi không được trả lời theo nhóm `mua ngay / chờ / không mua` trừ khi bạn yêu cầu rõ một lớp khuyến nghị ngoài phạm vi repo.
- Tôi không được biến forecast thành lệnh mua/bán, không nói `nên đặt trước phiên`, `chờ ATO`, `đợi sau ATO`, hoặc vùng resting buy/sell.
- Nếu bạn hỏi bằng ngôn ngữ mua/bán, tôi vẫn trả lời ở dạng prediction-only: xác suất/đường giá dự báo, vùng high/low/close, sai số, độ hit, và mức tin cậy. Nếu cần, tôi sẽ nói rõ repo này không còn đưa khuyến nghị giao dịch.
- Nếu có file `data/portfolios/portfolio.csv`, cấu hình mặc định vẫn không đọc danh mục. Danh mục chỉ được dùng khi bạn bật rõ `portfolio.enabled: true` cho một phân tích legacy.
- Tôi có thể tra cứu web/news/source mạng để tổng hợp bối cảnh, nhưng phải tách riêng khỏi phần `Model predict`.
- `Model predict` chỉ được lấy từ model/artifact nội bộ trong repo. Không được dùng bài báo, trang tài chính, mạng xã hội, hoặc nhận định bên ngoài để sửa số forecast, chọn model, đổi confidence, hay gọi đó là tín hiệu model nếu chưa được mã hoá thành feature và kiểm chứng bằng walk-forward/feature-lift.
- `External synthesis` là phần riêng nếu có tra cứu: phải ghi rõ nguồn/thời điểm, tóm tắt như bối cảnh ngoài model, và nói rõ nó không phải output của model.
- Network/API qua fetcher/pipeline của repo vẫn được dùng để lấy dữ liệu đầu vào có thể tái lập cho model, ví dụ OHLCV, intraday, factor cache.

## Active Model Surface

- Active forecast task mặc định:
  - `./broker.sh ohlc`: dự báo OHLC T+n.
  - `./broker.sh intraday`: dự báo close còn lại của phiên từ 1-minute OHLCV/depth nếu đang trong phiên hỗ trợ.
  - `./broker.sh eval_vic_index_expiry --models hist_gbm`: kiểm tra thêm feature VNINDEX/ex-Vin/derivative-expiry khi cần audit forecast daily.
  - `./broker.sh eval_curated_intraday_model`: pooled intraday audit/model khi cần kiểm chứng feature theo walk-back.
- Mỗi forecast phải đi cùng validation: model name/family/class, MAE, direction hit hoặc metric tương đương, sample/backtest window nếu artifact có.
- `T+N` luôn nghĩa là `N phiên giao dịch sau snapshot`, không phải ngày dương lịch.
- Các mốc kỹ thuật như breakout, high 20/60/120/252, RSI, SMA distance, return 5/20d chỉ được nhắc như feature/input hoặc diagnostics; không dùng làm rule hành động.
- `eval_deterministic` là harness replay/feature legacy, không dùng làm nguồn forecast chính hoặc nguồn quyết định.

## Contract Đầu Ra

- Câu trả lời mặc định phải có:
  - snapshot/artifact timestamp đang dùng
  - forecast OHLC T+1 và multi-session tới `T+20` nếu có
  - intraday forecast nếu đang trong phiên và artifact hợp lệ
  - validation/error band đủ để hiểu forecast đáng tin đến đâu
  - cảnh báo stale/missing artifact nếu có
- Không đưa recommendation, position sizing, order ladder, hoặc opportunity-cost vốn.
- Nếu forecast daily và intraday mâu thuẫn materially, gọi rõ là `model conflict` và mô tả mâu thuẫn bằng số liệu.
- Nếu có tra cứu ngoài repo, câu trả lời phải có phần `External synthesis` tách khỏi `Model predict`; không trộn bối cảnh web/news vào forecast/validation.
- Nếu muốn biến macro/event/factor thành tín hiệu predict thì phải mã hoá thành feature trong repo và kiểm chứng bằng walk-forward/feature-lift trước khi coi là model evidence.
- `config/market_events.json` là input dữ liệu nội bộ nếu model/pipeline dùng; tra cứu lịch/tin ngoài repo chỉ thuộc `External synthesis` cho bối cảnh, không tự động thành feature.

## Nguyên Tắc Kỹ Thuật

- Fail fast nếu thiếu input, sai schema, hoặc API lỗi.
- Validate các file structured quan trọng; không silently nuốt lỗi.
- Giữ logic deterministic, dễ audit, không bịa thêm tín hiệu mơ hồ.
- Cấu hình mặc định tắt danh mục bằng `portfolio.enabled: false`; không được giả định `data/portfolios/portfolio.csv` tồn tại hay có ý nghĩa trong forecast mặc định.
