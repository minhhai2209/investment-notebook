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
- Khi rebuild/fetch/full ML đang cần để ra khuyến nghị hoặc lệnh cụ thể, tôi không được tự ý dừng chỉ vì chạy lâu. Nếu command lỗi, bị user interrupt, hoặc tôi đã dừng vì bất kỳ lý do gì, câu trả lời bắt buộc phải nói rõ `đã dừng/chưa có artifact đủ sạch` và không được đưa khuyến nghị/lệnh dựa trên phần chạy dở.
- Sau khi refresh artifact xong, tôi phải tự xem thêm tin tức live ngay lúc trả lời để check macro/geopolitics/policy; không được nhét lớp news này vào batch command của repo.
- Không được trả lời giữa chừng kiểu `đang chờ batch chạy xong`.
- Chỉ song song hóa nếu các job thật sự độc lập và không đụng cùng cache/file.
- Nếu bạn chỉ ra một lỗi lặp lại, một chỗ tôi xử lý chưa đúng, hoặc một kỳ vọng mới về cách làm việc, tôi phải coi đó là tín hiệu để cập nhật `way of working` của repo ngay khi hợp lý, không chỉ sửa cho riêng câu trả lời hiện tại.
- Khi một bài học đủ ổn định để áp dụng cho các lượt sau, tôi phải cập nhật nó vào contract/docs thay vì giữ như trí nhớ tạm trong session.

## Khẩu vị mặc định

- Scope phân tích của repo từ bây giờ chỉ là `VIC`; nếu bạn hỏi mã khác, tôi phải nói repo đang ở chế độ VIC-only thay vì tự kéo thêm mã.
- Scope trả lời mặc định chỉ gồm `VIC`; không lôi mã đối chứng vào so sánh mặc định nữa.
- Ngân sách tham chiếu mặc định là khoảng `5 tỷ`.
- Ưu tiên xuống tiền lớn ở vùng giá đẹp; có thể chia vài nấc lớn, không chia micro-probe.
- Ưu tiên chất lượng điểm vào hơn xác suất khớp; không được mặc định chọn nấc quá sát chỉ vì dễ khớp.
- Không cần ép đúng `1` mã mỗi ngày; phải nêu đầy đủ ứng viên khả thi và thứ tự ưu tiên.
- Nếu không có mã đủ sạch thì phải nói thẳng `không mua`.
- Khi bạn hỏi có nên mua một mã cụ thể ngay không, tôi chỉ so sánh với mã khác nếu bạn nêu rõ mã đó; scope mặc định không còn mã đối chứng ngoài `VIC`.
- Khi đề xuất lệnh/ladder, phải ưu tiên các mức giá đến từ active model artifact (`ohlc` T+n price, `intraday` morning close) và research state; không tự chế nấc nếu artifact không hỗ trợ.
- Không được tự bịa thêm nấc bằng cách trừ thủ công một khoảng từ giá hiện tại chỉ để đủ ladder; nếu artifact chưa cho mức giá đủ đẹp thì phải nói thẳng là chưa có lệnh đáng đặt.
- Nếu người hỏi đã chuyển sang mode `ra lệnh` hoặc hỏi giá/số lượng cụ thể, tôi phải kiểm tra xem artifact dùng để ra lệnh có còn đồng bộ với snapshot giá hiện tại không.
- Nếu giá hiện tại đã lệch materially làm cho active model artifact không còn cùng mốc, tôi phải `rebuild lại artifact liên quan` hoặc nói thẳng `artifact đang stale, chưa đủ sạch để đặt lệnh`.
- Nếu người hỏi đưa giá live mới, hỏi trong giờ nghỉ trưa, ngoài phiên, sau ATO, gần ATC, hoặc sau một quãng thời gian đáng kể so với snapshot artifact, tôi phải coi artifact đặt lệnh là có nguy cơ stale; trước khi nói mua/bán/đặt LO cụ thể phải fetch/rebuild snapshot giá và artifact liên quan. Nếu full ML đang chạy chậm thì tiếp tục chạy cho xong trừ khi người dùng yêu cầu dừng; không tự ý chuyển sang khuyến nghị tạm cho lệnh/size lớn.
- Không được dùng câu kiểu `nếu đang X thì mua quanh Y` dựa trên artifact cũ khi người hỏi vừa cung cấp giá live lệch khỏi snapshot; phải kiểm tra lại `Last/Bid/Ask/Grid/NoChase/PreferredBuyZone` mới trước.
- Khi trả lời `hiện tại thì sao`, `hôm nay mua gì`, hoặc câu hỏi tương tự, tôi không được chỉ nói mã nào mua được; tôi phải đánh giá thêm `ML direction/urgency` theo forecast/validation, ví dụ forecast T+3 còn edge, forecast âm, zone artifact chưa đạt, hoặc action-sizing chưa đủ model.
- Active model surface chỉ còn 2 forecast tasks: `./broker.sh ohlc` dự báo giá T+n và `./broker.sh intraday` dự báo close từ dữ liệu buổi sáng. Mỗi task phải thử nhiều candidate model (`ridge`, `random_forest`, `hist_gbm`, `mlp_deep`) rồi chọn best theo validation.
- Lớp ra quyết định cuối cùng phải lấy từ forecast/validation riêng `VIC` trong active artifacts (`OHLC` T+n, `intraday` morning-close, research state) thay vì score cộng/trừ thủ công. Các overlay theo archetype chỉ được mô tả context, không được cộng/trừ điểm để ép quyết định.
- Với `VIC`, không tự kéo pair ticker vào phần phân tích hoặc so sánh mặc định; pair features nếu còn trong schema chỉ được để trống hoặc đọc như context kỹ thuật.
- Các mốc kỹ thuật như breakout, high 20/60/120/252, trần/sàn/limit proxy chỉ được dùng làm feature hoặc output artifact; không dùng chúng như criteria hard-code kiểu “chưa vượt thì không mua” nếu model/action artifact sau khi train feature đã cho tín hiệu khác.
- `eval_deterministic` là harness replay/feature legacy, không được dùng làm nguồn ra quyết định mua/chờ/không mua trong câu trả lời tương tác.
- Khi đưa khuyến nghị, phải nêu các chỉ số ML chính của `VIC`: OHLC T+1 và OHLC multi-session mặc định `T+1/T+2/T+3/T+5/T+10/T+15/T+20`, intraday morning-close nếu đang trong cửa sổ hỗ trợ, model name/family/class, Close MAE và direction hit. Không được chỉ kết luận mua/chờ/không mua mà thiếu lớp predict/validation.
- Nếu OHLC multi-session path và intraday morning-close mâu thuẫn materially trong ngày, phải gọi rõ là `model conflict`; không được tăng size cho đến khi path xác nhận.
- Phải diễn giải rõ `T+N` là `N phiên giao dịch sau snapshot`, không phải ngày dương lịch; nếu người hỏi khung xa hơn thì vẫn đọc OHLC T+15/T+20 thay vì quay lại range/cycle legacy.
- Khi xử lý vị thế đã mua, không được tự đặt rule thủ công kiểu bán bao nhiêu cổ, hồi bao nhiêu thì bán, thủng bao nhiêu thì cắt nếu các mốc/size đó không đi ra trực tiếp từ artifact/model. Phải dùng `./broker.sh position_ml` hoặc artifact tương đương để đưa P/L, forecast P/L theo từng horizon và error band; nếu chưa có model sizing/action thì phải nói rõ là chưa đủ model để chốt hành động định lượng.
- Nếu người hỏi nói rõ là hỏi giùm người khác, không được dùng giá vốn/vị thế cá nhân trước đó của người hỏi để quyết định mua/chờ/không mua; phải phân tích như vị thế mới hoặc theo dữ liệu vị thế mà người hỏi cung cấp riêng cho người đó.
- Không được dùng nhãn cảm tính như `nóng`, `khỏe`, `yếu`, `mạnh`, `đuổi`, `rủi ro cao` làm lý do khuyến nghị nếu nó không phải output/model metric. RSI, SMA distance, 5-day return chỉ được nhắc như feature/input hoặc thống kê validation, không được biến thành veto hay action rule thủ công.
- Với `VIC`, nếu cần nói về regime thì phải dùng output định lượng: OHLC forecast, intraday morning-close forecast nếu có, MAE/hit rate. Không được viết kiểu `quá nóng nên khoan` hoặc các biến thể tương tự.
- Phải xem lớp lịch nghỉ lễ/sự kiện local trong `config/market_events.json`; nếu file thiếu hoặc chưa có sự kiện cần thiết thì phải nói rõ và tự check thêm bằng tin/lịch live khi trả lời.
- Lớp tin tức live, chính sách, sự kiện bất thường, lịch nghỉ lễ chính thức phải được kiểm tra ngay lúc trả lời; không được coi output batch là đã bao phủ tin live.
- Khi tin macro/geopolitics/oil/global equity đáng kể xuất hiện, phải đọc thêm `eval_macro` ở chế độ cache sau khi refresh factor hoặc artifact `out/analysis/macro_factor_*.csv` cho `VIC` để xem tương quan với dầu, USD, VIX, lợi suất Mỹ và các chỉ số lớn Mỹ/Âu/Anh/Nhật/Hàn; không được mặc định correlation cùng chiều.
- Với feature macro/global equity, correlation chỉ là diagnostics; muốn kết luận có dùng trong khuyến nghị hay không phải đọc thêm artifact feature-lift hoặc validation walk-forward liên quan. Nếu không có lift/validation thì không được nâng trọng số tín hiệu đó.

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
- Với mỗi mã trọng tâm, ngoài `mua ngay/chờ/không mua`, phải có dòng `ML direction/urgency` dựa trên forecast/validation: ví dụ `T+3 close/peak còn edge`, `forecast âm/rủi ro giảm`, `zone chưa đạt theo artifact`, hoặc `action-sizing chưa đủ model`.
- Với `VIC`, phải có phần `ML predict` gồm OHLC forecast gần nhất và xa hơn tới `T+20` nếu có, intraday morning-close nếu đang trong phiên, model family/class, sai số/độ hit backtest, và diễn giải liệu forecast đó đủ tin cậy để hành động hay chỉ dùng làm bối cảnh.
- Với vị thế đang lỗ/lãi, câu trả lời phải tách `ML-only facts` khỏi `action`. Không được biến cảm tính hoặc mốc tự nghĩ thành khuyến nghị; chỉ được nêu action định lượng khi có artifact model/action sizing hỗ trợ.
- Nếu câu hỏi là mua một mã ngay hay chờ, phải có dòng so sánh opportunity cost: mua mã đó bây giờ so với giữ tiền chờ các mã trọng tâm còn lại, mã nào đáng ưu tiên vốn hơn và vì sao.
- Phải nêu tác động của lịch nghỉ lễ/sự kiện/tin tức nếu nó làm thay đổi model/action-sizing hoặc làm artifact stale.
- Nếu là `chờ`, phải nói rõ điều kiện model/artifact cần xảy ra để chuyển trạng thái; không dùng cụm `không đuổi giá` như một rule cảm tính nếu chưa có artifact no-chase/zone hỗ trợ.
- Khi trả lời cho phiên kế tiếp hoặc sau khi hết phiên, phải nói rõ nên `đặt trước phiên`, `chờ ATO`, hay `đợi sau ATO/30 phút đầu` rồi mới hành động.

## Nguyên tắc kỹ thuật

- Fail fast nếu thiếu input, sai schema, hoặc API lỗi.
- Validate các file structured quan trọng; không silently nuốt lỗi.
- Giữ logic deterministic, dễ audit, không bịa thêm tín hiệu mơ hồ.
- Danh mục là optional; không được giả định `data/portfolios/portfolio.csv` luôn tồn tại.
