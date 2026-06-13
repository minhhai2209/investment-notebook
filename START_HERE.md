# Start Here

Nếu đây là session Codex mới, flow ngắn nhất là:

1. mở Codex trong repo này
2. hỏi trực tiếp kiểu `Hôm nay mua gì?`

Codex phải tự hiểu rằng nếu artifact thiếu hoặc stale thì phải tự chạy tuần tự và tự đợi xong trước khi trả lời. Không được nhả ra một câu trả lời kiểu `đợi batch chạy xong rồi tính tiếp`.
Sau khi refresh artifact xong, Codex còn phải tự check thêm tin tức live gần nhất trước khi chốt câu trả lời `Hôm nay mua gì?`; bước news này làm ngay trong lúc trả lời, không đưa vào `broker.sh`.

Nếu bạn muốn precompute thủ công trước:

1. `./broker.sh prepare_default`

Hoặc chạy foreground một phát:

1. `./broker.sh map`
2. `./broker.sh prepare`

Nếu muốn precompute hai active forecast tasks trước khi hỏi:

1. `./broker.sh ohlc`
2. `./broker.sh intraday`

Nếu đang trong phiên và muốn xem case đang đỏ có khả năng hồi cuối phiên hay không:

1. `./broker.sh intraday`
2. đọc `out/analysis/ml_intraday_rest_of_session.csv`, nhất là `PredCloseUpFromSnapshotProbPct`, `PredRecoverToPrevCloseProbPct`, calibration rows, `RecoverySetup`, và `RecoveryCalibrationConflict`

Nếu muốn review một vị thế đã mua bằng ML-only, không tự chế rule xử lý:

1. `./broker.sh position_ml --ticker VIC --quantity 20000 --avg-price 214.53 --current-price <live_price>`

Prompt gợi ý:

- `Nếu artifact chưa có hoặc stale thì tự chạy và đợi xong rồi mới phân tích.`
- `Nếu rebuild/fetch/full ML cần cho khuyến nghị hoặc lệnh cụ thể đang chạy lâu, không được tự ý dừng. Nếu command lỗi, bị interrupt, hoặc đã dừng, phải nói chưa có artifact đủ sạch và không đưa khuyến nghị/lệnh dựa trên phần chạy dở.`
- `Sau khi artifact xong thì tự check tin tức live 12-24h gần nhất rồi mới chốt mua ngay / chờ / không mua.`
- `Nếu tôi chỉ ra một lỗi lặp lại hoặc một rule mới về cách làm việc, hãy cập nhật luôn contract/docs của repo.`
- `Dùng snapshot mới nhất, chỉ phân tích VIC hôm nay.`
- `Nếu chưa có mã nào sạch thì nói thẳng không mua.`
- `Đừng sinh orders; chỉ phân tích candidate, vùng giá, thesis và trade-off.`
- `Giả định ngân sách khoảng 5 tỷ; với mỗi ứng viên mua ngay hoặc chờ thì phải nói rõ vùng giá và size tham chiếu.`
- `Format mặc định: mua ngay / chờ / không mua. Nếu là chờ thì nêu vùng resting buy, không khớp thì thôi.`
- `Đừng ưu tiên nấc sát chỉ vì dễ khớp; phải nói rõ nấc nào là thăm dò, nấc nào là nấc chính.`
- `Ladder phải đi ra từ ML/research artifact; nếu chưa có nấc model-driven phù hợp thì phải nói chưa nên đặt, không tự bẻ thêm nấc từ giá hiện tại.`
- `Nếu đã hỏi giá/số lượng đặt lệnh cụ thể mà artifact không còn cùng mốc với giá hiện tại thì phải rebuild trước, hoặc nói thẳng artifact stale chưa đủ sạch để đặt.`
- `Nếu tôi đưa giá live mới, hỏi trong giờ nghỉ trưa, ngoài phiên, sau ATO, gần ATC, hoặc snapshot đã cũ so với thị trường hiện tại, phải fetch/rebuild lại snapshot giá và artifact liên quan trước khi nói lệnh cụ thể; nếu full ML chạy chậm thì tiếp tục chạy cho xong trừ khi tôi yêu cầu dừng, không tự ý chuyển sang khuyến nghị tạm cho lệnh/size lớn.`
- `Nếu hỏi hiện tại thì sao, phải trả lời thêm đà các phiên tới là tăng tiếp / đi ngang / giảm, độ gấp mua ngay hay có thể chờ, và overlay lịch nghỉ lễ/sự kiện/tin tức live.`
- `Khi đưa khuyến nghị, phải đọc hai active model: OHLC T+n tới T+20 và intraday morning-close nếu đang trong phiên; nêu model name/family/class, MAE và direction hit; không chỉ nói mua được hay không.`
- `Khi có tin dầu/Iran/global market, chạy hoặc đọc eval_macro ở chế độ cache sau khi refresh factor cho VIC để xem correlation với dầu, USD, VIX, lợi suất Mỹ và các chỉ số Mỹ/Âu/Anh/Nhật/Hàn; sau đó chạy hoặc đọc eval_macro_lift để kiểm tra các feature này có cải thiện predict so với baseline không.`
- `Với VIC, pair-feature ML nếu còn trong schema chỉ là context kỹ thuật; không tự kéo pair ticker vào phân tích mặc định. Breakout là feature cho model, không phải tiêu chí hard-code.`
- `Phải nói rõ T+N là N phiên giao dịch sau snapshot; nếu cần khung xa hơn thì đọc OHLC T+15/T+20 thay vì chỉ nhìn T+1/T+3.`
- `Nếu hỏi xử lý vị thế đã mua, phải dùng ML-only position review; không tự chế số lượng bán/cắt/mua thêm nếu artifact không có action-sizing.`
- `Không dùng nhãn cảm tính như nóng/khỏe/yếu/mạnh làm lý do; nếu nhắc RSI/SMA/ret5d thì phải là feature hoặc validation metric, không phải rule hành động.`
- `Nếu hỏi có nên mua một mã cụ thể ngay không, repo đang ở chế độ VIC-only; không kéo thêm mã đối chứng.`
- `Decision cuối cùng phải dựa vào forecast/validation riêng từng mã, không dùng score cộng/trừ thủ công hay overlay hard-code.`
- `Nếu hỏi cho phiên kế tiếp thì phải chốt luôn nên đặt trước phiên, chờ ATO, hay đợi sau ATO rồi mới làm.`
- `Nếu cần chạy batch thì tự chạy tuần tự rồi mới trả lời.`

Scope hiện tại:

- repo đang ở chế độ VIC-only
- `./broker.sh refresh_vic_map` hoặc `./broker.sh map` để đảm bảo `data/industry_map.csv` chỉ có `VIC`
- muốn đổi scope thì phải cập nhật contract trước rồi chạy lại pipeline liên quan
