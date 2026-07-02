# Start Here

Nếu đây là session Codex mới, flow ngắn nhất là:

1. mở Codex trong repo này
2. hỏi trực tiếp kiểu `Dự báo VIC hiện tại thế nào?`

Codex phải tự hiểu rằng nếu artifact thiếu hoặc stale thì phải tự chạy tuần tự và tự đợi xong trước khi trả lời. Không được nhả ra một câu trả lời kiểu `đợi batch chạy xong rồi tính tiếp`.
Nếu có tra cứu web/news/source mạng để tổng hợp bối cảnh, phải tách riêng khỏi phần model predict.

Nếu muốn precompute active forecast task trước khi hỏi:

1. `./broker.sh select_vic_model`
2. đọc `out/analysis/vic_single_model_current.csv`, `out/analysis/vic_single_model_holdout.csv`, `out/analysis/vic_single_model_walkback.csv`, và `out/analysis/vic_single_model_candidates.csv`

Active model hiện tại là `baseline_ohlc / logit_direction`: dự đoán chiều T+1..T+5, không dự đoán giá. Các model OHLC/intraday/index-expiry cũ chỉ là diagnostics khi gọi tay.

Prompt gợi ý:

- `Nếu artifact chưa có hoặc stale thì tự chạy và đợi xong rồi mới phân tích.`
- `Nếu rebuild/fetch/full ML cần cho forecast sạch đang chạy lâu, không được tự ý dừng. Nếu command lỗi, bị interrupt, hoặc đã dừng, phải nói chưa có artifact đủ sạch và không kết luận từ phần chạy dở.`
- `Dùng snapshot mới nhất, chỉ dự báo VIC.`
- `Chỉ đưa forecast/validation, không đưa khuyến nghị mua bán, không tính size, không dùng ngân sách hay danh mục.`
- `Tách phần Model predict và External synthesis. Model predict chỉ lấy từ artifact/model trong repo; External synthesis nếu có tra cứu web/news thì ghi rõ nguồn và không trộn vào số forecast.`
- `Trả lời bằng active single model: direction T+1..T+5, xác suất up, holdout hit, walkback hit.`
- `Feature engineering/candidate results nằm trong vic_single_model_candidates và vic_single_model_summary; giữ lại để audit, nhưng không dùng candidate thua làm forecast chính.`
- `Macro/global market từ tra cứu ngoài chỉ là External synthesis. Chỉ coi là tín hiệu predict nếu đã thành factor cache/feature trong repo và được kiểm chứng bằng eval_macro/eval_macro_lift.`
- `Với VIC, pair-feature ML nếu còn trong schema chỉ là context kỹ thuật; không tự kéo pair ticker vào phân tích mặc định. Breakout là feature cho model, không phải tiêu chí hard-code.`
- `Phải nói rõ T+N là N phiên giao dịch sau snapshot; nếu cần khung xa hơn thì đọc OHLC T+15/T+20 thay vì chỉ nhìn T+1/T+3.`
- `Không dùng nhãn cảm tính như nóng/khỏe/yếu/mạnh. Nếu nhắc RSI/SMA/ret5d thì phải là feature hoặc validation metric, không phải rule hành động.`
- `Nếu legacy diagnostics mâu thuẫn với active model thì gọi rõ diagnostics conflict, không thay active forecast.`
- `Nếu cần chạy batch thì tự chạy tuần tự rồi mới trả lời.`

Scope hiện tại:

- repo đang ở chế độ VIC-only, prediction-only
- `./broker.sh refresh_vic_map` hoặc `./broker.sh map` để đảm bảo `data/industry_map.csv` chỉ có `VIC`
- `config/data_engine.yaml` tắt portfolio mặc định bằng `portfolio.enabled: false`
- muốn đổi scope hoặc bật phân tích danh mục legacy thì phải cập nhật contract/config trước rồi chạy lại pipeline liên quan
