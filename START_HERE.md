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

Nếu muốn soi sâu một mã trước khi hỏi:

1. `./broker.sh deep VIC`

Nếu muốn xem ranking ứng viên thống nhất mà không hỏi Codex ngay:

1. `./broker.sh candidates auto`

Prompt gợi ý:

- `Nếu artifact chưa có hoặc stale thì tự chạy và đợi xong rồi mới phân tích.`
- `Sau khi artifact xong thì tự check tin tức live 12-24h gần nhất rồi mới chốt mua ngay / chờ / không mua.`
- `Nếu tôi chỉ ra một lỗi lặp lại hoặc một rule mới về cách làm việc, hãy cập nhật luôn contract/docs của repo.`
- `Dùng snapshot mới nhất, lọc VN30 + NVL và liệt kê đầy đủ ứng viên hôm nay.`
- `Nếu chưa có mã nào sạch thì nói thẳng không mua.`
- `Đừng sinh orders; chỉ phân tích candidate, vùng giá, thesis và trade-off.`
- `Giả định ngân sách khoảng 5 tỷ; với mỗi ứng viên mua ngay hoặc chờ thì phải nói rõ vùng giá và size tham chiếu.`
- `Format mặc định: mua ngay / chờ / không mua. Nếu là chờ thì nêu vùng resting buy, không khớp thì thôi.`
- `Đừng ưu tiên nấc sát chỉ vì dễ khớp; phải nói rõ nấc nào là thăm dò, nấc nào là nấc chính.`
- `Ladder phải đi ra từ ML/research artifact; nếu chưa có nấc model-driven phù hợp thì phải nói chưa nên đặt, không tự bẻ thêm nấc từ giá hiện tại.`
- `Nếu đã hỏi giá/số lượng đặt lệnh cụ thể mà artifact không còn cùng mốc với giá hiện tại thì phải rebuild trước, hoặc nói thẳng artifact stale chưa đủ sạch để đặt.`
- `Nếu hỏi cho phiên kế tiếp thì phải chốt luôn nên đặt trước phiên, chờ ATO, hay đợi sau ATO rồi mới làm.`
- `Nếu cần chạy batch thì tự chạy tuần tự rồi mới trả lời.`

Nếu bạn đổi scope:

- `./broker.sh refresh_hose_map` để mở rộng hơn `VN30`
- hoặc tự thay `data/industry_map.csv` rồi chạy lại `./broker.sh prepare`
