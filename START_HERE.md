# Start Here

Nếu đây là session Codex mới, flow ngắn nhất là:

1. `./broker.sh refresh_vn30_map`
2. `./broker.sh prepare`
3. hỏi trực tiếp trong repo này

Prompt gợi ý:

- `Dùng snapshot mới nhất, lọc VN30 và chọn đúng 1 mã đáng theo dõi nhất hôm nay.`
- `Nếu chưa có mã nào sạch thì nói thẳng không mua.`
- `Đừng sinh orders; chỉ phân tích candidate, vùng giá, thesis và trade-off.`
- `Giả định ngân sách khoảng 5 tỷ, muốn mua size lớn nhưng chỉ cho một mã duy nhất nếu vùng mua đủ tốt.`

Nếu bạn đổi scope:

- `./broker.sh refresh_hose_map` để mở rộng hơn `VN30`
- hoặc tự thay `data/industry_map.csv` rồi chạy lại `./broker.sh prepare`
