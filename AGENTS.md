# Investment Notebook — Agent Notes

## Working Style

- Fail fast. Thiếu config, thiếu file, schema sai, hoặc API lỗi thì raise rõ ràng.
- Validate mọi CSV đọc/ghi bằng required columns; không silently accept dữ liệu méo.
- Giữ logic deterministic và dễ audit. Không nhét thuật toán speculative khó giải thích.

## Scope

- Đây là repo notebook tương tác cho data prep, screening, research, và ML artifacts.
- Repo này không còn order workflow. Không reintroduce `orders.csv`, TCBS browser automation, codex batch runner, hay execution archive.
- Mọi output generated nằm dưới `out/` hoặc `research/`.

## Portfolio

- Danh mục là optional. Không được giả định `data/portfolios/portfolio.csv` luôn tồn tại.
- Nếu portfolio vắng mặt, engine vẫn phải chạy và giữ output hợp lệ cho screening thuần.
- Không thêm lại workflow scrape/import danh mục từ browser.

## Investment Profile

- Mặc định phân tích theo kiểu `single active idea`: chỉ tìm tối đa `1` mã để mua ở mỗi vòng.
- Khẩu vị vốn mặc định là khoảng `5 tỷ VND`.
- Ưu tiên size lớn và giải ngân nhanh gọn ở một mã duy nhất khi vùng mua đủ tốt; có thể chia tranche nhưng không chia nhỏ kiểu thăm dò.
- Sau khi vào lệnh, giả định workflow là chờ tới vùng giá/điều kiện exit phù hợp rồi mới xoay vòng sang mã khác.
- Nếu hôm đó không có mã đủ sạch, phải kết luận thẳng là `không mua`.
- Không tự biến workflow này thành order-generation; repo chỉ dùng để phân tích candidate, vùng mua, size gợi ý, thesis và trade-off.

## Expected Workflow

1. Xác nhận scope universe trong `data/industry_map.csv`.
2. Nếu scope sai, dùng tool refresh phù hợp trước khi phân tích.
3. Chạy `./broker.sh prepare` hoặc các builder cần thiết.
4. Đọc `out/` và `research/` trực tiếp trong session Codex.

## Tooling

- Sau khi sửa Python code, chạy `./broker.sh tests`.
- Giữ dependencies tối giản; không thêm package nặng nếu không thật cần cho data/research.
- Test không nên phụ thuộc vào browser thật hay network thật nếu có thể mock.

## Documentation

- Nếu đổi behavior hoặc contract của repo notebook, cập nhật `README.md` và `SYSTEM_DESIGN.md`.
- Giữ docs bám đúng notebook workflow hiện tại; xoá tham chiếu cũ nếu không còn dùng.
