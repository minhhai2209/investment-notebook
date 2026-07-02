# AGENTS.md

## Repo Contract

- Repo này là numeric data hub cho chứng khoán Việt Nam.
- Không xây model dự báo trong repo.
- Không xuất forecast, confidence, target price, khuyến nghị mua/bán, position sizing, ngân sách, danh mục, ladder, hoặc report giao dịch.
- Không thu thập tin tức. Chỉ dữ liệu số và metadata số/nhóm ngành cần thiết để tính toán.
- ChatGPT hoặc người dùng sẽ tự phân tích ở ngoài repo dựa trên artifact số.

## Default Workflow

1. Dùng `./broker.sh hub` để dựng `data-hub/latest/` từ cache hiện có.
2. Dùng `./broker.sh collect` khi cần refresh các API số đã bật trong `config/data_hub.yaml`.
3. Đọc `data-hub/latest/START_HERE.json` trước, rồi tới `bundles/source_audit.csv`, `bundles/market_snapshot.csv`, `bundles/symbol_latest.csv`, `index/ticker_catalog.csv`, `index/file_catalog.csv`, và chỉ mở file per-ticker trong `daily/`, `intraday/minute_profile/` khi cần drill-down.
4. Nếu thiếu dữ liệu nguồn, báo thiếu cache/source rõ ràng. Không suy diễn hay bịa số.

## Allowed Numeric Sources

- VNDIRECT dchart OHLCV.
- VNDIRECT priceboard/order-book snapshot.
- CafeF foreign/proprietary flows.
- Vietstock overview and financial statement numeric tables.
- FRED/Stooq macro numeric cache.
- Market membership/universe metadata.

## Code Rules

- Code mới phải phục vụ thu thập, chuẩn hóa, tính toán, hoặc đóng gói dữ liệu số.
- Output cho ChatGPT phải ưu tiên CSV/JSON nhỏ, dễ browser, có manifest mô tả rõ file nào cần đọc.
- API catalog phải ghi rõ source, endpoint, loại dữ liệu số, output mặc định, và xác nhận không phải news.
- Test nên kiểm tra contract file/output thay vì prediction quality.
- Calculations nên ưu tiên các lớp có ý nghĩa rộng: trend, volatility, drawdown, liquidity, relative strength, breadth, cross-section rank, intraday volume/price per minute.

## Commands

```bash
./broker.sh hub
./broker.sh collect
./broker.sh tests
./broker.sh validate_layout --ticker VIC
./broker.sh refresh_macro
./broker.sh refresh_bctt
./broker.sh refresh_vic_map
```
