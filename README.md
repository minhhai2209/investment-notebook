# Investment Notebook

Repo này chỉ còn một nhiệm vụ: tự động thu thập, chuẩn hóa và tính toán dữ liệu số cho thị trường chứng khoán Việt Nam để ChatGPT đọc nhanh khi cần phân tích.

Không có model dự báo, không có khuyến nghị mua bán, không có danh mục/ngân sách, không thu thập tin tức.

## Cách chạy

```bash
./broker.sh hub
```

Lệnh này dựng `data-hub/latest/` từ cache hiện có. ChatGPT nên bắt đầu đọc theo thứ tự:

1. `data-hub/latest/START_HERE.json`
2. `data-hub/latest/bundles/source_audit.csv`
3. `data-hub/latest/bundles/market_snapshot.csv`
4. `data-hub/latest/bundles/symbol_latest.csv`
5. `data-hub/latest/index/ticker_catalog.csv`
6. `data-hub/latest/index/file_catalog.csv`
7. `data-hub/latest/manifest.json`

Khi muốn gọi API và làm mới cache số:

```bash
./broker.sh collect
```

`collect` refresh các nguồn số nhanh được bật trong `config/data_hub.yaml`: OHLCV daily/intraday, depth, CafeF flows, Vietstock overview và macro cache. Vietstock BCTT có command riêng vì dữ liệu quarterly và collector dùng Playwright.

## Tự Động Cập Nhật

GitHub Action `Refresh Numeric Data` chạy từ thứ Hai đến thứ Sáu lúc 11:45 và 15:15 theo giờ Việt Nam, đồng thời hỗ trợ chạy tay. Workflow chạy test, refresh nguồn, tính lại data hub, kiểm tra layout rồi commit kết quả vào `main`.

Các cache nguồn và refresh-summary cần cho lần chạy incremental được lưu có chọn lọc trong `out/`. Daily giữ 900 ngày; intraday 1 phút và depth snapshot giữ rolling 30 ngày. ChatGPT vẫn đọc từ `data-hub/latest/`; `out/` chỉ là dữ liệu nguồn để collector tiếp tục từ lần trước hoặc để audit sâu.

## Dữ Liệu Có Thể Tính

Nguồn số đang được inventory trong repo:

- VNDIRECT dchart: daily/intraday OHLCV.
- VNDIRECT priceboard snapshot: bid/ask depth, spread, match price, foreign room.
- CafeF flows: mua bán khối ngoại và tự doanh theo cửa sổ 1/5/20 phiên.
- Vietstock overview: PE forward, PB, ROE suy ra từ PE/PB.
- Vietstock BCTT cache: EPS, BVPS, margin, ROE/ROA, nợ, tăng trưởng doanh thu/lợi nhuận.
- FRED/Stooq macro cache: oil, gold, USD, VIX, US yields, global index closes.
- Market membership: VN30/VN100/HOSE universe flags khi cần.

Các chỉ số được tính sẵn trong data hub gồm return 1/5/20/60/120/252 ngày, SMA/EMA, khoảng cách tới MA, RSI14, ATR14, realized/downside volatility, drawdown, gap, close-location, traded value, volume/value ratio, vị trí 52 tuần, relative strength/beta/correlation so với VNINDEX/VN30, intraday VWAP/range/return/volume-per-minute, breadth toàn universe, cross-section rank, sector aggregate, và 10-level depth imbalance nếu có cache depth.

## Output Chính

`data-hub/latest/latest_metrics.csv` là bảng đọc nhanh nhất: mỗi ticker một dòng, ghép các metric mới nhất từ daily, intraday, depth, flows, fundamentals và macro cache nếu có.

`data-hub/latest/START_HERE.json` là entrypoint cho ChatGPT/GDrive/repo connector. Nó chỉ rõ thứ tự đọc tối thiểu, rule không-news/không-model, universe, và file nào nên mở theo từng tình huống.

`data-hub/latest/bundles/` chứa các file nhỏ cho lượt đọc đầu: `source_audit.csv`, `market_snapshot.csv`, `symbol_latest.csv`, `retrieval_map.json`.

`data-hub/latest/index/` chứa catalog để truy xuất sâu: `ticker_catalog.csv`, `file_catalog.csv`, `column_catalog.csv`.

`data-hub/latest/manifest.json` là contract cho ChatGPT: nó mô tả mục đích, danh sách ticker, file nào có mặt, và API catalog. Nếu một nguồn chưa có cache thì data hub bỏ qua nguồn đó thay vì bịa dữ liệu.

`data-hub/latest/source_status.csv` ghi lại nguồn nào đã được thử, trạng thái `ok`/`partial`/`error`/`skipped`, số ticker, số dòng output, và file bằng chứng nếu có.

`data-hub/latest/calculation_catalog.csv` giải thích các nhóm phép tính đã được tạo. `market/breadth_daily.csv` và `market/cross_section_latest.csv` giúp ChatGPT hiểu bối cảnh thị trường trước khi soi từng mã.

## Maintenance

```bash
./broker.sh tests
./broker.sh validate_layout --ticker VIC
./broker.sh refresh_macro
./broker.sh refresh_bctt
./broker.sh refresh_vic_map
```

Repo này không đưa ra lời khuyên đầu tư. Nó chỉ chuẩn bị dữ liệu số để người dùng hoặc ChatGPT phân tích ở lớp bên ngoài.
