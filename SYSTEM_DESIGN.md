# System Design

## Mục tiêu

`investment-notebook` là spin-off interactive-only của repo cũ. Mục tiêu của repo này là:

- dựng snapshot dữ liệu sạch cho screening và thesis review
- build các report ML/heuristic theo từng mã
- cho phép Codex làm việc trực tiếp như một notebook nghiên cứu

Repo này không còn:

- TCBS browser automation
- order generation / order placement
- codex batch runner kiểu `orders.csv`
- bundle/prompt contract phục vụ execution workflow cũ

## Kiến trúc hiện tại

1. `scripts/tools/refresh_industry_map.py`

- refresh `data/industry_map.csv`
- hỗ trợ rebuild scope từ `VN30`, `HOSE`, `VN100`, hoặc CSV người dùng cung cấp

2. `scripts/engine/data_engine.py`

- đọc `data/industry_map.csv`
- lấy history + intraday từ các data fetchers
- tính technical snapshot, breadth, relative strength, sector context
- ghi `out/universe.csv`, `out/positions.csv`, `out/market_summary.json`, `out/sector_summary.csv`

3. `scripts/analysis/*`

- build range forecast
- build cycle forecast
- build per-ticker playbook
- build next-session OHLC forecast
- build intraday rest-of-session forecast
- build single-name timing
- build entry ladder evaluation
- giữ các harness offline để replay deterministic/ML baselines

4. `scripts/research/build_research_bundle.py`

- đọc snapshot live và các artifact trong `out/analysis/`
- dựng `research/manifest.json` và note/state per ticker
- layer này để Codex đọc nhanh hơn trong session tương tác

## Dòng dữ liệu

```text
refresh_industry_map -> data/industry_map.csv
                         |
                         v
                    data_engine
                         |
                         v
  out/universe.csv / market_summary.json / sector_summary.csv / positions.csv
                         |
                         v
                 analysis builders
                         |
                         v
                  out/analysis/*
                         |
                         v
                 research bundle
                         |
                         v
               Codex interactive session
```

## Portfolio handling

Danh mục không còn là dependency bắt buộc.

- Nếu `data/portfolios/portfolio.csv` tồn tại, engine vẫn merge context vị thế.
- Nếu file không tồn tại, engine vẫn chạy; `positions.csv` rỗng và các cột vị thế trong `universe.csv` về `0/NaN`.

Điểm này giúp repo dùng tốt cho screening thuần, không cần scrape portfolio trước.

## Wrapper CLI

`broker.sh` giờ chỉ là utility wrapper mỏng:

- `engine`
- `prepare`
- `research`
- các report builder riêng lẻ
- các harness offline
- `refresh_vn30_map`, `refresh_hose_map`
- `tests`

Không còn subcommand `tcbs`, `orders`, `codex`, `portfolio`.

## Nguồn dữ liệu

- VNDIRECT: daily OHLCV và intraday cache
- CafeF: foreign/proprietary flow
- Vietstock overview: valuation / quality snapshot
- Vietstock BCTT: quarterly financial statements cho harness lift/evaluation
- Vietstock board/company pages: constituent lists và sector mapping

## Nguyên tắc repo mới

- fail-fast nếu thiếu input bắt buộc hoặc schema sai
- mọi artifact generated nằm dưới `out/` hoặc `research/`
- không mang giả định execution downstream
- nếu cần thay workflow research, thay ngay ở tool/script trong repo thay vì vòng vo qua prompt bundle
