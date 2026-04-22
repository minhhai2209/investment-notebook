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
- hỗ trợ force-include extra tickers như `NVL` vào scope live khi cần

2. `scripts/engine/data_engine.py`

- đọc `data/industry_map.csv`
- lấy history + intraday từ các data fetchers
- tính technical snapshot, breadth, relative strength, sector context
- ghi `out/universe.csv`, `out/positions.csv`, `out/market_summary.json`, `out/sector_summary.csv`

3. `scripts/analysis/*`

- build candidate watchlist ở 2 mức `core` và `full`
- build range forecast
- build cycle forecast
- build per-ticker playbook
- build next-session OHLC forecast
- build intraday rest-of-session forecast
- build single-name timing
- build entry ladder evaluation
- build single-ticker deep dive that fuses snapshot, ML layers, and research state into one deterministic report
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
                         |
                         v
              live news overlay at answer time
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
- `prepare_default`
- `research`
- các report builder riêng lẻ
- các harness offline
- `refresh_vn30_map`, `refresh_hose_map`
- `map`, `refresh_vn30_nvl_map`
- `candidates`, `deep`
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
- contract của session tương tác là trả lời thực dụng theo `mua ngay / chờ / không mua`, liệt kê đầy đủ ứng viên khả thi thay vì ép đúng `1` mã
- nếu artifact còn thiếu hoặc stale, Codex phải tự chạy batch và tự đợi xong rồi mới trả lời; không được dừng ở một câu trả lời trung gian kiểu `đang chờ artifact`
- sau khi artifact đã sẵn sàng, Codex phải tự xem thêm tin tức live cùng ngày hoặc 12-24h gần nhất để overlay macro/geopolitics/policy trước khi chốt câu trả lời `hôm nay mua gì`; lớp này nằm ở interactive session, không phải builder hay subcommand batch
- chạy tuần tự là mặc định; chỉ song song hóa khi các job thật sự độc lập và không ghi/đọc chung cache hoặc history
- `prepare` giữ nghĩa là rebuild tuần tự trên scope hiện có; `prepare_default` là shortcut tuần tự cho scope mặc định `VN30 + NVL`
