# System Design

## Mục tiêu

`investment-notebook` là notebook tương tác để dự báo thị trường/mã bằng dữ liệu. Mục tiêu hiện tại của repo là:

- dựng snapshot dữ liệu sạch cho forecast
- build các artifact ML/diagnostic theo từng horizon
- cho phép Codex đọc artifact và giải thích forecast trực tiếp

Repo này không còn:

- TCBS browser automation
- order generation / order placement
- ngân sách mặc định, position sizing, ladder, hoặc khuyến nghị giao dịch
- portfolio workflow mặc định
- codex batch runner kiểu `orders.csv`
- bundle/prompt contract phục vụ execution workflow cũ

## Kiến trúc hiện tại

1. `scripts/tools/refresh_industry_map.py`

- refresh `data/industry_map.csv`
- hỗ trợ rebuild scope từ `VIC`, `VN30`, `HOSE`, `VN100`, hoặc CSV người dùng cung cấp
- workflow mặc định pin scope vào `VIC`

2. `scripts/engine/data_engine.py`

- đọc `data/industry_map.csv`
- lấy history + intraday từ các data fetchers
- tính technical snapshot, breadth, relative strength, sector context
- ghi `out/universe.csv`, `out/market_summary.json`, `out/sector_summary.csv`
- vẫn ghi `out/positions.csv` rỗng để tương thích schema cũ, nhưng config mặc định không đọc danh mục

3. `scripts/analysis/*`

- build next-session/multi-session OHLC forecast
- build intraday rest-of-session forecast
- audit feature/model bằng walk-back
- giữ một số builder legacy để diagnostic khi gọi tay, không phải nguồn trả lời mặc định

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
  out/universe.csv / market_summary.json / sector_summary.csv / positions.csv(empty by default)
                         |
                         v
                 forecast/evaluation builders
                         |
                         v
                  out/analysis/*
                         |
                         v
                 research bundle / reports
                         |
                         v
               Codex interactive session
```

## Portfolio Handling

Danh mục không còn là workflow mặc định.

- `config/data_engine.yaml` đặt `portfolio.enabled: false`.
- Khi tắt portfolio, engine không đọc `data/portfolios/portfolio.csv`, không merge vị thế vào universe, và không kéo ticker từ portfolio vào scope.
- `out/positions.csv` vẫn có thể được ghi rỗng để tránh phá schema cũ.
- Chỉ bật lại `portfolio.enabled: true` khi cần chạy một phân tích legacy rõ ràng.

## Wrapper CLI

`broker.sh` giờ chỉ là utility wrapper mỏng:

- `engine`
- `prepare`
- `prepare_default`
- `research`
- active forecast builders: `ohlc`, `intraday`
- feature/model evaluations
- legacy diagnostic builders
- `refresh_vic_map`, `refresh_vn30_map`, `refresh_hose_map`, `map`
- `tests`

Không còn subcommand `tcbs`, `orders`, `codex`, `portfolio`.

## Nguồn dữ liệu

- VNDIRECT: daily OHLCV và intraday cache
- CafeF: foreign/proprietary flow
- Vietstock overview: valuation / quality snapshot
- Vietstock BCTT: quarterly financial statements cho harness lift/evaluation
- Vietstock board/company pages: constituent lists và sector mapping

## Nguyên tắc repo

- fail-fast nếu thiếu input bắt buộc hoặc schema sai
- mọi artifact generated nằm dưới `out/`, `research/`, hoặc `reports/`
- không mang giả định execution downstream
- contract của session tương tác tách hai lớp: `Model predict` cho forecast/validation/error band từ artifact repo, và `External synthesis` cho bối cảnh tổng hợp từ nguồn ngoài nếu có
- không tự sinh recommendation, ngân sách, danh mục, position sizing, hoặc ladder
- không trộn web/news/source mạng vào forecast; dữ liệu ngoài chỉ trở thành model evidence khi được fetcher/pipeline của repo đưa thành artifact/model feature có thể audit
- nếu artifact forecast còn thiếu hoặc stale, Codex phải tự chạy batch và tự đợi xong rồi mới trả lời
- chạy tuần tự là mặc định; chỉ song song hóa khi các job thật sự độc lập và không ghi/đọc chung cache hoặc history
- `prepare` giữ nghĩa là rebuild tuần tự trên scope hiện có; `prepare_default` là shortcut tuần tự cho scope mặc định `VIC`
