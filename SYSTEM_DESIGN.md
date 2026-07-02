# System Design

## Mục Tiêu

Repo này là numeric data hub cho thị trường chứng khoán Việt Nam. Nó thu thập dữ liệu số, chuẩn hóa cache, tính metric cơ bản, rồi đóng gói thành artifact nhỏ để ChatGPT browser nhanh.

Không có lớp dự báo, không có report khuyến nghị, không có tin tức.

## Luồng Dữ Liệu

```mermaid
flowchart LR
    A["Numeric APIs"] --> B["Raw caches under out/"]
    B --> C["Normalize and enrich"]
    C --> D["data-hub/latest"]
    D --> E["ChatGPT browsing"]
```

## API Inventory

Nguồn số đang được hỗ trợ hoặc inventory:

- VNDIRECT dchart: OHLCV daily/intraday.
- VNDIRECT priceboard: snapshot bid/ask depth.
- CafeF flows: khối ngoại và tự doanh.
- Vietstock overview: valuation snapshot.
- Vietstock BCTT: financial statement numeric tables.
- FRED/Stooq: macro market numeric series.
- Market membership sources: VN30/VN100/HOSE flags.

`data-hub/latest/api_catalog.csv` là bản catalog để ChatGPT xem nhanh source nào tạo ra loại số nào.

## Artifact Contract

`data-hub/latest/manifest.json` là điểm bắt đầu. Manifest chứa:

- timestamp tạo artifact
- danh sách ticker
- mục đích numeric-only
- thứ tự đọc file cho ChatGPT
- danh sách file có mặt
- API catalog

Output quan trọng nhất:

- `source_status.csv`: audit nguồn nào đã thử, thành công, partial, lỗi, bị skip, và output bằng chứng.
- `latest_metrics.csv`: một dòng mỗi ticker, ghép metric mới nhất.
- `calculation_catalog.csv`: nhóm phép tính đã tạo và input/output của từng nhóm.
- `market/cross_section_latest.csv`: xếp hạng latest returns, liquidity, volatility, drawdown, relative strength.
- `market/breadth_daily.csv`: breadth theo ngày trên configured universe.
- `market/sector_latest.csv`: sector aggregate nếu có `data/industry_map.csv`.
- `daily/{TICKER}.csv`: daily OHLCV và technical metrics.
- `intraday/{TICKER}.csv`: intraday 1m recent rows.
- `intraday/minute_profile/{TICKER}.csv`: latest-day minute-by-minute return, volume, VWAP, volume rate.
- `depth/latest_depth.csv`: order book metrics nếu có cache.
- `fundamentals/vietstock_overview.csv`: valuation metrics nếu có cache.
- `flows/cafef_flows.csv`: flow metrics nếu có cache.
- `macro/latest_macro.csv`: macro latest values nếu có cache.

## Refresh Policy

`./broker.sh hub` chỉ build artifact từ cache, không tự gọi API.

`./broker.sh collect` gọi các API số được bật trong `config/data_hub.yaml`, sau đó build lại artifact. Default bật các nguồn nhanh và hữu ích: OHLCV, depth, CafeF flows, Vietstock overview và macro. Vietstock BCTT là nguồn quarterly dùng Playwright nên có command riêng `./broker.sh refresh_bctt`.

## Calculation Layers

- Trend/momentum: returns 1/5/20/60/120/252 ngày, SMA/EMA distance, RSI, 52-week position.
- Risk: ATR, realized volatility, downside volatility, drawdown, gap, close-location.
- Liquidity: traded value, ADV, average value, volume/value ratios.
- Relative strength: excess return, beta và correlation so với VNINDEX/VN30.
- Market breadth: advancers/decliners, up-value share, above-MA share, new highs/lows.
- Intraday microstructure: volume per minute, cumulative VWAP, return per minute, latest-day minute profile.
- Cross-section: rank các ticker theo return, liquidity, volume spike, volatility, drawdown.

## Non-Goals

- Không backtest.
- Không model selection.
- Không forecast.
- Không tin tức.
- Không recommendation.
- Không portfolio/order sizing.
