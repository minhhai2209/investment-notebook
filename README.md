# Investment Notebook

> Repo này là notebook tương tác cho nghiên cứu và screening cổ phiếu Việt Nam. Nó không còn flow đặt lệnh, không còn TCBS automation, và không còn wrapper chạy Codex theo kiểu `orders.csv`.

## Repo này giữ gì

- Engine lấy dữ liệu và dựng snapshot: `scripts/engine/data_engine.py`
- Các fetcher/indicator hỗ trợ: `scripts/data_fetching/`, `scripts/indicators/`
- Các report/harness phân tích và ML: `scripts/analysis/`
- Research bundle có cấu trúc để Codex đọc trực tiếp: `scripts/research/build_research_bundle.py`
- Tool refresh universe map theo rổ live: `scripts/tools/refresh_industry_map.py`

## Repo này đã bỏ gì

- TCBS login / scrape danh mục qua browser
- TCBS order placement
- `codex_universe/`, `orders.csv`, `DONE.md`, prompt bundle, archive lịch sử lệnh
- Strategy buckets và overlay phục vụ order workflow cũ

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

1. Refresh universe map nếu bạn muốn screen `VN30` ngay:

```bash
./broker.sh refresh_vn30_map
```

2. Build snapshot và toàn bộ report cần cho phiên phân tích:

```bash
./broker.sh prepare
```

3. Mở Codex ngay trong repo này và hỏi trực tiếp.

Ví dụ:

- `Dựa trên snapshot mới nhất, VN30 hôm nay có đúng 1 mã đáng stalk không?`
- `So top 5 candidate trong VN30 rồi chọn ra 1 mã tốt nhất theo reward/risk hiện tại.`
- `Nếu chưa có mã đủ chuẩn thì nói thẳng không mua.`

## Wrapper commands

```bash
./broker.sh tests
./broker.sh engine
./broker.sh prepare
./broker.sh research
./broker.sh refresh_vn30_map
./broker.sh refresh_hose_map
./broker.sh range
./broker.sh cycle
./broker.sh playbook
./broker.sh ohlc
./broker.sh intraday
./broker.sh timing
./broker.sh entry_ladder
```

Các harness offline vẫn còn:

```bash
./broker.sh eval_deterministic
./broker.sh eval_ml
./broker.sh eval_vnindex
./broker.sh eval_ohlc
./broker.sh eval_macro
./broker.sh eval_bctt
```

## Output chính

- `out/universe.csv`: snapshot hợp nhất để screen và đọc tape/context
- `out/positions.csv`: vẫn được ghi, nhưng có thể rỗng nếu repo không có `data/portfolios/portfolio.csv`
- `out/market_summary.json`: breadth, range, co-movement ở cấp thị trường
- `out/sector_summary.csv`: breadth/relative strength ở cấp ngành
- `out/analysis/`: các report ML và evaluation
- `research/`: bundle research theo mã để Codex đọc nhanh hơn trong session tương tác

## Danh mục là optional

Repo notebook không cần danh mục để chạy. Nếu không có `data/portfolios/portfolio.csv`:

- engine vẫn chạy bình thường
- `positions.csv` sẽ rỗng
- các cột vị thế trong `universe.csv` sẽ về `0` hoặc `NaN`

Nếu bạn muốn dùng thêm context vị thế nội bộ thì chỉ cần tự đặt file `data/portfolios/portfolio.csv` với schema:

```csv
Ticker,Quantity,AvgPrice
FPT,1000,118.5
MBB,2000,23.4
```

## Universe mặc định

`config/data_engine.yaml` không còn pin working universe vào một mã cũ. Scope thực tế được quyết định bởi `data/industry_map.csv`.

Khuyến nghị:

- dùng `./broker.sh refresh_vn30_map` nếu repo này chủ yếu để chọn ứng viên trong `VN30`
- dùng `./broker.sh refresh_hose_map` nếu muốn screen rộng hơn

## Ghi chú vận hành với Codex

Repo này được thiết kế để mở một session Codex mới rồi làm việc như notebook:

- Codex được phép đọc `out/`, `research/`, `config/`, `scripts/`
- Codex được phép sửa tool hoặc thêm utility nếu cần cho workflow research
- Không giả định có flow order execution downstream
- Khẩu vị mặc định của repo này là `single-name, fast deployment`: ngân sách khoảng `5 tỷ`, ưu tiên giải ngân nhanh cho một mã duy nhất nếu vùng mua đủ tốt; nếu không có mã sạch thì kết luận `không mua`

## Kiểm thử

Sau khi sửa Python code:

```bash
./broker.sh tests
```

## Cảnh báo

Công cụ này không đưa ra lời khuyên đầu tư. Nó chỉ chuẩn bị dữ liệu và artifact để bạn phân tích thủ công.
