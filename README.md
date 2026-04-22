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

1. Refresh universe map nếu bạn muốn screen `VN30 + NVL` ngay:

```bash
./broker.sh map
```

2. Hoặc chạy một phát toàn bộ warm-up cần thiết:

```bash
./broker.sh prepare_default
```

Lệnh trên sẽ refresh scope thành `VN30 + NVL` rồi build snapshot/report tuần tự cho session. Nếu map đã đúng sẵn và bạn chỉ muốn rebuild artifact thì vẫn có thể dùng `./broker.sh prepare`.

3. Mở Codex ngay trong repo này và hỏi trực tiếp.

Nếu muốn soi sâu một mã sau khi warm-up xong:

```bash
./broker.sh deep VIC
```

Lệnh này sẽ tổng hợp `snapshot + range + timing + OHLC + entry ladder + playbook + research state` thành một report riêng dưới `out/deep_dive/`.

Nếu muốn dựng ranking ứng viên thống nhất thành artifact trước:

```bash
./broker.sh candidates auto
```

Artifact sẽ nằm dưới `out/analysis/candidates/` với 2 mức:

- `candidate_watchlist_core.*`: đủ để trả lời nhanh bằng snapshot + playbook
- `candidate_watchlist_full.*`: thêm timing + OHLC + ladder + research state

Nếu bạn muốn dùng artifact đã build trên GitHub Actions mà không commit snapshot vào repo:

```bash
./broker.sh sync_artifacts
```

Lệnh này sẽ tìm artifact mới nhất có prefix `core-artifacts-` trên branch `main`, chỉ download nếu `digest` chưa có trong cache local, cập nhật `.cache/gh-artifacts/latest/core-artifacts`, và prune cache local cũ.

Ví dụ:

- `Nếu artifact chưa có hoặc stale thì tự chạy và chờ xong rồi mới phân tích.`
- `Sau khi refresh artifact xong, tự check thêm tin tức live 12-24h gần nhất rồi mới chốt câu trả lời; không đưa news vào broker.sh.`
- `Dựa trên snapshot mới nhất, VN30 + NVL hôm nay có những ứng viên nào?`
- `Liệt kê đầy đủ ứng viên theo format mua ngay / chờ / không mua.`
- `Nếu chưa có mã đủ chuẩn thì nói thẳng không mua.`
- `Nếu có mã mua được hoặc chờ được thì phải ghi rõ vùng giá cụ thể và size tham chiếu cho ngân sách 5 tỷ.`
- `Nếu cần chạy batch thì tự chạy tuần tự xong rồi mới trả lời, không trả lời giữa chừng rằng vẫn đang đợi artifact.`

## Wrapper commands

```bash
./broker.sh tests
./broker.sh engine
./broker.sh prepare
./broker.sh research
./broker.sh map
./broker.sh refresh_vn30_map
./broker.sh refresh_vn30_nvl_map
./broker.sh refresh_hose_map
./broker.sh sync_artifacts
./broker.sh prepare_default
./broker.sh candidates auto
./broker.sh deep VIC
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
- `out/analysis/candidates/`: watchlist xếp hạng thống nhất ở mức `core` và `full`
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
- dùng `./broker.sh refresh_vn30_nvl_map` nếu bạn muốn scope mặc định là `VN30 + NVL`
- dùng `./broker.sh refresh_hose_map` nếu muốn screen rộng hơn

## Ghi chú vận hành với Codex

Repo này được thiết kế để mở một session Codex mới rồi làm việc như notebook:

- Codex được phép đọc `out/`, `research/`, `config/`, `scripts/`
- Codex được phép sửa tool hoặc thêm utility nếu cần cho workflow research
- Không giả định có flow order execution downstream
- Nếu artifact thiếu hoặc stale, Codex phải tự chạy tuần tự và tự đợi batch xong trước khi trả lời
- Sau bước refresh artifact, Codex phải tự browse tin tức live cùng ngày hoặc 12-24h gần nhất để overlay macro/geopolitics/policy khi trả lời `hôm nay mua gì`; lớp này là bước hỏi đáp, không phải lệnh batch của repo
- Không dùng flow nền hay nhiều builder chồng nhau, trừ khi từng job ghi ra output riêng và không dùng chung cache/history
- Khẩu vị mặc định của repo này là: ngân sách tham chiếu khoảng `5 tỷ`, ưu tiên size lớn, và phải liệt kê đầy đủ ứng viên khả thi thay vì ép chọn đúng một mã
- Contract đầu ra mặc định là: liệt kê ứng viên theo `mua ngay`, `chờ`, hoặc `không mua`; với mỗi mã `mua ngay` hoặc `chờ`, phải nêu `vùng giá cụ thể` và `quy mô vốn/số lượng` nếu chọn mã đó làm idea chính

## Kiểm thử

Sau khi sửa Python code:

```bash
./broker.sh tests
```

## Cảnh báo

Công cụ này không đưa ra lời khuyên đầu tư. Nó chỉ chuẩn bị dữ liệu và artifact để bạn phân tích thủ công.
