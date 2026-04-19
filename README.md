# Broker GPT Data Engine

> Công cụ này không đưa ra lời khuyên đầu tư. Nó chỉ thu thập dữ liệu giá, tính toán chỉ số kỹ thuật và ghi lại kết quả để bạn ra quyết định thủ công.

## Tổng quan

Data engine được thiết kế lại để làm đúng một việc: chuẩn bị dữ liệu sạch cho ChatGPT (hoặc bất kỳ công cụ phân tích nào khác) sử dụng. Mỗi lần chạy engine sẽ:

1. Thu thập dữ liệu lịch sử và intraday cho toàn bộ vũ trụ mã.
2. Chuẩn hoá snapshot kỹ thuật (SMA/EMA/RSI/ATR/MACD, lợi suất và biên độ 52w) để làm nền cho output hợp nhất.
3. Tính biên trần/sàn, sizing và tín hiệu phụ trợ rồi gộp thành snapshot hợp nhất.
4. Làm giàu danh mục/sector theo giá hiện tại và kết xuất `out/universe.csv` (toàn bộ trường kỹ thuật + Sector + cột vị thế + cột EngineRunAt + các chỉ số khách quan như relative strength, breadth/co-movement, market-relative/sector-relative returns), `out/positions.csv` (snapshot vị thế chi tiết để hậu kiểm), `out/market_summary.json` (range/breadth/co-movement ở cấp thị trường), và `out/sector_summary.csv` (breadth/return/flow ở cấp ngành).

Không còn bước tạo lệnh tự động, không còn phụ thuộc Vietstock, không còn overlay policy phức tạp. Bạn chủ động đọc các file CSV và đưa ra quyết định.

## Chuẩn bị môi trường

- Python 3.10 trở lên.
- macOS, Linux hoặc WSL đều chạy được.
- Khuyến nghị tạo virtualenv trước khi chạy.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Cấu hình engine

File chính: `config/data_engine.yaml`

```yaml
universe:
  csv: data/industry_map.csv    # Danh sách mã + sector
  core_tickers: [NVL]  # Working universe daily hiện tại
technical_indicators:
  moving_averages: [20, 50, 200]
  ema_periods: [20, 50]
  rsi_periods: [14]
  atr_periods: [14]
  returns_periods: [5, 20, 60]
  bollinger:
    windows: [20]
    k: 2
    include_bands: false
  range_lookback_days: 252
  adv_periods: [20]
  macd:
    fast: 12
    slow: 26
    signal: 9
portfolio:
  directory: data/portfolios     # Chứa duy nhất file portfolio.csv
output:
  base_dir: out
  presets_dir: .
  portfolios_dir: .
execution:
  aggressiveness: med
  max_order_pct_adv: 0.1
  slice_adv_ratio: 0.25
  min_lot: 100
  max_qty_per_order: 500000
data:
  history_cache: out/data
  history_min_days: 1
  intraday_window_minutes: 60
  # Optional: override HOSE reference prices when có điều chỉnh tham chiếu
  # reference_overrides: data/reference_overrides.csv   # schema: Ticker,Ref
```

Bạn có thể tinh chỉnh tham số chỉ báo, đường dẫn output hoặc giới hạn sizing (`execution`) tuỳ nhu cầu.

### Cập nhật rổ VN100 (`data/industry_map.csv`)

`data/industry_map.csv` là nguồn dữ liệu gốc cho universe (schema bắt buộc: `Ticker,Sector`).

Mặc định, `./broker.sh tcbs` giờ sẽ tự refresh file này theo flow:
- lấy `core_tickers` từ `config/data_engine.yaml`
- merge thêm ticker đang có trong `data/portfolios/portfolio.csv`
- giữ lại sector đã cache sẵn trong `data/industry_map.csv` nếu còn dùng được, và chỉ fetch sector còn thiếu từ hồ sơ doanh nghiệp Vietstock

- Rebuild từ `out/universe.csv` (offline, giữ nguyên sector đang có trong file output):
  - `python scripts/tools/refresh_industry_map.py --from-tickers-csv out/universe.csv --output data/industry_map.csv`
- Refresh theo đúng flow runtime hiện tại:
  - `./broker.sh tcbs`
- Nếu muốn refetch sector từ Vietstock cho mọi mã thay vì chỉ mã còn thiếu:
  - thêm `--refresh-existing`

Các nguồn dữ liệu bổ sung được điều khiển qua khối `data` trong cấu hình:
- `cafef_flow_enabled` + `cafef_flow_cache` + `cafef_flow_max_age_hours`: khi bật, engine sẽ merge thêm dòng tiền khối ngoại/tự doanh từ CafeF (tính chất daily). Giá trị `cafef_flow_max_age_hours` > 0 (mặc định `4` giờ trong `config/data_engine.yaml`) cho phép refresh định kỳ khi cache cũ hơn TTL; đặt `0` nếu bạn muốn chỉ dùng cache hiện có và không fetch thêm.
- `vietstock_overview_enabled` + `vietstock_overview_cache` + `vietstock_overview_max_age_hours`: tương tự cho chỉ tiêu cơ bản tóm tắt từ Vietstock (sức khoẻ doanh nghiệp, non‑daily). Ở đây mặc định `vietstock_overview_max_age_hours: 0` để coi dữ liệu là non‑daily và chỉ fetch khi cache chưa có; đặt giá trị > 0 nếu bạn muốn refresh định kỳ.

## Cách chạy

### Engine + Scraper (mặc định)

1. Chuẩn bị `.env` ở repo root:

```bash
TCBS_USERNAME=you-username
TCBS_PASSWORD=your-password-here
```

`broker.sh` sẽ tự nạp `.env` ở repo root cho toàn bộ subcommand (`tcbs`, `codex`, `research`, `orders`, ...), nên các biến như `TCBS_USERNAME`, `TCBS_PASSWORD`, `TCBS_ACCOUNT_SLUG`, hay `INDUSTRY_TICKER_FILTER` không cần export tay nếu đã có trong file này.

2. Chạy theo từng bước (không còn lệnh mặc định / `all`):

```bash
./broker.sh tcbs_login      # mở TCBS, bấm login, chờ MFA rồi thoát để giữ session
./broker.sh tcbs --headful   # lấy danh mục + chạy engine để cập nhật universe.csv
./broker.sh research         # rebuild research per-ticker và sync sang codex_universe/
./broker.sh codex            # chạy Codex với universe.csv hiện tại
```

### Engine

Engine chạy tự động sau `./broker.sh tcbs`. Nó sẽ:

- Gọi API VNDIRECT để cập nhật giá lịch sử + intraday.
- Refresh `data/industry_map.csv` theo working universe hiện hành. Nếu `config/data_engine.yaml` có `universe.core_tickers` thì wrapper refresh theo đúng `core_tickers`; nếu không có thì fallback về seed mặc định `VN100 live + portfolio + NVL`.
- Tính SMA/RSI/ATR/MACD theo cấu hình.
- Tính thêm breadth/co-movement, thống kê theo ngành, và relative strength để hỗ trợ việc đọc dữ liệu thị trường/ngành/mã.
- Xuất `universe.csv` (ghép snapshot thị trường + cột vị thế + EngineRunAt + các đại lượng khách quan) và `positions.csv` (tách riêng vị thế hiện hành).
- Xuất `market_summary.json` và `sector_summary.csv` trong `out/` để review insight ngoài file universe chính.
- Chạy thêm `scripts/analysis/build_range_forecast_report.py --universe-csv out/universe.csv` để tạo forecast ML giá/xu hướng ngắn hạn.
- Chạy thêm `scripts/analysis/build_cycle_forecast_report.py --universe-csv out/universe.csv` để tạo cycle forecast `1M..6M` ngay trên working universe hiện hành.
- Chạy thêm `scripts/analysis/build_ticker_playbook_report.py` trên chính working universe hiện hành để rút `playbook` kỹ thuật tốt nhất cho từng mã.
- Chạy thêm `scripts/analysis/build_ohlc_next_session_report.py --universe-csv out/universe.csv` để tạo forecast OHLC `T+1` tốt nhất theo từng mã.
- Nếu snapshot vẫn còn cửa giao dịch trong ngày (`AM_EARLY`, `AM_LATE`, `LUNCH_BREAK`, `PM_EARLY`, `PM_LATE`), chạy thêm `scripts/analysis/build_intraday_rest_of_session_report.py --universe-csv out/universe.csv` để tạo forecast cho phần còn lại của phiên.
- Chạy thêm `scripts/analysis/build_single_name_timing_report.py --universe-csv out/universe.csv` để tạo lớp forecast ngắn hạn theo từng mã về upside, days-to-peak, drawdown và hiệu quả giữ vốn trong một vòng trade tập trung.
- Chạy thêm `scripts/analysis/build_entry_ladder_eval_report.py --universe-csv out/universe.csv ...` để re-anchor edge theo từng mức `LimitPrice` BUY hợp lệ, thay vì chỉ chấm theo `Base/Last`; bản hiện tại còn fit classifier riêng theo từng mã/horizon để ước lượng xác suất chạm giá ở `T+1/T+5/T+10`.
- Chạy thêm `scripts/research/build_research_bundle.py` để build research per-ticker dưới `research/`: `manifest.json`, `profile.md`, `weekly/*.md`, `daily/*.md`, và `state.json` cho từng mã trong working universe.
- Chạy thêm `scripts/codex/build_bundle_manifest.py` để dựng `bundle_manifest.json` ngay trong bundle live; đây là contract kỹ thuật cho prompt daily khi kiểm tra schema/presence của các file input.
- Trong live flow có `out/universe.csv`, cả range forecast lẫn cycle forecast đều phải phủ toàn bộ tập ticker được sync sang `codex_universe/`; nếu thiếu ticker so với live universe thì wrapper sẽ fail thay vì sync file hụt.
- Đồng bộ `out/universe.csv`, `out/market_summary.json`, `out/sector_summary.csv`, `out/analysis/ml_range_predictions_full_2y.csv`, `out/analysis/ml_range_predictions_recent_focus.csv`, `out/analysis/ml_cycle_forecast/cycle_forecast_ticker_matrix.csv`, `out/analysis/ml_cycle_forecast/cycle_forecast_best_horizon_by_ticker.csv`, `out/analysis/ticker_playbooks_live/ticker_playbook_best_configs.csv`, `out/analysis/ml_ohlc_next_session.csv`, và nếu có thì cả `out/analysis/ml_intraday_rest_of_session.csv`, `out/analysis/ml_single_name_timing.csv`, `out/analysis/ml_entry_ladder_eval.csv`, `human_notes.md`, `strategy_buckets.csv`, và thư mục `research/` sang `codex_universe/`; sau đó wrapper build thêm `codex_universe/bundle_manifest.json` để Codex đọc dữ liệu mới nhất theo contract thống nhất.
- Nếu `config/data_engine.yaml` có `universe.core_tickers`, engine sẽ coi đó là working universe active và chỉ tính universe/positions/bundle trên đúng tập mã này.
- Nếu endpoint VN30 từ Vietstock bị lỗi tạm thời, engine fallback sang danh sách VN30 static để không chặn pipeline.
- Xoá sạch thư mục `out/` trước khi chạy để tính toán lại toàn bộ.

### Hai lớp universe

Repo hiện có hai lớp rõ ràng:

- `Selection universe`: dùng cho bài tuyển chọn offline rộng hơn như `VN30`, `VN100`, hoặc một rổ người dùng chỉ định. Lớp này chỉ phục vụ screening và không thay đổi flow live.
- `Working universe`: dùng cho flow hàng ngày `tcbs -> engine -> codex`. Lớp này lấy từ active ticker filter, mặc định là `config/data_engine.yaml: universe.core_tickers`.

Ở snapshot cấu hình hiện tại trong repo, working universe daily là:

- `MBB`
- `HPG`
- `NVL`

Nói ngắn gọn:

- tuyển chọn thì có thể chạy rộng trên toàn `VN30`/`VN100`
- tính toán hàng ngày chỉ chạy trên `working universe` đã chốt

### Research per-ticker

Lớp research này phục vụ đúng nhu cầu "thesis theo mã có kế thừa" mà không phải chat tay mỗi lần:

- `./broker.sh tcbs` sẽ build luôn bundle research mới nhất sau khi các artifact ML live đã xong.
- `./broker.sh research` cho phép rebuild riêng bundle research từ snapshot hiện có trong `out/` mà không phải scrape TCBS lại.
- Output nằm dưới `research/` và không commit vào git:
  - `research/manifest.json`
  - `research/tickers/<TICKER>/profile.md`
  - `research/tickers/<TICKER>/weekly/<YYYY-Www>.md`
  - `research/tickers/<TICKER>/daily/<YYYY-MM-DD>.md`
  - `research/tickers/<TICKER>/state.json`

Contract dùng bundle này:

- `profile.md`: archetype/bản chất mã và kiểu đánh mặc định.
- `weekly/*.md`: thesis cấu trúc hiện hành trong tuần. Builder luôn rewrite theo `EngineRunAt` hiện tại, không reuse file cũ chỉ vì path cùng tuần đã tồn tại.
- `daily/*.md`: tactical note gần nhất từ snapshot live. Builder luôn rewrite theo `EngineRunAt` hiện tại để tránh lệch với `state.json`.
- `state.json`: bản tóm tắt machine-readable cho prompt daily đọc nhanh, gồm cả target weight band, weakness build, strength reserve, allocator reason, cặp cờ `AddOnStrengthAllowed` / `AddOnWeaknessAllowed` theo allowance hiện tại, và nếu có thì cả `HumanTargetPrice`, `PersistentWeaknessBid`, thống kê burst/follow-through kiểu `T+2.5`, cùng execution fields gần nhất như `ExecutionBias`, `OpeningSqueezeFailure`, `BurstExecutionBias`, `TrimAggression`, `UrgentTrimMode`, `MustSellFractionPct`; các cờ `Default*` giữ lại style mặc định của archetype để tránh lẫn với quyền add thực tế ở snapshot hiện tại. Khi allocator bật staged build, `SuggestedNewCapitalPct` là ngân sách cho phiên kế tiếp còn `DeferredBuildPct` là phần gap giữ lại cho các phiên sau. Nếu có `SessionBuyTranches`, đây là ladder split mặc định của riêng phiên đó theo từng `LimitPrice` và `% ngân sách phiên`; các role như `continuation_reserve`, `bridge`, `shallow_core`, `mid_core`, `deep_core` cho biết tranche đó đang chống lỡ nhịp, bắt cầu nối hay ưu tiên mua sâu.
- `manifest.json`: index chuẩn để prompt hoặc tool khác biết mã nào có artifact nào; manifest hiện cũng mang `PortfolioAllocator` để gợi ý tổng mức giải ngân mục tiêu và deployable gap ở cấp danh mục. `PortfolioAllocator` dùng cùng `total_capital_kVND` với prompt khi giá trị này được cung cấp, thay vì chỉ nhìn phần vốn đang nằm trong cổ phiếu. Nếu working universe chỉ còn rất ít mã, breadth thị trường trong `market_summary.json` sẽ chuyển sang lấy từ benchmark basket thay vì chính working universe để tránh regime bị méo bởi sample quá nhỏ. Nếu working universe chỉ còn một mã đầu tư, allocator tự chuyển sang `SingleNameMode` và để mã đó kế thừa target invested của cả danh mục, thay vì tự giới hạn ở band 10-20% kiểu danh mục nhiều mã; đồng thời `SessionBuildCapPct` giới hạn lượng vốn được phép đưa vào ngay phiên kế tiếp để không dồn hết gap vào một lượt. Khi universe có nhiều mã, allocator sinh thêm `GlobalBuyTranches` để xếp hạng chung các nấc BUY xuyên mã, tránh phải áp quota vốn cứng theo ticker.

### Quy trình tuyển chọn core để revisit sau này

Khi muốn thay đổi `core_tickers`, repo hiện theo quy trình thực dụng này:

1. Chạy screening rộng trên `VN100` làm `selection universe` mặc định, không sửa working universe ngay.
2. Đọc tin tức mới nhất liên quan tới thị trường và các ứng viên cuối trước khi chốt shortlist; không chỉ dựa vào ML/data snapshot.
3. Đọc đồng thời hai lớp ML:
   - `ml_range_predictions_*`: xem forecast giá/xu hướng ngắn hạn
   - `ml_cycle_forecast_*`: xem peak kỳ vọng, số phiên tới peak, drawdown, và `SelectionScore`
4. Loại sớm các nhóm có overhang cấu trúc mà ML không phản ánh hết:
   - bất động sản dự án/pháp lý phức tạp
   - commodity hoặc event-driven quá mạnh nếu không hợp mục tiêu hiện tại
   - ownership/governance khó nhìn hoặc headline risk cao
5. Giữ lại các doanh nghiệp đủ “xương sống”:
   - quy mô đủ lớn
   - thanh khoản đủ tốt
   - business dễ giải thích
   - ít màu tail risk kiểu pháp lý/chính sách/governance
6. Khi so ML, không chỉ nhìn upside:
   - ưu tiên mã có `SelectionScore` thấp hơn
   - drawdown dự báo không quá xấu
   - số phiên tới peak không quá dài nếu mục tiêu là tránh giam vốn
   - nếu có thêm close-fit/horizon-fit offline thì ưu tiên mã có model ổn định hơn
7. Chỉ promote mã vào `core_tickers` khi qua được cả ba lớp:
   - news / overhang filter
   - quality / legal-risk filter
   - ML filter
8. `preferred_tickers` là vùng người dùng tự thêm theo thesis riêng; không dùng chúng làm bằng chứng để thay đổi `core_tickers`, và cũng không tự mở rộng fetch universe nếu chưa có vị thế thực.
9. Revisit tối thiểu mỗi quý, hoặc ngay khi có một trong các trigger:
   - thesis ngành đổi mạnh
   - xuất hiện overhang pháp lý/governance mới
   - thanh khoản suy giảm rõ
   - ML score / drawdown profile xấu đi rõ rệt
   - hành vi giá cho thấy mã thường xuyên giam vốn lâu hơn mục tiêu

Contract `orders.csv` và flow `tcbs_orders` không thay đổi; prompt live hiện đọc range forecast, cycle forecast, `ticker_playbook_best_configs.csv`, `ml_ohlc_next_session.csv`, và nếu có thì cả `ml_intraday_rest_of_session.csv` lẫn `ml_single_name_timing.csv`. Điểm execution quan trọng là prompt giờ tách rõ hai ngữ cảnh:
- `overnight / phiên kế tiếp chưa mở`: batch phải là `single-side per ticker`. Prompt phải chọn một `ATO priority side` ở cấp batch (`BUY-first` hoặc `SELL-first`) theo objective/risk context của phiên, thay vì hard-code một phía. Nếu cùng một mã vừa có luận điểm `SELL` vừa có luận điểm `BUY`, prompt chỉ được giữ một phía cuối cùng cho mã đó; đồng thời `orders.csv` phải xếp các lệnh của phía ưu tiên lên trước để người vận hành biết nên nhập ATO phía nào trước.
- `in-session / phần còn lại của phiên hiện tại`: vẫn được phép cùng một mã có cả `SELL` và `BUY`, nhưng mọi dòng `SELL` của mã đó phải đứng trước mọi dòng `BUY` trong `orders.csv`.

### Replay deterministic strategies

Để hậu kiểm thesis `market regime -> breadth/co-movement -> relative strength -> chọn mã`, repo có thêm harness offline:

```bash
venv/bin/python scripts/analysis/evaluate_deterministic_strategies.py \
  --case-tickers HPG FPT SSI VCB NKG
```

Script này chỉ đọc cache sẵn có trong `out/data/` + `data/industry_map.csv`, không chạm vào flow live. Nó sẽ:

- so sánh nhiều market overlays và chấm bằng forward return / hit rate / drawdown của `VNINDEX`
- chọn market gate tốt nhất trong sample hiện có để gate các ticker algorithms
- replay nhiều ticker algorithms deterministic và xuất report dưới `out/analysis/`
- tạo case study cho các mã chỉ định như `HPG`, `FPT`, `SSI`, `VCB`, `NKG`

Các file chính trong `out/analysis/`:

- `market_algorithm_summary.csv`
- `market_bias_history.csv`
- `ticker_algorithm_summary.csv`
- `ticker_replay_top_picks.csv`
- `current_market_snapshot.csv`
- `current_top_picks.csv`
- `case_studies.csv`
- `summary.json`

### Per-ticker playbook backtest

Khi muốn rút ra “tính cách kỹ thuật” riêng cho từng mã thay vì áp một thesis giao dịch chung lên cả rổ, dùng harness này:

```bash
PYTHONPATH=. venv/bin/python scripts/analysis/build_ticker_playbook_report.py \
  --tickers REE,MBB,HPG,VCB,GMD,VNM,CTR,NVL,VIC,VHM \
  --output-dir out/analysis/ticker_playbooks_core_plus_event
```

Script này:

- backtest một tập family rule đơn giản theo từng mã (`washout_reclaim`, `trend_pullback`, `breakout_followthrough`, `trend_reacceleration`)
- tách lịch sử thành `train/test` theo thời gian để tránh nhìn một phía
- chọn `playbook` tốt nhất cho từng mã bằng điểm tổng hợp train/test
- ghi artifact dưới `out/analysis/` như:
  - `ticker_playbook_all_configs.csv`
  - `ticker_playbook_best_configs.csv`
  - `ticker_playbook_all_trades.csv`
  - `ticker_playbook_summary.md`

Mục tiêu của report này là trả lời câu hỏi kiểu:

- mã nào hợp `washout rồi mua phiên reclaim`
- mã nào hợp `pullback trong xu hướng`
- mã nào chỉ nên đợi `breakout-followthrough`
- mã nào chưa có rule cơ học đủ sạch dù doanh nghiệp vẫn tốt

Đây là lớp nghiên cứu offline để revisit working universe và cách vào lệnh theo từng mã. Hiện tại `broker.sh tcbs` đã build một bản live rút gọn trên working universe và sync `ticker_playbook_best_configs.csv` sang `codex_universe/` để prompt dùng như clue theo từng mã.

#### Cách regen

- Regen research trên một danh sách tự chọn:

```bash
PYTHONPATH=. venv/bin/python scripts/analysis/build_ticker_playbook_report.py \
  --tickers REE,MBB,HPG,VCB,GMD,VNM,CTR,NVL,VIC,VHM \
  --output-dir out/analysis/ticker_playbooks_core_plus_event
```

- Regen live bundle theo working universe hiện tại:

```bash
./broker.sh tcbs
```

Lệnh này sẽ tự:
- refresh `out/universe.csv`
- chạy `build_range_forecast_report.py`
- chạy `build_cycle_forecast_report.py`
- chạy `build_ticker_playbook_report.py` trên đúng working universe hiện tại
- sync `ticker_playbook_best_configs.csv` sang `codex_universe/`

#### Cách đọc nhanh

File chính để đọc là:
- `out/analysis/.../ticker_playbook_best_configs.csv`
- hoặc bản live `codex_universe/ticker_playbook_best_configs.csv`

Mỗi dòng là `một mã` và `một playbook tốt nhất hiện có` cho mã đó.

Đọc theo thứ tự này:
- `RuleFamily`: loại setup đang hợp nhất với mã đó.
  - `washout_reclaim`: mã hợp kiểu rơi mạnh rồi hồi lại
  - `trend_pullback`: mã hợp kiểu pullback trong xu hướng
  - `breakout_followthrough`: mã hợp kiểu vượt nền rồi chạy tiếp
  - `trend_reacceleration`: mã hợp kiểu đi ngang/hãm lại rồi tăng tốc lại
- `RuleLabel`: phiên bản cụ thể của rule đó. Đây là chuỗi để tra cứu/research lại, không cần cố nhớ từng tham số.
- `TestTrades`, `TestWinRatePct`, `TestAvgRetPct`, `TestAvgHoldDays`: 4 cột quan trọng nhất.
  - `TestTrades`: số mẫu gần đây có đủ dày hay không. Quá ít thì phải nghi ngờ.
  - `TestWinRatePct`: tỷ lệ thắng ở giai đoạn test gần đây.
  - `TestAvgRetPct`: mỗi lệnh trung bình lời/lỗ bao nhiêu phần trăm.
  - `TestAvgHoldDays`: thường phải cầm bao lâu.
- `TestWorstDrawdownPct`: setup đó có hay làm mình bị âm sâu không.
- `All*`: cùng ý nghĩa như trên nhưng trên toàn lịch sử, để so với `Test*`.

#### Cách diễn giải thực dụng

- Nếu `TestTrades` ít nhưng `AllTrades` đẹp: setup có thể đúng về mặt tính cách mã, nhưng sample gần đây thưa.
- Nếu `TestAvgRetPct` dương, `TestWinRatePct` ổn, và `TestWorstDrawdownPct` không quá sâu: đây là playbook có thể dùng như clue mạnh.
- Nếu `Test*` xấu nhưng `All*` đẹp: mã có thể đã đổi regime gần đây; không nên tin cứng vào lịch sử dài.
- Nếu cả `Test*` lẫn `All*` đều yếu: xem như chưa có setup cơ học sạch cho mã đó.
- Playbook không phải lệnh tự động. Nó chỉ trả lời câu hỏi: `mã này trong lịch sử gần đây hợp kiểu vào lệnh nào hơn`.

#### Ví dụ cách đọc

- `MBB` ra `trend_reacceleration`:
  đọc là mã này hợp kiểu tăng lại sau một nhịp hãm ngắn, không nhất thiết phải chờ washout sâu.
- `GMD` ra `breakout_followthrough`:
  đọc là mã này hợp kiểu vượt nền rồi đi tiếp hơn là bắt đáy.
- `VHM` ra `washout_reclaim`:
  đọc là nếu muốn đánh mã này thì logic lịch sử đẹp hơn là chờ rơi đủ mạnh rồi mua khi hồi xác nhận.

Nên dùng playbook như lớp bổ sung bên cạnh:
- `ml_range_predictions_*` để xem hướng/giá ngắn hạn
- `ml_cycle_forecast_*` để xem peak, thời gian tới peak và drawdown
- dữ liệu kỹ thuật trực tiếp trong `universe.csv`

### Replay ML baseline

Để thử machine learning mà vẫn giữ backtest sạch theo time split, repo có thêm baseline `scikit-learn`:

```bash
venv/bin/python scripts/analysis/evaluate_ml_models.py \
  --case-tickers HPG FPT SSI VCB NKG
```

Script này:

- dựng tập mẫu cross-sectional từ cache `out/data/`
- dùng walk-forward train/predict với target `xác suất outperform VNINDEX trong 10 phiên tới`
- so sánh các model baseline như `logistic`, `random forest`, `hist gradient boosting`
- ghi artifact dưới `out/analysis/` như `ml_model_summary.csv`, `ml_prediction_history.csv`, `ml_current_predictions.csv`, `ml_current_top_picks.csv`, `ml_case_studies.csv`, `ml_summary.json`

Đây mới là harness đánh giá offline; nó chưa thay đổi flow live `tcbs -> codex -> orders`.

### Replay ML cho VNINDEX

Nếu muốn chạy cùng bộ model ML trực tiếp trên state của `VNINDEX`:

```bash
venv/bin/python scripts/analysis/evaluate_vnindex_models.py
```

Script này dùng feature theo market breadth/regime/co-movement để train/predict walk-forward cho horizon `5` và `10` phiên, rồi ghi:

- `out/analysis/vnindex_ml_model_summary.csv`
- `out/analysis/vnindex_ml_prediction_history.csv`
- `out/analysis/vnindex_ml_current_forecast.csv`
- `out/analysis/vnindex_ml_summary.json`

Mục tiêu ở đây là trả lời câu hỏi kiểu `xác suất VNINDEX tăng trong 5/10 phiên tới là bao nhiêu` dựa trên cache hiện có, không thay thế deterministic overlay trong runtime.

### Replay ML dự báo OHLC/range trực tiếp cho T+1..T+10

Nếu muốn thử dự báo trực tiếp `open/high/low/close` và `range` cho từng horizon từ `T+1` tới `T+10` cho từng mã, có thêm harness:

```bash
venv/bin/python scripts/analysis/evaluate_ohlc_models.py \
  --tickers HPG FPT SSI VCB GAS \
  --retrain-every 20 \
  --min-train-dates 160 \
  --max-horizon 10
```

Script này:

- train riêng cho từng mã bằng lịch sử daily OHLCV của chính mã đó cộng với context `VNINDEX`
- shared daily feature layer hiện giữ cả context gần theo tuần như `week-to-date`, khoảng cách tới `high/low` của tuần trước, range/return của tuần trước, và nhịp volume tuần, nên thêm mã mới vẫn đi cùng một flow feature thay vì rẽ nhánh single-name
- train direct model riêng cho từng horizon `T+1..T+10`
- dự báo `open`, `close`, rồi biên trên/biên dưới để ghép lại thành OHLC hợp lệ cho đúng horizon đó
- chấm walk-forward theo sai số `%` của `open/high/low/close`, `range`, và độ đúng hướng của giá đóng cửa trên từng horizon
- ghi artifact dưới `out/analysis/` như `ohlc_model_summary.csv`, `ohlc_prediction_history.csv`, `ohlc_current_forecasts.csv`, `ohlc_case_studies.csv`, `ohlc_summary.json`

Đây vẫn là harness phân tích đầy đủ ở chế độ offline. Trong flow live, repo hiện có thêm một builder rút gọn `scripts/analysis/build_ohlc_next_session_report.py` để chọn model đang thắng theo từng mã và sync riêng file `ml_ohlc_next_session.csv` cho `T+1`.

### Forecast intraday cho phần còn lại của phiên

Nếu muốn có lớp forecast riêng cho run giữa phiên, repo có thêm builder:

```bash
PYTHONPATH=. venv/bin/python scripts/analysis/build_intraday_rest_of_session_report.py \
  --universe-csv out/universe.csv
```

Script này:

- kích hoạt khi `EngineRunAt` vẫn nằm trong một bucket intraday còn phần phiên chưa diễn ra (`AM_EARLY`, `AM_LATE`, `LUNCH_BREAK`, `PM_EARLY`, `PM_LATE`)
- fetch/cache intraday `5m` theo từng mã trong working universe cộng với `VNINDEX`
- train/chọn model riêng theo từng `ticker x SnapshotTimeBucket` trên các snapshot intraday lịch sử của chính mã đó cho phần còn lại của phiên
- feature layer intraday giữ thêm context gần kiểu `30m/60m` ngay trên chuỗi `5m` như return, range, và vị trí giá trong range ngắn; đây là lớp hourly-style gần hơn mà vẫn không cần mở nhánh tick riêng
- selection score không còn nhìn đối xứng thuần theo `close/range`; nó phạt riêng các trường hợp underpredict `high` và underestimate downside để giảm bias bỏ lỡ tail quan trọng
- current snapshot được re-label theo context `EngineRunAt`, để run giữa giờ nghỉ trưa không bị neo máy móc vào bucket của bar intraday cuối cùng
- dự báo `low / close còn lại / high` cho phần còn lại của phiên
- ghi file optional `out/analysis/ml_intraday_rest_of_session.csv`

### Forecast single-name timing cho vòng trade ngắn hạn

Nếu muốn có thêm một lớp ML thiên về kiểu quyết định “nếu chỉ được tập trung vốn vào một mã”, repo có thêm builder:

```bash
PYTHONPATH=. venv/bin/python scripts/analysis/build_single_name_timing_report.py \
  --universe-csv out/universe.csv
```

Script này:

- tái dùng feature daily của từng mã cộng với context `VNINDEX`
- train/chọn model riêng theo từng `ticker x horizon x variant` cho các horizon ngắn mặc định `T+3`, `T+5`, `T+10`, với các lát cắt `full_2y`, `recent_focus`, và `quarter_focus`
- dự báo `peak return`, `days-to-peak`, `drawdown`, và `close return` trong horizon đó
- dẫn xuất thêm `PredRewardRisk`, `PredTradeScore`, `PredNetEdgePct`, `PredCapitalEfficiencyPctPerDay`
- ghi live file `out/analysis/ml_single_name_timing.csv` và hai sidecar `ml_single_name_timing_model_metrics.csv`, `ml_single_name_timing_selected_models.csv`

### Báo cáo ML range ~2 năm giao dịch

Nếu muốn chuẩn hoá một report dễ dùng hơn trên cùng một cửa sổ gần `2 năm giao dịch`, có thêm script:

```bash
PYTHONPATH=. venv/bin/python scripts/analysis/build_range_forecast_report.py
```

Script này:

- fetch riêng một cache daily ~`800` ngày lịch cho `VNINDEX` và một rổ mã thanh khoản/đại diện
- đóng cứng snapshot cuối trong report theo dữ liệu fetch được, ví dụ `2026-03-20`
- train và chọn model riêng theo từng `ticker x horizon x variant` với hai biến thể:
  - `full_2y`: dùng toàn bộ cửa sổ ~2 năm
  - `recent_focus`: chỉ dùng lát cắt gần hơn bên trong cùng cửa sổ để ưu tiên trạng thái hiện tại
- tái dùng shared daily feature layer nên mỗi mã đều có thêm weekly-context chứ không chỉ nhìn chuỗi theo ngày thuần
- ghi thêm `ml_range_2y_model_metrics.csv` và `ml_range_2y_selected_models.csv` để xem model nào thắng theo từng mã/horizon
- xuất các artifact dễ đọc như `ml_range_2y_easy_view.csv`, `ml_range_2y_top_bottom.csv`, `ml_range_2y_fpt_hpg_comparison.csv`, `ml_range_2y_horizon_metrics.csv`, `ml_range_2y_summary.json`

### Báo cáo ML cycle forecast cho 1M..6M

Nếu muốn dự báo theo kiểu `nếu mua hôm nay thì trong tối đa 1-6 tháng, vòng lên tốt nhất kỳ vọng là bao nhiêu và thường mất bao lâu`, có thêm script:

```bash
PYTHONPATH=. venv/bin/python scripts/analysis/build_cycle_forecast_report.py
```

Script này:

- mặc định fetch `VN30` live và làm đầy cache daily đủ sâu
- dựng target cycle theo từng khung `1M..6M` với sell-delay mặc định `T+3`
- train và chấm riêng cho từng `ticker x horizon x variant` trên ba model `ridge`, `random_forest`, `hist_gbm`
- chọn config tốt nhất theo sai số dự báo `peak return`, `days to peak`, và `drawdown`
- ghi artifact dưới `out/analysis/` như:
  - `cycle_forecast_model_metrics.csv`
  - `cycle_forecast_best_configs.csv`
  - `cycle_forecast_current_selected.csv`
  - `cycle_forecast_ticker_matrix.csv`

`cycle_forecast_ticker_matrix.csv` là file dễ đọc nhất nếu muốn xem nhanh mỗi mã có `PredPeakRetPct`, `PredPeakDays`, và `PredDrawdownPct` trong từng khung `1M..6M`.

Quy trình dùng report này để tuyển chọn working universe:

1. Chạy screening rộng trên `VN30`, `VN100`, hoặc một rổ lớn hơn.
2. Đọc `cycle_forecast_best_horizon_by_ticker.csv` để xem mỗi mã hợp nhất với khung nào và model nào.
3. Đọc `cycle_forecast_ticker_matrix.csv` để xem đường cong kỳ vọng `1M..6M` của chính mã đó:
   - `PredPeakRetPct`: upside tuyệt đối kỳ vọng
   - `PredPeakDays`: số phiên thường mất tới đỉnh kỳ vọng
   - `PredDrawdownPct`: mức âm sâu nhất trong lúc cầm
4. Loại các mã có `PredDrawdownPct` quá sâu, error quá cao, hoặc business risk không phù hợp.
5. Chỉ sau bước đó mới promote mã vào `config/data_engine.yaml`:
   - `core_tickers`: rổ daily backbone chính
   - `preferred_tickers`: rổ tactical/thesis phụ; chỉ đi vào flow live khi đang có vị thế hoặc được promote sang core

Ví dụ chạy screening trên `VN100`:

```bash
venv/bin/python - <<'PY'
from pathlib import Path
from scripts.data_fetching.market_members import fetch_vn100_members
from scripts.analysis.build_cycle_forecast_report import run_report

run_report(
    tickers=sorted(fetch_vn100_members(timeout=30)),
    history_dir=Path("out/data"),
    output_dir=Path("out/analysis/vn100_cycle_forecast"),
    history_calendar_days=1100,
    horizon_months=[1, 2, 3, 4, 5, 6],
    holdout_dates=40,
    recent_focus_dates=252,
    quarter_focus_dates=63,
    sell_delay_days=3,
)
PY
```

### Replay ML với BCTT Vietstock

Nếu muốn thử thêm dữ liệu báo cáo tài chính quý từ trang `Vietstock BCTT`:

```bash
venv/bin/python scripts/analysis/evaluate_bctt_feature_lift.py \
  --case-tickers HPG FPT SSI VCB GAS VNM MBB CTG TCB MWG GVR REE DCM PLX \
  --top-k 5 \
  --min-train-dates 80 \
  --retrain-every 10
```

Script này:

- dùng Playwright lấy và cache bảng `KQKD`, `CĐKT`, `CSTC` từ `https://finance.vietstock.vn/<TICKER>/tai-chinh.htm?tab=BCTT`
- chuẩn hoá dữ liệu quý với độ trễ công bố bảo thủ để tránh leakage
- so sánh ba biến thể feature: `baseline`, `ratios_only`, `hybrid_growth`
- ghi artifact dưới `out/analysis/` như `ml_bctt_feature_summary.csv`, `ml_bctt_current_top_picks.csv`, `ml_bctt_case_studies.csv`, `ml_bctt_summary.json`
- lưu cache thô dưới `out/vietstock_bctt/`

Đây vẫn là harness offline để đo xem BCTT có cải thiện stock ranking hay không; nó chưa được nối vào flow live `tcbs -> codex -> orders`.

### Làm đầy cache BCTT Vietstock

Nếu muốn cache sẵn `BCTT` cho một universe lớn hơn thay vì fetch lẻ tẻ khi replay:

```bash
venv/bin/python scripts/data_fetching/refresh_vietstock_bctt_cache.py \
  --max-age-hours 720
```

Script này:

- lấy ticker từ `data/industry_map.csv` và `data/portfolios/portfolio.csv`
- dùng cache `out/vietstock_bctt/` nếu còn mới, chỉ fetch các mã thiếu hoặc quá cũ
- ghi summary xuống `out/analysis/vietstock_bctt_cache_summary.csv`

Hướng này phù hợp để dùng lâu dài vì dữ liệu quý thay đổi chậm; chạy cache refresh riêng sẽ ổn định hơn việc để từng harness tự fetch toàn bộ từ đầu.

### Replay macro factor sensitivity

Để xem từng mã đang nhạy với `dầu`, `vàng`, `USD`, `VIX`, `Nasdaq`, `S&P500`, `US10Y` ra sao:

```bash
venv/bin/python scripts/data_fetching/macro_factor_cache.py --max-age-hours 24
venv/bin/python scripts/analysis/evaluate_macro_factor_sensitivity.py \
  --case-tickers HPG FPT SSI VCB GAS PLX GVR MWG MBB TCB \
  --top-factors 3 \
  --no-refresh-factors
```

Script này:

- cache factor vĩ mô dưới `out/macro_factors/` từ `FRED` và `Stooq` theo cấu hình `config/macro_factors.yaml`
- đo `rolling correlation`, `beta`, và phản ứng lịch sử của từng mã khi factor có phiên shock mạnh
- ghi artifact dưới `out/analysis/` như `macro_factor_current_regime.csv`, `macro_factor_sensitivity.csv`, `macro_factor_case_studies.csv`, `macro_factor_ticker_summary.csv`, `macro_factor_summary.json`

Đây là lớp offline để trả lời câu hỏi kiểu `mã này hiện đang ăn theo dầu hay đang nhạy với risk-off`, chưa được merge vào prompt hoặc engine live.

### Lấy danh mục + lệnh khớp (TCBS, Playwright)

Thay cho server HTTP, repo cung cấp scraper Playwright để đăng nhập TCBS (qua Chrome) và trích xuất một danh mục duy nhất.

Chuẩn bị:
- Tạo file `.env` ở repo root:

```bash
TCBS_USERNAME=you-username
TCBS_PASSWORD=your-password-here
```

Chạy lần đầu (headful để xác nhận thiết bị nếu TCBS yêu cầu):

```bash
./broker.sh tcbs_login
./broker.sh tcbs
```

Các lần sau có thể tiếp tục dùng `./broker.sh tcbs` (luôn chạy Chrome ở chế độ headful).

Mặc định script sẽ:
- Dùng Chrome (Playwright channel `chrome`) với profile persistent tại `.playwright/tcbs-user-data/default`.
- Tự đăng xuất nếu đang có session TCBS cũ, sau đó đăng nhập lại bằng `TCBS_USERNAME`/`TCBS_PASSWORD`.
- `./broker.sh tcbs` giữ flow scrape như cũ, không cộng thêm nhịp chờ 20 giây sau login.
- Nếu chỉ muốn làm bước login/MFA trước, chạy `./broker.sh tcbs_login`; lệnh này dùng cùng persistent profile với `./broker.sh tcbs`, chờ thêm 20 giây sau khi bấm login rồi thoát. Có thể override bằng `TCBS_POST_LOGIN_WAIT_MS` hoặc `--post-login-wait-ms`.
- Ghi đè `data/portfolios/portfolio.csv` với cột `Ticker,Quantity,AvgPrice`.
- Dọn sạch thư mục `codex_universe/` (hoặc `CODEX_DIR`) trước khi lấy danh mục để tránh sót file cũ.

Đặt lệnh TCBS từ file Codex:
- Chạy `./broker.sh tcbs_orders` để đọc `codex_universe/orders.csv` và đặt lệnh qua Playwright.
- Script sẽ snapshot file input sang `out/orders_snapshots/` trước khi đặt lệnh để tiện đối chiếu nếu run bị gián đoạn.
- Flow đặt lệnh dùng profile Playwright tạm riêng dưới `.playwright/tcbs-orders-user-data/`, tự chặn popup xin notification, và chờ cửa sổ MFA/session transition thay vì giả định form lệnh luôn xuất hiện ngay.
- `success/total` phản ánh số lệnh bấm **Xác nhận** thành công.
- Kết quả xác nhận click nút "Xác nhận" được ghi vào `out/tcbs_confirmed_orders.csv` (schema `Ticker,Side,Quantity,Price`).
- Mọi error dialog trong TCBS được ghi vào `out/tcbs_error_orders.csv` (schema `Ticker,Side,Quantity,Price,Reason`).

### Codex scripted chat

`./broker.sh codex` thực hiện đúng quy trình "test broker-gpt với Codex" bằng exec/resume (không TTY):

1. Trước khi launch Codex, wrapper sẽ re-sync thư mục `codex_universe/` từ bundle live mới nhất trong `out/`: `universe.csv`, `market_summary.json`, `sector_summary.csv`, `ml_range_predictions_*.csv`, `ml_cycle_forecast_*.csv`, `ticker_playbook_best_configs.csv`, `ml_ohlc_next_session.csv`, và nếu snapshot vẫn còn cửa giao dịch trong ngày thì có thể có thêm `ml_intraday_rest_of_session.csv`; ngoài ra có thể có thêm `ml_single_name_timing.csv`, `ml_entry_ladder_eval.csv`, `human_notes.md`, `strategy_buckets.csv`, và thư mục `research/`. Nếu thiếu source file ở `out/`, wrapper sẽ fail-fast thay vì chạy trên bundle cũ hoặc bundle rỗng.
1.1. Sau khi sync xong, wrapper build thêm `codex_universe/bundle_manifest.json`; đây là index kỹ thuật để prompt live biết file nào bắt buộc, file nào optional, schema tối thiểu, cùng `Summary`/`UsageNotes` canonical của từng file, và working universe hiện tại của bundle.
2. Đọc prompt đầu tiên **trực tiếp** từ `prompts/PROMPT.txt` (không hard-code) và chạy `codex exec` + `resume` với working dir chính là `codex_universe/`.
3. Trong mỗi lần sync bundle sang `codex_universe/` (gồm cả `./broker.sh codex` và `./broker.sh research`), wrapper sẽ **tạo/ghi đè** file `codex_universe/total_capital_kVND.txt` (chỉ 1 số, đơn vị kVND) dựa trên `CODEX_TOTAL_CAPITAL_KVND` (ưu tiên) hoặc parse từ `CODEX_BUDGET_TEXT` (ví dụ `5 tỉ`).
4. Ở bước đầu tiên của prompt, Codex sẽ tự kiểm tra các file/cột được mô tả trong prompt có khớp artifact thực tế hay không; nếu có mismatch thì nêu lỗi ngay trong câu trả lời của bước đó nhưng vẫn tiếp tục flow bình thường.
5. Sau prompt, script gửi `Tiếp tục` rồi gửi thêm `CODEX_CONTINUE_MESSAGE` liên tục cho đến khi thư mục `codex_universe/` xuất hiện file hoàn tất `DONE.md` (hoặc tên tuỳ chỉnh qua `CODEX_DONE_FILE`). Wrapper chỉ kiểm tra file này ở ranh giới giữa các vòng `resume`, không cắt ngang một lượt `resume` đang chạy bình thường.
6. Nếu một lượt `codex exec`/`resume` bị im lặng quá `CODEX_IDLE_TIMEOUT_SECONDS`, wrapper sẽ kill tiến trình CLI đó và retry bằng chính thread hiện có với `resume` + `CODEX_CONTINUE_MESSAGE` thay vì treo vô hạn.
7. Wrapper sẽ fail-fast nếu vượt timeout / số lần nhắc tối đa (nếu cấu hình).
8. Khi phát hiện file hoàn tất giữa các vòng `resume`, script chờ `CODEX_GRACE_AFTER_CSV` giây để Codex flush nốt artifact, fail-fast nếu chưa có `orders.csv`, copy `orders.csv`, `DONE.md`, và log Codex tương ứng vào `archives/codex_runs/<YYYYMMDD_HHMMSS>/`. Ngoài local staging này, wrapper còn copy nguyên thư mục `codex_universe/` sang `codex_universe_history/<TCBS account>/<YYYY>/<timestamp>/` trong repo chính để giữ snapshot đầy đủ của bundle live phục vụ việc đánh giá lại. Nếu `CODEX_ORDER_HISTORY_ENABLED=1`, wrapper tiếp tục clone/pull repo lịch sử lệnh riêng, chép artifact sang cấu trúc `<TCBS account>/<YYYY>/<timestamp>/`, commit và push ngay ở repo đó trước khi in toàn bộ nội dung file ra stdout.

Ngay khi lấy được `thread_id`, wrapper cũng ghi sẵn `out/codex/resume_last_codex.sh` để có thể `resume` thủ công ngay trong lúc phiên còn đang chạy; không cần chờ tới cuối khi `DONE.md` đã xuất hiện.

Snapshot trong thư mục `codex_universe/` được cập nhật tự động sau mỗi lần chạy `./broker.sh tcbs`, và `./broker.sh codex` cũng re-sync lại bundle này ngay trước khi launch Codex để tránh chạy trên snapshot rỗng hoặc snapshot cũ. Bundle gồm `bundle_manifest.json`, `universe.csv`, `market_summary.json`, `sector_summary.csv`, range forecast, cycle forecast, playbook, `ml_ohlc_next_session.csv`, và nếu snapshot vẫn còn cửa giao dịch trong ngày thì thêm `ml_intraday_rest_of_session.csv`; nếu builder timing mới chạy thành công thì cũng có thêm `ml_single_name_timing.csv`; nếu builder ladder mới chạy thành công thì có thêm `ml_entry_ladder_eval.csv`; nếu repo root có `strategy_buckets.csv` thì wrapper sẽ synthesize file này sang `codex_universe/strategy_buckets.csv` bằng cách chỉ giữ các mã còn nằm trong active ticker filter, rồi mới tự thêm `exit_only` cho các mã đang có trong portfolio nhưng vẫn còn nằm trong filter mà chưa có trong source; nếu repo root có `research/` thì wrapper sẽ copy nguyên thư mục này sang `codex_universe/research/`. Thư mục `research/tickers/` cũng được prune theo working universe mỗi lần rebuild để không giữ note cũ của các mã đã bị loại khỏi core. Các thư mục này không được commit (đã `gitignore`), và wrapper sẽ tự tạo nếu chưa tồn tại.

Nếu repo có thư mục `research/`, `./broker.sh tcbs` và `./broker.sh research` sẽ copy nguyên thư mục đó sang `codex_universe/research/`. Contract ở đây không phải note tay tự do mà là artifact research có cấu trúc:

- `research/manifest.json` là index chính.
- `profile.md` mô tả archetype/bản chất mã.
- `weekly/*.md` giữ thesis tuần hiện hành.
- `daily/*.md` giữ tactical note gần nhất.
- `state.json` là summary machine-readable để prompt daily đọc nhanh.

Prompt live sẽ đọc `bundle_manifest.json` trước, rồi đọc `research/manifest.json` và các artifact liên quan của từng mã trong working universe trước khi đối chiếu với dữ liệu live.

Nếu repo có file `human_notes.md`, `./broker.sh tcbs` cũng sẽ copy file này sang `codex_universe/human_notes.md`. Prompt live sẽ đọc file đó như chỉ thị/giả định do người dùng cung cấp trước khi phân tích vốn, tin tức và market regime.
`human_notes.md` là file văn xuôi ở repo root để người dùng ghi note ngắn kiểu facts/assumptions như target giá, thesis riêng, hoặc market color. Khi builder research parse được target/theory theo mã từ file này, overlay đó sẽ được đẩy xuống `research/state.json` như một strategic bias có cấu trúc thay vì chỉ nằm ở prompt. Không nên lặp lại working universe đã có sẵn trong `universe.csv`, và cũng không nên viết thành policy dài hay mệnh lệnh điều khiển model.

Nếu repo có file `strategy_buckets.csv`, `./broker.sh tcbs` sẽ build `codex_universe/strategy_buckets.csv` từ source này. Source ở repo root chỉ nên giữ các mã bạn muốn control bucket một cách chủ động; wrapper sẽ chỉ giữ các mã còn nằm trong active ticker filter, rồi mới tự append các mã đang có trong `data/portfolios/portfolio.csv` nhưng vẫn còn nằm trong filter mà chưa có trong source thành bucket `exit_only` với `AllowNewBuy=0`, `AllowAvgDown=0`, `TargetState=exit_all`. Prompt live sẽ đọc file đã synthesize đó như một lớp chỉ thị có cấu trúc, tách biệt với `human_notes.md`.

Lớp note theo mã giờ được gom về một contract duy nhất là `research/` với ba nhịp `profile`, `weekly`, `daily`. Điều này tránh trạng thái hai nguồn thesis song song, đồng thời giúp prompt live chỉ phải đọc một nguồn có cấu trúc.

Prompt live mặc định hiểu bộ lệnh là cho cửa giao dịch gần nhất có thể thực thi; nếu snapshot vẫn còn cửa giao dịch trong ngày thì bộ lệnh là cho phần còn lại của ngày giao dịch hiện tại.

Prompt cũng hard-code giả định phí giao dịch `0.3%` mỗi chiều khớp lệnh, nên Codex phải đánh giá reward/risk theo lợi nhuận sau phí thay vì gross return thuần.

Ghi log: Tất cả log và sự kiện Codex được ghi vào `out/codex/` (file `.log` và `.jsonl` theo timestamp). Trước khi chạy, wrapper re-sync bundle live vào `codex_universe/`; bước sync này cũng materialize `total_capital_kVND.txt` để contract của bundle luôn nhất quán, kể cả khi chỉ chạy `./broker.sh research`. Các file output hỗ trợ do Codex tạo sẽ bị xoá ở lần chạy kế tiếp. Riêng `orders.csv`, `DONE.md`, và log của đúng phiên tạo lệnh sẽ được copy sang `archives/codex_runs/<timestamp>/` như local staging; đồng thời, một snapshot đầy đủ của `codex_universe/` sẽ được copy sang `codex_universe_history/<TCBS account>/<YYYY>/<timestamp>/` trong repo chính để phục vụ hậu kiểm bundle. Source-of-truth để lưu lịch sử lệnh dài hạn vẫn là repo lịch sử lệnh riêng nếu bật `CODEX_ORDER_HISTORY_ENABLED`.

Watchdog completion: khi cả `orders.csv` và `DONE.md` đã xuất hiện trong lúc một lượt `codex exec/resume` vẫn còn chạy, wrapper sẽ cho phép thêm một khoảng `grace_after_csv` ngắn rồi chủ động terminate lượt đó thay vì chờ vô hạn cho tới khi process tự thoát. Mục tiêu là cắt phần hậu kỳ dư thừa sau khi output đã hoàn chỉnh.

Tuỳ chỉnh nhanh qua biến môi trường:

| Biến | Mặc định | Giải thích |
| ---- | -------- | ---------- |
| `INDUSTRY_TICKER_FILTER` | *(trống)* | Danh sách mã (phân tách bằng dấu phẩy/khoảng trắng) để hard-override working universe. Nếu set, biến này override `universe.core_tickers` và áp thẳng cho `universe.csv`, `positions.csv`, và các bundle live downstream. |
| `CODEX_DIR` | `codex_universe` | Thư mục chạy Codex (phải chứa đúng 1 file `universe.csv`). |
| `CODEX_PROMPT_FILE` | `prompts/PROMPT.txt` | Prompt đầu tiên. |
| `CODEX_BUDGET_TEXT` | `5 tỉ` | Tổng vốn có thể dùng (gồm vốn đã mua + phần còn có thể giải ngân) dùng để **tạo** `total_capital_kVND.txt` nếu `CODEX_TOTAL_CAPITAL_KVND` không được set. |
| `CODEX_TOTAL_CAPITAL_KVND` | *(trống)* | Nếu set, wrapper sẽ ghi `total_capital_kVND.txt` bằng đúng số này (kVND). Nếu trống, wrapper sẽ parse từ `CODEX_BUDGET_TEXT`. |
| `CODEX_OUTPUT_CSV` | `orders.csv` | Tên file Codex phải ghi (mặc định `./orders.csv` trong `codex_universe/`). |
| `CODEX_DONE_FILE` | `DONE.md` | File marker completion; wrapper chỉ coi run đã xong khi file này xuất hiện. |
| `CODEX_BIN` | `codex` | Đường dẫn binary Codex. |
| `CODEX_MODEL` | `gpt-5.4` | Model Codex dùng cho phiên chat scripted. |
| `CODEX_REASONING` | `xhigh` | Reasoning mode truyền vào Codex qua `--config model_reasoning_effort="..."`. |
| `CODEX_IDLE_TIMEOUT_SECONDS` | `900` | Nếu một lượt `codex exec`/`resume` không có stdout/stderr quá ngưỡng này, wrapper sẽ kill lượt đó và thử `resume` lại cùng thread. |
| `CODEX_TIMEOUT_SECONDS` / `CODEX_MAX_WALL_SECONDS` | `7200` | Timeout tối đa (giây) cho wrapper Codex; hết thời gian mà chưa có file completion sẽ fail. |
| `CODEX_MAX_CONTINUES` | `200` | Số lần gửi `CODEX_CONTINUE_MESSAGE` tối đa; `0` = không giới hạn (vẫn bị chặn bởi timeout). |
| `CODEX_GRACE_AFTER_CSV` | `8` | Thời gian đợi sau khi wrapper phát hiện file completion ở ranh giới giữa các vòng `resume`, trước khi finalize artifact. |
| `CODEX_CONTINUE_MESSAGE` | `Tiếp tục.` | Chuỗi lặp lại cho đến khi Codex tạo xong file. |
| `CODEX_ARCHIVE_ROOT` | `archives/codex_runs` | Thư mục lưu archive mỗi run Codex sau khi tạo `DONE.md`. |
| `CODEX_ARCHIVE_ENABLED` | `1` | Bật/tắt bước archive `orders.csv` + log Codex tương ứng. |
| `CODEX_ARCHIVE_GIT_COMMIT` | `0` | Bật/tắt bước `git commit` riêng cho thư mục archive local trong repo chính. |
| `CODEX_TRACKED_SNAPSHOT_ENABLED` | `1` | Bật/tắt bước copy nguyên `codex_universe/` sang snapshot root trong repo chính sau mỗi run thành công. |
| `CODEX_TRACKED_SNAPSHOT_ROOT` | `codex_universe_history` | Root repo-relative cho snapshot đầy đủ của `codex_universe/`, tổ chức theo `<TCBS account>/<YYYY>/<timestamp>/`. |
| `CODEX_ORDER_HISTORY_ENABLED` | `1` | Bật/tắt bước sync `orders.csv` + `DONE.md` + log sang repo lịch sử lệnh riêng rồi commit/push ngay. |
| `CODEX_ORDER_HISTORY_REPO` | `minhhai2209/tcbs-orders-history` | Repo GitHub dùng làm source-of-truth cho lịch sử lệnh. |
| `CODEX_ORDER_HISTORY_CLONE_DIR` | `../tcbs-orders-history` | Thư mục clone local của repo lịch sử lệnh; path tương đối được hiểu từ repo root `broker-gpt-3`. |
| `CODEX_ORDER_HISTORY_GH_BIN` | `gh` | Binary GitHub CLI dùng cho lần clone đầu tiên. |
| `TCBS_ACCOUNT_SLUG` | `TCBS_USERNAME` | Tên account dùng để xếp artifact theo nhiều tài khoản trong repo lịch sử lệnh. |

Bạn có thể chạy riêng bước này để kiểm tra:

```bash
./broker.sh codex
```

### Kiểm thử

```bash
./broker.sh tests
```

Test bao gồm:
- Bảo đảm engine sinh đầy đủ output khi dùng nguồn dữ liệu giả lập.
- Xác thực bộ phân tích bảng TCBS tạo đúng schema `Ticker,Quantity,AvgPrice` từ dữ liệu giả lập.
- Smoke test các hàm chọn market gate / tổng hợp replay cho harness deterministic.
- Smoke test tổng hợp metrics/chọn model cho harness ML baseline.
- Smoke test tổng hợp metrics/chọn model cho harness VNINDEX ML.

## Output chính

Ngoài hai file output cũ (`out/universe.csv`, `out/positions.csv`), engine hiện còn ghi:
- `out/market_summary.json` để tóm tắt range, breadth, co-movement, dispersion, drawdown/rebound của VNINDEX.
- `out/sector_summary.csv` để tóm tắt breadth/return/flow/thanh khoản theo ngành.
- Các cột mới trong `out/universe.csv` như `Ret20dVsIndex`, `Ret60dVsIndex`, `Ret20dVsSector`, `Ret60dVsSector`, `CoMoveWithIndex20Pct`, `Corr20_Index`, `Beta20_Index`, `RelStrength20Rank`, `RelStrength60Rank`, `SectorBreadthAboveSMA20Pct`, `SectorBreadthAboveSMA50Pct`, `SectorBreadthPositive5dPct`, `SectorADTVRank`, `VNINDEX_ATR14PctRank`.
- `out/analysis/ml_range_predictions_full_2y.csv` để lưu forecast ML giá/range của biến thể `full_2y`.
- `out/analysis/ml_range_predictions_recent_focus.csv` để lưu forecast ML giá/range của biến thể `recent_focus`.
- `out/analysis/ml_ohlc_next_session.csv` để lưu forecast OHLC `T+1` tốt nhất theo từng mã trong working universe.
- `out/analysis/ml_intraday_rest_of_session.csv` để lưu forecast intraday optional cho phần còn lại của phiên khi snapshot vẫn còn cửa giao dịch trong ngày.
- `out/analysis/ml_single_name_timing.csv` để lưu forecast timing ngắn hạn theo từng mã cho các vòng trade tập trung.
- `out/analysis/ml_entry_ladder_eval.csv` để lưu bảng chấm điểm `Ticker x LimitPrice` cho từng nấc BUY hợp lệ theo trade-off giữa độ sâu giá, upside và khả năng khớp ước lượng; nếu đủ lịch sử thì `FillScoreT1/T5/T10` được suy ra từ model per-ticker thay vì heuristic thuần khoảng cách. `EntryScore` hiện dùng fill-score như trọng số trực tiếp cho edge, để các lệnh quá sâu nhưng khó khớp không tự động thắng điểm.
- `out/analysis/` để lưu replay offline của nhiều market/ticker algorithms; thư mục này nằm dưới `out/` nên không được commit.
- `out/analysis/ml_*.csv|json` để lưu baseline machine-learning walk-forward trên cùng bộ cache daily.
- `out/analysis/vnindex_ml_*.csv|json` để lưu baseline machine-learning riêng cho chỉ số `VNINDEX`.
- `out/analysis/ohlc_*.csv|json` để lưu baseline machine-learning dự báo `OHLC/range` trực tiếp theo từng horizon `T+1..T+10`.
- `out/analysis/ml_range_2y_*.csv|json` để lưu report ML được tune riêng theo từng mã trên cửa sổ gần `2 năm giao dịch`, gồm easy-view `T+1/T+5/T+10`, top/bottom lists, model metrics, và so sánh `FPT` vs `HPG`.
- `out/analysis/ml_bctt_*.csv|json` để lưu so sánh baseline ML với các biến thể feature lấy từ BCTT Vietstock.
- `out/analysis/vietstock_bctt_cache_summary.csv` để theo dõi coverage/độ mới của cache BCTT.
- `out/analysis/macro_factor_*.csv|json` để lưu report sensitivity của từng mã với vàng/dầu/USD/VIX/Nasdaq/SP500/US10Y.
- `out/vietstock_bctt/` để cache dữ liệu quý thô từ trang `Vietstock BCTT`.
- `out/macro_factors/` để cache chuỗi thời gian factor vĩ mô.

| File/Thư mục | Ý nghĩa |
| ---- | ------- |
| `out/universe.csv` | Bản hợp nhất cho prompt: trường kỹ thuật & biến động (Last/Ref/ChangePct/Vol20Pct/Vol60Pct/Beta60_Index/Corr60_Index/SMA/EMA/RSI/ATR/ATR14Pct/MACD/Z20/Ret/Pos52wPct, IsVN30), thanh khoản & execution (ADTV20_shares/IntradayVol_shares/IntradayValue_kVND/IntradayPctADV20/ForeignFlowDate/ProprietaryFlowDate/NetBuySellForeign_shares_{1d,5d,20d}/NetBuySellForeign_kVND_{1d,5d,20d}/NetBuySellProprietary_shares_{1d,5d,20d}/NetBuySellProprietary_kVND_{1d,5d,20d}/ForeignRoomRemaining_shares/ForeignHoldingPct/ADTV20Rank/ADTV20PctRank/Ceil/Floor/TickSize/LotSize/SlippageOneTickPct/FloorValid/CeilValid/ValidBid1/ValidAsk1/TicksToFloor/TicksToCeil), chỉ tiêu cơ bản (PE_fwd/PB/ROE), grid (DistRefPct/DistSMA20Pct/DistSMA50Pct, GridBelow/Above T1–T3), ATR sizing (OneLotATR_kVND) + 52w range, các cột khách quan theo thị trường/ngành (`CoMoveWithIndex20Pct`, `Corr20_Index`, `Beta20_Index`, `Ret20dVsIndex`, `Ret60dVsIndex`, `Ret20dVsSector`, `Ret60dVsSector`, `RelStrength20Rank`, `RelStrength60Rank`, `SectorBreadthAboveSMA20Pct`, `SectorBreadthAboveSMA50Pct`, `SectorBreadthPositive5dPct`, `SectorADTVRank`, `VNINDEX_ATR14PctRank`), và toàn bộ cột vị thế `EngineRunAt`, `PositionQuantity`, `PositionPctADV20`, `PositionAvgPrice`, `PositionMarketValue_kVND`, `PositionWeightPct`, `EnginePortfolioMarketValue_kVND`, `PositionCostBasis_kVND`, `PositionUnrealized_kVND`, `PositionATR_kVND`, `PositionPNLPct`, kèm thêm `SectorWeightPct` và `BetaContribution`. Dòng đầu tiên luôn là VNINDEX để giữ benchmark sẵn có trong prompt. |
| `out/positions.csv` | Ảnh chụp vị thế thô: mỗi dòng gồm `Ticker,Quantity,AvgPrice,Last,MarketValue_kVND,CostBasis_kVND,Unrealized_kVND,PNLPct` giúp đối chiếu holdings độc lập với prompt chính. |
| `out/analysis/ml_range_predictions_full_2y.csv` | Forecast ML giá/range cho từng mã và từng horizon `T+N` của biến thể `full_2y`, gồm `Base`, `Low-Mid-High`, return dự báo, `RecentFocusWeight`, `Full2YWeight`, và metric replay `*MAEPct`, `*DirHitPct`. |
| `out/analysis/ml_range_predictions_recent_focus.csv` | Forecast ML giá/range cho từng mã và từng horizon `T+N` của biến thể `recent_focus`, gồm cùng schema với file `full_2y`. |
| `out/analysis/ml_range_2y_model_metrics.csv` | Bảng chấm toàn bộ candidate model cho từng `ticker x horizon x variant` của range report, gồm `SelectionScore`, `CloseMAEPct`, `RangeMAEPct`, `CloseDirHitPct`. |
| `out/analysis/ml_range_2y_selected_models.csv` | Model được chọn cuối cùng cho từng `ticker x horizon x variant` trong range report live/offline. |
| `out/analysis/ml_ohlc_next_session.csv` | Forecast OHLC `T+1` tốt nhất theo từng mã trong working universe, gồm `Base`, `ForecastDate`, model được chọn, replay metrics, và `ForecastOpen/High/Low/Close` cho phiên kế tiếp. |
| `out/analysis/ml_intraday_rest_of_session.csv` | Forecast intraday optional cho phần còn lại của phiên hiện tại, gồm `Base`, `Low/Mid/High`, `SnapshotTimeBucket` và replay metrics. File chỉ tồn tại khi snapshot vẫn còn cửa giao dịch trong ngày. |
| `out/analysis/ml_single_name_timing.csv` | Forecast timing ngắn hạn theo từng `ticker x horizon`, gồm `PredPeakRetPct`, `PredPeakDay`, `PredDrawdownPct`, `PredCloseRetPct` và các score dẫn xuất như `PredRewardRisk`, `PredTradeScore`, `PredNetEdgePct`, `PredCapitalEfficiencyPctPerDay` để đánh giá một vòng trade tập trung. |
| `out/analysis/ml_entry_ladder_eval.csv` | Bảng `ticker x LimitPrice`, re-anchor edge từ `Base/Last` sang từng nấc BUY hợp lệ; `FillScoreT1/T5/T10` ưu tiên xác suất chạm giá học từ lịch sử riêng từng mã/horizon, còn `EntryScore` dùng fill-score như trọng số trực tiếp để tránh thiên lệch có hệ thống sang các lệnh BUY quá sâu. |
| `out/tcbs_confirmed_orders.csv` | Danh sách lệnh đã bấm nút xác nhận trong TCBS (schema `Ticker,Side,Quantity,Price`). |
| `out/tcbs_error_orders.csv` | Danh sách lệnh bị TCBS báo lỗi (schema `Ticker,Side,Quantity,Price,Reason`). |

## GitHub Actions

Hiện tại GitHub Action đã được tạm thời gỡ khỏi repo. Hãy chạy pipeline thủ công trên máy local bằng `./broker.sh tcbs` rồi `./broker.sh codex`. Khi cần bật lại, thêm lại workflow vào `.github/workflows/` (xem lịch sử commit trước đó để tham khảo cấu hình).

## Hỏi nhanh

**Có cần sửa danh mục thủ công?** — Có. Mỗi tài khoản là một thư mục trong `data/portfolios/` chứa `portfolio.csv`. Engine chỉ đọc và ghi báo cáo, không can thiệp vào file gốc.

**Muốn thêm chỉ báo mới?** — Bổ sung vào `scripts/indicators/` hoặc tính trực tiếp trong `scripts/engine/data_engine.py`, sau đó khai báo trong `config/data_engine.yaml` nếu cần tham số.

**Có còn chính sách/overlay?** — Không. Engine không sinh lệnh nên mọi cấu hình policy trước đây đã bị loại bỏ.

## Prompt gợi ý cho ChatGPT

- Prompt chuẩn: mở trực tiếp `prompts/PROMPT.txt` (plain text, không cần lệnh hỗ trợ).
- Ghi chú: Prompt chuẩn nên giữ vai trò là khung policy + workflow + hard constraints; phần execution vi mô nên ưu tiên đẩy xuống `bundle_manifest.json`, `research/state.json`, và các artifact ML để Codex còn room hòa giải trong bước ra lệnh cuối.

### Quy tắc HOSE để tính giá/khối lượng hợp lệ

Áp dụng đúng quy chế HOSE (QĐ 352/QĐ-SGDHCM, hiệu lực 05/07/2021) để tránh bước giá không hợp lệ:

- Đơn vị yết giá (tick) cổ phiếu/CCQ đóng:
  - < 10.000 VND: bước 10 VND
  - 10.000 – 49.950 VND: bước 50 VND
  - ≥ 50.000 VND: bước 100 VND
- ETF và chứng quyền: bước 10 VND cho mọi mức giá.
- Lô chẵn: bội số 100 cổ phiếu; khối lượng tối đa mỗi lệnh: 500.000 cổ.
- Biên độ giá trong ngày HOSE: ±7% so với giá tham chiếu.
- Làm tròn giá trần/sàn theo quy chế: trần làm tròn xuống, sàn làm tròn lên theo đúng đơn vị yết giá.

Lưu ý về giá tham chiếu: các cột `Ref/Ceil/Floor` trong `universe.csv` được tính từ snapshot kỹ thuật (mặc định dùng giá đóng cửa gần nhất). Khi sàn điều chỉnh tham chiếu do cổ tức/CP thưởng/gộp tách…, hãy cung cấp `data/reference_overrides.csv` và khai báo `data.reference_overrides` như trên để engine dùng đúng giá tham chiếu của sàn.

Gợi ý kiểm tra nhanh (giá báo theo nghìn đồng):
- `p_vnd = round(LimitPrice * 1000)`, chọn `tick` theo bảng trên tại mức `p_vnd`.
- Hợp lệ khi `(p_vnd % tick == 0)` và `floor_to_tick(ref*1.07) ≥ p_vnd ≥ ceil_to_tick(ref*0.93)`; `ref` là giá tham chiếu VND.
- `Quantity` là bội số 100 và ≤ 500.000.
