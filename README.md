# Investment Notebook

> Repo này là notebook tương tác cho forecast thị trường/mã Việt Nam. Nó không còn flow đặt lệnh, không còn TCBS automation, không còn ngân sách/danh mục mặc định, và không đưa khuyến nghị giao dịch.

## Repo này giữ gì

- Engine lấy dữ liệu và dựng snapshot: `scripts/engine/data_engine.py`
- Các fetcher/indicator hỗ trợ: `scripts/data_fetching/`, `scripts/indicators/`
- Các report/harness phân tích và ML: `scripts/analysis/`
- Research bundle có cấu trúc để Codex đọc trực tiếp: `scripts/research/build_research_bundle.py`
- Tool refresh universe map theo rổ live: `scripts/tools/refresh_industry_map.py`

## Repo này đã bỏ gì khỏi workflow mặc định

- TCBS login / scrape danh mục qua browser
- TCBS order placement
- `codex_universe/`, `orders.csv`, `DONE.md`, prompt bundle, archive lịch sử lệnh
- ngân sách tham chiếu, position sizing, order ladder, và khuyến nghị mua/bán
- portfolio merge mặc định
- strategy buckets và overlay phục vụ order workflow cũ

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Scope mặc định đã được pin vào `VIC` trong `config/data_engine.yaml`; không cần refresh map nếu chỉ phân tích VIC.

```bash
./broker.sh prepare_default
```

Nếu chỉ muốn rebuild artifact từ map hiện có:

```bash
./broker.sh prepare
```

Active forecast tasks:

```bash
./broker.sh select_vic_model
```

- `select_vic_model`: active model duy nhất cho VIC. Selector thử các candidate giá/chiều hướng, feature-set thường và feature-engineered, rồi giữ đúng một winner theo holdout 5 phiên gần nhất cộng walkback trước holdout.
- Winner hiện tại là `baseline_ohlc / logit_direction`: logistic classifier dự đoán chiều `TargetCloseRetPct > 0` cho T+1..T+5.
- Các builder/model cũ như `ohlc`, `intraday`, `eval_vic_index_expiry`, `eval_curated_intraday_model`, `range`, `cycle`, `timing`, `sequence_dl` chỉ là legacy/diagnostic khi gọi tay, không phải output mặc định hay model publish chính.

Ví dụ prompt:

- `Nếu artifact chưa có hoặc stale thì tự chạy và chờ xong rồi mới phân tích.`
- `Chỉ dự báo VIC bằng snapshot mới nhất; không đưa khuyến nghị mua bán.`
- `Không dùng ngân sách, không tính size, không đọc danh mục.`
- `Tách phần Model predict và External synthesis. Model predict chỉ lấy từ artifact/model trong repo; External synthesis nếu có tra cứu web/news thì ghi rõ nguồn và không trộn vào số forecast.`
- `Trả lời bằng single active model: direction T+1..T+5, xác suất up, holdout hit, walkback hit.`
- `Nếu legacy diagnostics mâu thuẫn với active model thì gọi rõ diagnostics conflict và không dùng chúng thay forecast chính.`
- `Phải nói rõ T+N là N phiên giao dịch sau snapshot.`
- `Nếu cần chạy batch thì tự chạy tuần tự xong rồi mới trả lời, không trả lời giữa chừng rằng vẫn đang đợi artifact.`

## Wrapper Commands

```bash
./broker.sh tests
./broker.sh engine
./broker.sh prepare
./broker.sh research
./broker.sh map
./broker.sh refresh_vic_map
./broker.sh sync_artifacts
./broker.sh prepare_default
./broker.sh refresh_macro
./broker.sh eval_macro --no-refresh-factors --case-tickers VIC
./broker.sh eval_macro_lift --case-tickers VIC
./broker.sh select_vic_model
```

Các harness/builder legacy vẫn còn để audit khi cần:

```bash
./broker.sh candidates auto
./broker.sh deep VIC
./broker.sh range
./broker.sh cycle
./broker.sh playbook
./broker.sh timing
./broker.sh entry_ladder
./broker.sh position_ml --ticker VIC --quantity 20000 --avg-price 214.53 --current-price <live_price>
./broker.sh sequence_dl
./broker.sh eval_deterministic
./broker.sh eval_ml
./broker.sh eval_vnindex
./broker.sh eval_ohlc
./broker.sh eval_macro
./broker.sh eval_bctt
./broker.sh eval_vic_index_expiry --models hist_gbm
./broker.sh ohlc
./broker.sh intraday
./broker.sh eval_intraday_features
./broker.sh eval_daily_features
./broker.sh eval_curated_intraday_model
```

## Output chính

- `out/universe.csv`: snapshot hợp nhất để đọc tape/context
- `out/positions.csv`: được ghi rỗng theo mặc định để tương thích schema cũ; config mặc định không đọc danh mục
- `out/market_summary.json`: breadth, range, co-movement ở cấp thị trường
- `out/sector_summary.csv`: breadth/relative strength ở cấp ngành
- `out/analysis/`: các report ML và evaluation
- `out/analysis/vic_single_model_current.csv`: output active duy nhất cho VIC theo selector last-5 holdout; có thể là model `price` hoặc `direction`
- `out/analysis/vic_single_model_holdout.csv`: backtest 5 phiên gần nhất, train bằng dữ liệu trước holdout
- `out/analysis/vic_single_model_walkback.csv`: walkback trước holdout
- `out/analysis/vic_single_model_candidates.csv`: bảng candidate/feature engineering để audit vì sao winner được chọn
- `out/analysis/vic_single_model_feature_engineering.csv`: bảng giữ lại feature-set và engineered features để tái dùng/audit sau này
- `out/analysis/ml_ohlc_*.csv`, `out/analysis/ml_intraday_*.csv`, `out/analysis/vic_index_expiry_*.csv`: legacy diagnostics, không phải model publish chính
- `out/analysis/macro_factor_*.csv`: độ nhạy/correlation với dầu, vàng, USD, VIX, lợi suất Mỹ và các chỉ số chứng khoán lớn
- `out/analysis/ml_macro_*.csv`: walk-forward feature-lift của ML khi thêm macro/global equity features
- `reports/active-models/latest/`: bản publish nhẹ của các forecast/report mới nhất để lưu trong repo khi cần
- `research/`: bundle research theo mã để Codex đọc nhanh hơn trong session tương tác

## VIC Single Model Technical Notes

- Target active hiện tại là direction: `TargetCloseRetPct > 0` cho từng horizon T+1..T+5.
- Winner hiện tại là `baseline_ohlc / logit_direction`, dùng `LogisticRegression(class_weight="balanced")` sau `median imputer + StandardScaler`.
- Holdout chuẩn là 5 phiên cuối trong daily cache; mỗi horizon được train bằng dữ liệu có `Date < holdout_base_date`, nên không nhìn vào 5 phiên test.
- Walkback dùng các mốc trước holdout để kiểm tra generalization trước khi chấp nhận winner.
- Candidate search vẫn thử cả model giá và model chiều hướng trên `baseline_ohlc`, `index_expiry_exvin`, và `engineered_all`.
- Feature engineering được giữ lại trong `vic_single_model_feature_engineering.csv`; candidate metrics giữ trong `vic_single_model_candidates.csv`. Candidate thua không được publish như forecast chính.

## Portfolio

Portfolio bị tắt trong workflow mặc định:

```yaml
portfolio:
  enabled: false
```

Khi tắt portfolio, engine không đọc `data/portfolios/portfolio.csv`, không merge vị thế vào `universe.csv`, và không kéo ticker từ portfolio vào scope. Chỉ bật lại `portfolio.enabled: true` khi cần chạy một phân tích legacy rõ ràng.

## Universe mặc định

`config/data_engine.yaml` pin working universe vào `VIC`. Notebook vận hành ở chế độ VIC-only, prediction-only.

Khuyến nghị vận hành:

- dùng `./broker.sh refresh_vic_map` hoặc alias `./broker.sh map` để quay lại scope mặc định `VIC`
- không mở rộng universe trong workflow phân tích mặc định; nếu cần đổi scope, hãy đổi contract trước rồi chạy lại pipeline liên quan

## Ghi chú vận hành với Codex

- Codex được phép đọc `out/`, `research/`, `reports/`, `config/`, `scripts/`
- Codex được phép sửa tool hoặc thêm utility nếu cần cho workflow research/forecast
- Không giả định có flow order execution downstream
- Nếu artifact thiếu hoặc stale, Codex phải tự chạy tuần tự và tự đợi batch xong trước khi trả lời
- Không tự dừng batch vì lâu nếu batch đó là điều kiện để ra forecast sạch
- Có thể tra cứu bài báo/trang tài chính/mạng xã hội/source mạng để tổng hợp bối cảnh, nhưng phải đặt trong phần `External synthesis` riêng
- Phần `Model predict` chỉ được lấy từ artifact/model trong repo; không dùng nguồn ngoài để sửa forecast, đổi confidence, hoặc gọi là tín hiệu model nếu chưa thành feature được backtest
- Network/API qua fetcher/pipeline trong repo được dùng để lấy dữ liệu đầu vào cho model; các dữ liệu này phải đi vào artifact có thể audit
- Macro/global market từ tra cứu ngoài chỉ là bối cảnh; chỉ coi là tín hiệu predict khi đã thành factor/feature trong repo và có walk-forward/feature-lift để kiểm chứng
- Với `VIC`, pair features nếu còn trong schema chỉ là context kỹ thuật; không tự kéo pair ticker vào phân tích mặc định
- `eval_deterministic` chỉ là harness replay/feature legacy
- Output mặc định phải tách `Model predict` khỏi `External synthesis` nếu có dùng nguồn ngoài

## Kiểm thử

Sau khi sửa Python code:

```bash
./broker.sh tests
```

## Cảnh báo

Công cụ này không đưa ra lời khuyên đầu tư. Nó chỉ chuẩn bị dữ liệu, forecast và artifact để bạn tự phân tích.
