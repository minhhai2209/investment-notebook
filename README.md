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

1. Scope mặc định đã được pin vào `VIC` trong `config/data_engine.yaml`; không cần refresh map nếu chỉ phân tích VIC.

```bash
./broker.sh prepare_default
```

2. Hoặc nếu chỉ muốn rebuild artifact từ map hiện có:

```bash
./broker.sh prepare
```

Scope phân tích mặc định hiện chỉ là `VIC`; các mã khác chỉ vào phân tích khi bạn hỏi rõ.

3. Mở Codex ngay trong repo này và hỏi trực tiếp.

Active model surface hiện đã consolidate còn 2 forecast tasks đáng tin cậy:

```bash
./broker.sh ohlc
./broker.sh intraday
```

- `ohlc`: dự báo giá T+n (`T+1/T+2/T+3/T+5/T+10/T+15/T+20`) từ daily OHLCV bằng nhiều candidate model rồi chọn best theo validation.
- `intraday`: dự báo close còn lại của phiên từ 1-minute OHLCV cộng market-depth/order-book feature bằng nhiều candidate model rồi chọn best theo validation. Pipeline tự refresh VNDIRECT priceboard depth và merge vào cache intraday trước khi build feature.
- Candidate set active: `ridge`, `random_forest`, `hist_gbm`, `mlp_deep`; deep-style MLP được thử nhưng không mặc định thắng nếu validation yếu hơn.

Các builder cũ như `range`, `cycle`, `timing`, `entry_ladder`, `sequence_dl`, `candidates`, `momentum`, `deep` vẫn có thể gọi tay để diagnostic/legacy, nhưng không còn là default model source.

Nếu bạn muốn dùng artifact đã build trên GitHub Actions mà không commit snapshot vào repo:

```bash
./broker.sh sync_artifacts
```

Lệnh này sẽ tìm artifact mới nhất có prefix `core-artifacts-` trên branch `main`, chỉ download nếu `digest` chưa có trong cache local, cập nhật `.cache/gh-artifacts/latest/core-artifacts`, và prune cache local cũ.

Ví dụ:

- `Nếu artifact chưa có hoặc stale thì tự chạy và chờ xong rồi mới phân tích.`
- `Nếu rebuild/fetch/full ML cần cho khuyến nghị hoặc lệnh cụ thể đang chạy lâu, Codex không được tự ý dừng. Nếu command lỗi, bị interrupt, hoặc đã dừng, phải nói rõ chưa có artifact đủ sạch và không đưa khuyến nghị/lệnh dựa trên phần chạy dở.`
- `Sau khi refresh artifact xong, tự check thêm tin tức live 12-24h gần nhất rồi mới chốt câu trả lời; không đưa news vào broker.sh.`
- `Nếu tôi chỉ ra một lỗi lặp lại hoặc một rule mới về cách làm việc, hãy cập nhật contract/docs của repo, không chỉ sửa cho một câu trả lời.`
- `Dựa trên snapshot mới nhất, VIC hôm nay xử lý thế nào?`
- `Liệt kê đầy đủ ứng viên theo format mua ngay / chờ / không mua.`
- `Nếu chưa có mã đủ chuẩn thì nói thẳng không mua.`
- `Nếu có mã mua được hoặc chờ được thì phải ghi rõ vùng giá cụ thể và size tham chiếu cho ngân sách 5 tỷ.`
- `Ưu tiên nấc đẹp hơn nấc dễ khớp; nếu ladder có nhiều nấc thì phải chỉ rõ nấc thăm dò và nấc chính.`
- `Ladder phải ưu tiên lấy từ artifact ML/research; nếu model chưa có nấc đủ đẹp thì nói thẳng chưa có lệnh, không tự nội suy thêm nấc từ giá hiện tại.`
- `Nếu đã hỏi tới mức giá/số lượng đặt lệnh cụ thể, Codex phải kiểm tra artifact còn cùng mốc với giá hiện tại; nếu stale thì phải rebuild hoặc nói rõ chưa đủ sạch để đặt.`
- `Nếu tôi đưa giá live mới, hỏi trong giờ nghỉ trưa, ngoài phiên, sau ATO, gần ATC, hoặc sau một quãng thời gian đáng kể so với snapshot, Codex phải fetch/rebuild lại snapshot giá và artifact liên quan trước khi nói lệnh cụ thể. Nếu full ML chạy chậm thì phải tiếp tục chạy cho xong trừ khi tôi yêu cầu dừng; không tự ý chuyển sang khuyến nghị tạm cho lệnh/size lớn.`
- `Khi hỏi cho phiên kế tiếp, phải nói rõ nên đặt trước phiên, chờ ATO, hay đợi sau ATO rồi mới hành động.`
- `Khi hỏi hiện tại còn mua không, phải nói thêm đà các phiên tới là tăng tiếp / đi ngang / giảm, độ gấp mua ngay hay có thể chờ, và overlay lịch nghỉ lễ/sự kiện/tin tức live.`
- `Khi hỏi có nên mua một mã cụ thể ngay không, phải so sánh với việc giữ tiền chờ các mã trọng tâm còn lại; không được chỉ xét riêng mã đó.`
- `Decision cuối cùng phải dựa trên forecast/validation riêng từng mã; không dùng score cộng/trừ thủ công hoặc overlay hard-code để ép mua/chờ/không mua.`
- `Không dùng nhãn cảm tính như nóng/khỏe/yếu/mạnh để khuyến nghị. RSI/SMA/ret5d chỉ là feature hoặc validation context; action phải đi từ ML forecast, MAE/hit rate, zone/ladder artifact hoặc position action-sizing.`
- `Nếu cần chạy batch thì tự chạy tuần tự xong rồi mới trả lời, không trả lời giữa chừng rằng vẫn đang đợi artifact.`

## Wrapper commands

```bash
./broker.sh tests
./broker.sh engine
./broker.sh prepare
./broker.sh research
./broker.sh map
./broker.sh refresh_vic_map
./broker.sh sync_artifacts
./broker.sh prepare_default
./broker.sh position_ml --ticker VIC --quantity 20000 --avg-price 214.53 --current-price <live_price>
./broker.sh refresh_macro
./broker.sh eval_macro --no-refresh-factors --case-tickers VIC
./broker.sh eval_macro_lift --case-tickers VIC
./broker.sh ohlc
./broker.sh intraday
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
./broker.sh sequence_dl
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
- `out/analysis/ml_ohlc_next_session.csv`: active T+n price forecast cho phiên kế tiếp, model được chọn theo validation
- `out/analysis/ml_ohlc_multi_session.csv`: active T+n price forecast nhiều horizon mặc định `T+1/T+2/T+3/T+5/T+10/T+15/T+20`; `T+N` là N phiên giao dịch sau snapshot
- `out/analysis/ml_ohlc_model_metrics.csv`: backtest metrics của active T+n price model
- `out/analysis/ml_intraday_rest_of_session.csv`: active 1-minute OHLCV + depth close forecast; gồm nhãn/xác suất `PredCloseUpFromSnapshot*`, `PredRecoverToPrevClose*`, số mẫu calibration, và `RecoveryCalibrationConflict`
- `out/analysis/ml_intraday_rest_of_session_metrics.csv`: validation metrics của intraday model theo bucket/time window
- `out/analysis/ml_intraday_rest_of_session_backtest.csv`: recent holdout predictions của best intraday model theo bucket, gồm actual/pred recover và close error để audit từng phiên
- `out/analysis/positions/`: review vị thế đang nắm theo ML-only, gồm P/L hiện tại, forecast P/L theo horizon và error band; không tự sinh rule bán/mua thêm
- `out/analysis/macro_factor_*.csv`: độ nhạy/correlation của từng mã với dầu, vàng, USD, VIX, lợi suất Mỹ và các chỉ số chứng khoán lớn như S&P 500, Nasdaq, Dow Jones, Euro Stoxx 50, DAX, FTSE 100, CAC 40, Nikkei 225, KOSPI
- `out/analysis/ml_macro_*.csv`: walk-forward feature-lift của ML khi thêm macro/global equity features; dùng để biết correlation có thật sự cải thiện predict hay không
- `research/`: bundle research theo mã để Codex đọc nhanh hơn trong session tương tác

## Danh mục là optional

Repo notebook không cần danh mục để chạy. Nếu không có `data/portfolios/portfolio.csv`:

- engine vẫn chạy bình thường
- `positions.csv` sẽ rỗng
- các cột vị thế trong `universe.csv` sẽ về `0` hoặc `NaN`

Nếu bạn muốn dùng thêm context vị thế nội bộ thì chỉ cần tự đặt file `data/portfolios/portfolio.csv` với schema:

```csv
Ticker,Quantity,AvgPrice
VIC,20000,214.53
```

## Universe mặc định

`config/data_engine.yaml` pin working universe vào `VIC` để trả lời nhanh và nhất quán. Notebook vận hành ở chế độ VIC-only.

Khuyến nghị:

- dùng `./broker.sh refresh_vic_map` hoặc alias `./broker.sh map` để quay lại scope mặc định `VIC`
- không mở rộng universe trong workflow phân tích mặc định; nếu cần đổi scope, hãy đổi contract trước rồi chạy lại pipeline liên quan

## Ghi chú vận hành với Codex

Repo này được thiết kế để mở một session Codex mới rồi làm việc như notebook:

- Codex được phép đọc `out/`, `research/`, `config/`, `scripts/`
- Codex được phép sửa tool hoặc thêm utility nếu cần cho workflow research
- Không giả định có flow order execution downstream
- Nếu artifact thiếu hoặc stale, Codex phải tự chạy tuần tự và tự đợi batch xong trước khi trả lời
- Không tự dừng batch vì lâu nếu batch đó là điều kiện để ra khuyến nghị/lệnh. Nếu batch chưa xong, lỗi, bị interrupt, hoặc đã bị dừng, output phải là `chưa đủ artifact sạch để khuyến nghị`, không được chốt lệnh từ dữ liệu chạy dở.
- Khi đang trong intraday, nghỉ trưa, ngoài phiên, sau ATO/gần ATC, hoặc người dùng đưa giá live lệch snapshot, Codex phải coi artifact lệnh là stale cho tới khi fetch/rebuild lại snapshot giá; không dùng ladder/no-chase cũ để chốt LO cụ thể.
- Model intraday chuẩn phải dùng đơn vị nhỏ nhất hiện có là 1-minute bars, gồm giá OHLC, volume, và depth/order book. Pipeline phải tự refresh depth trước khi build intraday forecast.
- Sau bước refresh artifact, Codex phải tự browse tin tức live cùng ngày hoặc 12-24h gần nhất để overlay macro/geopolitics/policy khi trả lời `hôm nay mua gì`; lớp này là bước hỏi đáp, không phải lệnh batch của repo
- Khi macro/geopolitics/oil/global market có thể ảnh hưởng HOSE, chạy hoặc đọc `./broker.sh eval_macro --no-refresh-factors --case-tickers VIC` sau khi cache đã refresh để xem correlation không nhất thiết cùng chiều, rồi đọc `./broker.sh eval_macro_lift --case-tickers VIC` để kiểm tra macro/global equity features có cải thiện predict so với baseline không.
- Với `VIC`, pair features nếu còn trong schema chỉ là context kỹ thuật; không tự kéo pair ticker vào phân tích mặc định.
- Khi trả lời `hiện tại thì sao`, Codex phải đọc active OHLC T+n và intraday morning-close nếu trong phiên, rồi check lịch nghỉ lễ/sự kiện/tin tức live trước khi chốt độ gấp.
- Khi trả lời câu hỏi mua một mã cụ thể ngay hay chờ, Codex chỉ so sánh với mã khác nếu người hỏi nêu rõ mã đó; scope mặc định không còn mã đối chứng ngoài `VIC`.
- Lớp quyết định không được cộng/trừ điểm thủ công; nếu có overlay ticker/archetype thì chỉ dùng làm mô tả context, còn mua/chờ/không mua phải đến từ forecast/validation per-ticker và vùng giá từ artifact.
- `eval_deterministic` chỉ là harness replay/feature legacy; không dùng nó làm nguồn ra quyết định trong câu trả lời tương tác.
- Khi xử lý vị thế đã mua, không được tự chế mốc bán/cắt/size. Phải dùng `./broker.sh position_ml` để đọc forecast P/L và error band; nếu chưa có model action-sizing thì nói rõ chưa đủ sạch để đưa lệnh định lượng.
- Output khuyến nghị phải model-first: các cụm như `quá nóng`, `đang khỏe`, `yếu`, `đuổi giá` không được dùng làm lý do nếu không có artifact/model metric tương ứng.
- Artifact forecast active phải ghi rõ model family/class. Hai active forecast tasks đều thử `ridge`, `random_forest`, `hist_gbm`, `mlp_deep` rồi chọn best theo validation; các builder khác chỉ là legacy/diagnostic khi gọi tay.
- Không dùng flow nền hay nhiều builder chồng nhau, trừ khi từng job ghi ra output riêng và không dùng chung cache/history
- Khẩu vị mặc định của repo này là: ngân sách tham chiếu khoảng `5 tỷ`, ưu tiên size lớn, và phải liệt kê đầy đủ ứng viên khả thi thay vì ép chọn đúng một mã
- Contract đầu ra mặc định là: liệt kê ứng viên theo `mua ngay`, `chờ`, hoặc `không mua`; với mỗi mã `mua ngay` hoặc `chờ`, phải nêu `vùng giá cụ thể`, `quy mô vốn/số lượng`, và `nên đặt trước phiên / chờ ATO / đợi sau ATO`

## Kiểm thử

Sau khi sửa Python code:

```bash
./broker.sh tests
```

## Cảnh báo

Công cụ này không đưa ra lời khuyên đầu tư. Nó chỉ chuẩn bị dữ liệu và artifact để bạn phân tích thủ công.
