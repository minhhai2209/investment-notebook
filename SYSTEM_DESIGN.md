# Kiến trúc Data Engine (2025)

## Mục tiêu

Phiên bản này bỏ hoàn toàn order engine. Toàn bộ hệ thống chỉ còn các thành phần sau:

1. **Engine thu thập dữ liệu** (`scripts/engine/data_engine.py`): tải dữ liệu giá, tính chỉ số kỹ thuật, dựng bands/sizing/signals rồi hợp nhất thành snapshot. Khi khởi chạy, engine sẽ xoá sạch `out/`; kết thúc sẽ ghi `out/universe.csv` (kết hợp trường kỹ thuật + Sector + cột vị thế + EngineRunAt + các đại lượng khách quan cho breadth/relative strength/co-movement), `out/positions.csv`, `out/market_summary.json`, `out/sector_summary.csv`, rồi đồng bộ các input live sang `codex_universe/` (fail-fast nếu thư mục Codex không sạch).
2. **Kho dữ liệu danh mục** (`data/portfolios/`): lưu trữ đúng một danh mục duy nhất (file canonical `portfolio.csv`).
3. **TCBS Scraper** (`scripts/scrapers/tcbs.py`): đăng nhập TCBS bằng Playwright (Chrome) và ghi `data/portfolios/portfolio.csv`. `broker.sh tcbs` sẽ dọn sạch `codex_universe/` (hoặc `CODEX_DIR`) trước khi lấy danh mục để tránh file cũ. Script đặt lệnh (`scripts/scrapers/tcbs_orders.py`) đọc `codex_universe/orders.csv`, bấm xác nhận trên TCBS và ghi log `out/tcbs_confirmed_orders.csv` + `out/tcbs_error_orders.csv`.
4. **Codex scripted chat** (target `broker.sh codex`): dùng `codex exec` + `resume` (không cần TTY) trong thư mục `codex_universe/`. Trước khi gọi `codex exec`, wrapper sẽ re-sync lại toàn bộ bundle live mới nhất từ `out/` sang `codex_universe/`, materialize `total_capital_kVND.txt` (1 số duy nhất, kVND) từ `CODEX_TOTAL_CAPITAL_KVND` hoặc parse từ `CODEX_BUDGET_TEXT`, rồi build thêm `bundle_manifest.json` tại đó. Cùng contract sync này cũng được dùng cho `broker.sh research` để bundle không thiếu `total_capital_kVND.txt` khi chỉ rebuild research. Bundle live hiện dùng một lớp thesis theo mã duy nhất là `research/`; khi rebuild research, builder sẽ prune các thư mục ticker cũ ngoài working universe để tránh note stale cho mã đã bị loại khỏi core. Nếu thiếu source file sau bước engine hoặc manifest build lỗi, wrapper fail-fast thay vì chạy trên bundle cũ hoặc bundle rỗng. Prompt chuẩn ở `prompts/PROMPT.txt` được giữ ở mức policy/workflow/hard-constraints; phần thesis, allocation guidance, và execution anchors ưu tiên lấy từ `bundle_manifest.json`, `research/state.json`, và các artifact ML để Codex còn room làm lớp hòa giải cuối cùng. Sau đó wrapper đọc `prompts/PROMPT.txt` làm prompt đầu tiên, gửi kickoff + lặp lại "Tiếp tục." cho đến khi xuất hiện `DONE.md`; khi một lượt `resume` đang chạy mà cả `orders.csv` lẫn `DONE.md` đã có, watchdog sẽ chờ thêm đúng `grace_after_csv` rồi chủ động terminate lượt đó để tránh treo hậu kỳ quá lâu. Nếu khi kết thúc vẫn chưa có `orders.csv` thì fail-fast, rồi in nội dung `orders.csv` ra stdout. Wrapper fail‑fast nếu vượt timeout / số lần nhắc tối đa hoặc prompt trống. Đồng thời, script cũng ghi sẵn file `resume_last_codex.sh` trong thư mục log chung (`out/codex/`) để có thể resume lại thread gần nhất theo đúng tham số model/working dir. Sau khi có `DONE.md`, wrapper copy `orders.csv`, `DONE.md`, và log Codex tương ứng vào `archives/codex_runs/<YYYYMMDD_HHMMSS>/` như local staging, đồng thời copy nguyên thư mục `codex_universe/` sang `codex_universe_history/<TCBS account>/<YYYY>/<timestamp>/` trong repo chính để giữ snapshot bundle đầy đủ cho hậu kiểm. Nếu bật `CODEX_ORDER_HISTORY_ENABLED`, wrapper vẫn clone/pull repo lịch sử lệnh riêng, copy artifact sang `<TCBS account>/<YYYY>/<timestamp>/`, commit và push ngay ở repo đó.
5. **Offline replay harness** (`scripts/analysis/evaluate_deterministic_strategies.py`): đọc cache `out/data/*.csv` + `data/industry_map.csv`, replay nhiều market overlays và ticker selectors, rồi ghi báo cáo đánh giá dưới `out/analysis/` để hậu kiểm thesis `market -> sector -> ticker`.
6. **Offline ML baseline** (`scripts/analysis/evaluate_ml_models.py`): dựng tập mẫu cross-sectional từ cùng bộ cache, train/predict theo walk-forward với `scikit-learn`, và ghi `ml_*.csv/json` dưới `out/analysis/` để so deterministic vs ML.
7. **Offline VNINDEX ML baseline** (`scripts/analysis/evaluate_vnindex_models.py`): dùng feature state của chỉ số + breadth/co-movement để train/predict walk-forward cho hướng đi `VNINDEX` trong `5` và `10` phiên tới.
8. **Offline multi-horizon OHLC/range ML baseline** (`scripts/analysis/evaluate_ohlc_models.py`): train direct model riêng theo từng mã và từng horizon `T+1..T+10` từ lịch sử OHLCV của mã đó cộng với context `VNINDEX`, rồi dự báo `open/high/low/close` và `range` cho từng horizon, ghi `ohlc_*.csv/json` dưới `out/analysis/`. Shared daily feature layer của baseline này hiện không chỉ có daily lag/momentum mà còn có weekly-context như `week-to-date`, khoảng cách tới `high/low` tuần trước, range/return tuần trước, và nhịp volume tuần để flow vẫn thêm mã mới được mà không cần branch riêng cho single-name.
9. **Offline BCTT feature-lift harness** (`scripts/analysis/evaluate_bctt_feature_lift.py`): lấy/cached bảng `BCTT` quý từ Vietstock, dựng feature set `ratios_only` và `hybrid_growth`, rồi replay walk-forward để đo lift so với baseline ML; artifact nằm dưới `out/analysis/`, cache thô nằm dưới `out/vietstock_bctt/`.
10. **Offline BCTT cache refresh** (`scripts/data_fetching/refresh_vietstock_bctt_cache.py`): làm đầy cache `out/vietstock_bctt/` cho một universe lớn hơn, rồi ghi coverage summary dưới `out/analysis/vietstock_bctt_cache_summary.csv` để các replay sau không phải fetch lại từng mã từ đầu.
11. **Offline macro-factor cache + sensitivity** (`scripts/data_fetching/macro_factor_cache.py`, `scripts/analysis/evaluate_macro_factor_sensitivity.py`): cache chuỗi thời gian vàng/dầu/USD/VIX/Nasdaq/SP500/US10Y dưới `out/macro_factors/`, rồi đo rolling correlation/beta và phản ứng lịch sử của từng mã với các shock factor; artifact nằm dưới `out/analysis/`.
12. **ML range report + live ML input** (`scripts/analysis/build_range_forecast_report.py`): fetch một cache daily riêng gần `2 năm giao dịch`, rồi train/chấm model riêng theo từng `ticker x horizon x variant` (`full_2y`, `recent_focus`) trước khi ghi report dưới `out/analysis/`. Trong flow live, `broker.sh tcbs` cũng chạy report này trên chính `out/universe.csv`, rồi sync `ml_range_predictions_full_2y.csv` và `ml_range_predictions_recent_focus.csv` sang `codex_universe/`.
13. **Cycle forecast report + live ML input** (`scripts/analysis/build_cycle_forecast_report.py`): fetch cache daily đủ sâu, dựng target `peak return / time-to-peak / drawdown` trong các khung `1M..6M` với sell-delay mặc định `T+3`, rồi train/chấm riêng theo từng `ticker x horizon x variant` cho các model candidate. Ngoài `full_2y` và `recent_focus`, builder hiện có thêm `quarter_focus` để bắt nhịp regime gần hơn mà vẫn giữ nguyên contract output live. Trong flow live, `broker.sh tcbs` chạy report này trên chính `out/universe.csv`, rồi sync `cycle_forecast_ticker_matrix.csv` và `cycle_forecast_best_horizon_by_ticker.csv` sang `codex_universe/` dưới tên `ml_cycle_forecast_ticker_matrix.csv` và `ml_cycle_forecast_best_horizon.csv`.
14. **Per-ticker technical playbook harness** (`scripts/analysis/build_ticker_playbook_report.py`): backtest nhiều family rule đơn giản theo từng mã (`washout_reclaim`, `trend_pullback`, `breakout_followthrough`, `trend_reacceleration`), tách `train/test` theo thời gian, rồi chọn playbook tốt nhất cho từng ticker. Artifact được ghi dưới `out/analysis/ticker_playbooks_*/`; trong flow live, `broker.sh tcbs` chạy một bản rút gọn trên chính `out/universe.csv` và sync `ticker_playbook_best_configs.csv` sang `codex_universe/`.
15. **Next-session OHLC live report** (`scripts/analysis/build_ohlc_next_session_report.py`): tái dùng harness OHLC daily để chọn model đang thắng theo từng mã ở horizon `T+1`, rồi ghi flat file `ml_ohlc_next_session.csv` dưới `out/analysis/`. Selection score của builder này hiện có penalty bất đối xứng cho `upside miss` và `downside miss`, để model không thắng máy móc chỉ vì close/range đẹp nhưng bỏ lỡ `high` hoặc đánh giá quá nông phần downside. Trong flow live, `broker.sh tcbs` sync file này sang `codex_universe/` như lớp execution-price structure cho phiên kế tiếp.
16. **Intraday rest-of-session live report** (`scripts/analysis/build_intraday_rest_of_session_report.py`): cache intraday `5m` theo từng mã + `VNINDEX`, dựng sample lịch sử trên nhiều bucket trong phiên (`AM_EARLY`, `AM_LATE`, `LUNCH_BREAK`, `PM_EARLY`, `PM_LATE`), train/chọn model riêng theo từng `ticker x SnapshotTimeBucket`, rồi ghi file optional `ml_intraday_rest_of_session.csv` dưới `out/analysis/` nếu `EngineRunAt` vẫn còn cửa giao dịch có thể thực thi trong ngày. Feature layer của report này không chỉ nhìn ret `5m/15m/30m` mà còn suy ra thêm context kiểu hourly từ chuỗi `5m` như `60m return`, `30m/60m range`, và vị trí giá trong range ngắn, để tăng độ nhạy ngắn hạn mà chưa cần mở nhánh tick riêng. Selection score cũng có penalty bất đối xứng cho `upside miss` và `downside miss`, giúp giảm bias underpredict `high` ở các tape đang squeeze. Current snapshot được re-label theo context `EngineRunAt`, để run trong giờ nghỉ trưa không bị neo nhầm vào bucket của bar intraday cuối cùng. File này chỉ được sync sang `codex_universe/` khi thực sự tồn tại.
17. **Single-name timing live report** (`scripts/analysis/build_single_name_timing_report.py`): tái dùng feature daily theo từng mã để dựng target `peak return / days-to-peak / drawdown / close return` trên các horizon ngắn như `T+3`, `T+5`, `T+10`, rồi train/chấm model riêng theo từng `ticker x horizon x variant`. Bên cạnh `full_2y` và `recent_focus`, builder hiện có thêm `quarter_focus` để ưu tiên trạng thái gần hạn khi regime đổi nhanh; output live vẫn giữ nguyên schema cũ và chỉ đổi giá trị `Variant` khi nhánh này thắng. Trong flow live, `broker.sh tcbs` chạy report này trên `out/universe.csv`, ghi `ml_single_name_timing.csv` dưới `out/analysis/`, và sync file sang `codex_universe/` như lớp trade-efficiency bổ sung cho từng mã.
18. **Entry ladder evaluation live report** (`scripts/analysis/build_entry_ladder_eval_report.py`): dùng các output forecast hiện có (`ml_ohlc_next_session`, blended range `T+5/T+10`, cycle best horizon, single-name timing`) để re-anchor edge theo từng mức `LimitPrice` BUY hợp lệ của mỗi mã. Bản hiện tại fit thêm classifier riêng theo từng `ticker x horizon` trên cache daily để ước lượng xác suất “giá low chạm entry” ở `T+1/T+5/T+10`; nếu mẫu quá ít thì fallback về fill-score heuristic. `EntryScore` dùng fill-score như trọng số trực tiếp cho edge thay vì giữ một sàn cố định, để các lệnh BUY rất sâu nhưng xác suất chạm thấp không thắng điểm chỉ nhờ upside cơ học từ entry thấp hơn. Report cuối vẫn là bảng `ticker x LimitPrice` với upside, downside, fill-score và `EntryScore` để Codex chọn nấc giá hợp lý hơn thay vì chấm cả mã chỉ tại `Base/Last`.
19. **Per-ticker research bundle** (`scripts/research/build_research_bundle.py`): dựng lớp research có cấu trúc cho working universe sau khi các artifact live đã sẵn sàng. Output nằm dưới `research/` gồm `manifest.json`, `tickers/<TICKER>/profile.md`, `weekly/<YYYY-Www>.md`, `daily/<YYYY-MM-DD>.md`, và `state.json`. `state.json` không chỉ mang thesis/vùng giá mà còn có allocation guidance như `TargetWeightMinPct`, `TargetWeightMaxPct`, `WeaknessBuildPct`, `StrengthReservePct`; đồng thời phân biệt rõ allowance mặc định của archetype (`DefaultAddOnStrengthAllowed`, `DefaultAddOnWeaknessAllowed`) với allowance thực tế ở snapshot hiện tại (`AddOnStrengthAllowed`, `AddOnWeaknessAllowed`) để prompt không nhầm style mặc định với quyền add hiện hành. Nếu repo có `human_notes.md` và builder parse được overlay theo mã, `state.json` còn ghi thêm `HumanTargetPrice`, `HumanTargetYear`, `PersistentWeaknessBid`, cùng thống kê burst/follow-through kiểu `T+2.5`. Builder cũng đọc tape `5m` gần nhất để suy ra execution fields như `ExecutionBias`, `OpeningSqueezeFailure`, `BurstExecutionBias`, `TrimAggression`, `UrgentTrimMode`, `MustSellFractionPct`; đây là lớp heuristic mô tả hành vi tape gần nhất, không phải order-book microstructure. `UrgentTrimMode` và `MustSellFractionPct` được dùng để tránh tình huống SELL treo quá tham vọng khi tape xấu: nếu prompt quyết định có SELL ở mã đó thì phải front-load ít nhất một tranche thực dụng trước khi đặt các mức stretch. Builder luôn rewrite lại `profile/weekly/daily` theo đúng `EngineRunAt` hiện tại để tránh trường hợp `state.json` mới nhưng markdown artifact còn stale. `manifest.json` có thêm `PortfolioAllocator` ở cấp danh mục để giảm risk under-invested; allocator này dùng cùng `total_capital_kVND` với prompt nếu wrapper cung cấp giá trị đó, thay vì chỉ lấy `EnginePortfolioMarketValue_kVND` làm mẫu số. Nếu working universe chỉ có một mã đầu tư, allocator tự bật `SingleNameMode` và để mã đó kế thừa target invested của cả danh mục, thay vì giữ band cap kiểu multi-name; đồng thời nó tách `DeployableGapPct` cho toàn bộ kế hoạch nhiều phiên khỏi `SessionBuildCapPct` cho phiên kế tiếp, để tránh dồn hết gap vào một lượt mua. `SessionBuyTranches` giờ có thể sinh thêm `continuation_reserve` bên cạnh `bridge` và các nấc core sâu, nhằm giữ một reserve nhỏ cho trường hợp mã không pullback mà vẫn continuation sạch. Khi universe rất nhỏ, market breadth trong `market_summary.json` chuyển sang lấy từ benchmark basket thay vì chính working universe để regime không bị méo bởi sample quá ít. Khi có nhiều mã, allocator chuyển sang `GlobalBuyTranches`: research vẫn sinh `SessionBuyTranches` theo từng mã, nhưng phân bổ vốn phiên được quyết định ở cấp tranche xuyên mã thay vì quota cứng theo ticker. `broker.sh tcbs` chạy builder này tự động; `broker.sh research` cho phép rebuild/sync riêng bundle research mà không scrape TCBS lại.
20. **Bundle manifest builder** (`scripts/codex/build_bundle_manifest.py`): dựng `bundle_manifest.json` ngay trong `codex_universe/` sau mỗi lần sync bundle live. File này là contract kỹ thuật để prompt daily biết file nào bắt buộc, file nào optional, schema tối thiểu của từng file, working universe hiện đang có trong bundle, cùng `Summary`/`UsageNotes` canonical để tránh phải lặp mô tả file quá dài trong prompt.
21. (Tạm thời vô hiệu) GitHub Action: trước đây workflow tại `.github/workflows/data-engine.yml` chạy engine định kỳ và commit kết quả. Hiện đã gỡ; chạy local thay thế.

Mọi quyết định giao dịch vẫn do người vận hành xử lý dựa trên dữ liệu đầu ra. Prompt/Codex flow hiện đọc `market_summary.json`, `sector_summary.csv`, `universe.csv`, `ml_range_predictions_full_2y.csv`, `ml_range_predictions_recent_focus.csv`, `ml_cycle_forecast_ticker_matrix.csv`, `ml_cycle_forecast_best_horizon.csv`, `ticker_playbook_best_configs.csv`, `ml_ohlc_next_session.csv`, và nếu có thì cả `ml_intraday_rest_of_session.csv`, `ml_single_name_timing.csv`, lẫn `ml_entry_ladder_eval.csv`; contract lệnh cuối cùng vẫn là `orders.csv` cũ. Execution rule mới được tách theo ngữ cảnh phiên:
- `overnight / phiên kế tiếp chưa mở`: prompt phải sinh batch một chiều theo từng mã (`single-side per ticker`), vì hệ thống tạo lệnh trước mở cửa không hỗ trợ vừa `SELL` vừa `BUY` trên cùng ticker trong cùng batch. Ở cấp batch, prompt phải chọn `ATO priority side` (`BUY-first` hoặc `SELL-first`) theo objective/risk context của phiên; `orders.csv` phải xếp các lệnh của phía ưu tiên lên trước để người vận hành biết nên nhập ATO phía nào trước.
- `in-session / phần còn lại của phiên hiện tại`: prompt vẫn được phép giữ cả `SELL` và `BUY` trên cùng ticker nếu execution context yêu cầu; khi đó thứ tự dòng trong `orders.csv` phải ưu tiên `SELL` trước `BUY`.

## Dòng dữ liệu

```
┌───────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│ data_engine.py│──►──│ out/*.csv            │────►│ ChatGPT / analyst UI │
└─────▲─────────┘     └────────────▲────────┘     └──────────▲───────────┘
      │                              │                           │
      │                              │                           │
      │            ┌─────────────────┴─────────────┐             │
      │            │ data/portfolios/portfolio.csv  │◄────┐      │
      │            └───────────────────────────────┘      │      │
      │                     ▲                            │      │
      │                     │                            │      │
      └────────────── tcbs.py ◄────────────── Fetch via browser ◄┘

       │
       ▼
               └──► codex exec/resume (CLI) ──► codex_universe/universe.csv (input)
```

- Engine đọc universe từ `config/data_engine.yaml` (tối thiểu cột `Ticker` và `Sector`).
- Trước bước engine, `broker.sh tcbs` refresh `data/industry_map.csv`; nếu `config/data_engine.yaml` có `universe.core_tickers` thì wrapper refresh theo đúng `core_tickers`, thay vì seed từ `VN100 + portfolio + NVL`.
- Nếu đặt biến môi trường `INDUSTRY_TICKER_FILTER` (nạp từ `.env` nếu có), engine sẽ chỉ giữ các mã được liệt kê trong danh sách này (phân tách bởi dấu phẩy/khoảng trắng) cho cả universe và danh mục.
- Nếu không đặt `INDUSTRY_TICKER_FILTER` nhưng `config/data_engine.yaml` có `universe.core_tickers`, engine sẽ coi `core_tickers` là working universe active và tự loại các mã ngoài danh sách khỏi universe, positions, và các bundle live downstream.
- Khi không có active ticker filter, engine mới fallback về hành vi rộng hơn là union current portfolio để tránh rơi mất coverage.
- Repo tách rõ hai lớp universe:
  - `selection universe`: chạy offline rộng hơn trên `VN30`, `VN100`, hoặc list chỉ định để tuyển chọn
  - `working universe`: flow live hàng ngày dùng active ticker filter (`INDUSTRY_TICKER_FILTER` hoặc `config.data_engine.yaml: universe.core_tickers`)
- Ở trạng thái hiện tại, base core hằng ngày được giữ ở:
  - `MBB`, `HPG`, `NVL`
- Dữ liệu lịch sử và intraday lấy từ API VNDIRECT (module `collect_intraday` và `fetch_ticker_data`). Cache được lưu ở `out/data/`.
- Tất cả output CSV nằm dưới `out/`. Khi workflow bị gỡ, bạn cần tự commit/push khi có thay đổi.
- Replay harness chỉ ghi thêm dưới `out/analysis/`; đây là artifact offline, không ảnh hưởng flow `tcbs -> codex -> orders`.
- Ticker playbook harness là artifact offline phục vụ research, nhưng `broker.sh tcbs` hiện đã sync `ticker_playbook_best_configs.csv` sang bundle live để prompt dùng như clue theo từng mã; nó vẫn không tự động override các lớp dữ liệu khác.
- Cách regen playbook:
  - research rộng: chạy trực tiếp `scripts/analysis/build_ticker_playbook_report.py --tickers ... --output-dir ...`
  - live bundle: chạy `./broker.sh tcbs`; wrapper sẽ tự build `out/analysis/ticker_playbooks_live/ticker_playbook_best_configs.csv` từ working universe hiện tại rồi sync sang `codex_universe/`
- Cách đọc playbook:
  - xem `RuleFamily` để biết mã hợp kiểu setup nào
  - xem `TestTrades`, `TestWinRatePct`, `TestAvgRetPct`, `TestAvgHoldDays`, `TestWorstDrawdownPct` để đánh giá độ dùng được ở giai đoạn gần đây
  - dùng các cột `All*` để đối chiếu lịch sử dài hơn
  - không dùng playbook như rule tự động; chỉ dùng như clue về “tính cách kỹ thuật” riêng của từng mã
- ML baseline `T+1..T+10` vẫn ghi dưới `out/analysis/` như harness offline; trong flow live, prompt/order flow hiện inject cả hai lớp ML khách quan: range forecast ngắn hạn `T+N` và cycle forecast `1M..6M`.
- VNINDEX ML baseline cũng chỉ là artifact offline, phục vụ review bias thị trường chứ chưa override `market_summary.json`.
- OHLC ML baseline cũng chỉ là artifact offline; mục tiêu là đo xem bài toán dự báo giá/range ngắn hạn `T+1..T+10` có đủ tín hiệu để dùng làm insight bổ sung hay không.
- Cycle forecast report vừa là harness offline, vừa là một phần của contract live hiện tại qua hai file `ml_cycle_forecast_*`.
- Quy tắc vận hành của cycle forecast:
  - screening rộng thì chạy trên `selection universe`
  - sau khi chọn xong mới promote mã vào `core_tickers` hoặc `preferred_tickers`
  - engine live không tự lấy kết quả screening để mở rộng working universe
- Checklist tuyển chọn `core_tickers` đang dùng:
  - bắt đầu từ `VN100` làm selection universe mặc định, không từ working universe hiện tại
  - đọc tin tức mới nhất của thị trường và các ứng viên cuối trước khi chốt shortlist
  - đọc đồng thời `ml_range_predictions_*` và `ml_cycle_forecast_*`
  - loại sớm các cụm có overhang pháp lý / dự án / commodity / event-driven nếu không hợp mục tiêu hiện tại
  - ưu tiên doanh nghiệp đủ “xương sống”: lớn, thanh khoản tốt, business dễ giải thích, governance sạch hơn mặt bằng
  - khi chấm ML, nhìn cả `PredPeakRetPct`, `PredPeakDays`, `PredDrawdownPct`, và `SelectionScore`; không chọn chỉ vì upside cao
  - `preferred_tickers` chỉ là vùng người dùng tự thêm theo thesis riêng, không được dùng để làm loãng tiêu chuẩn core
  - revisit core tối thiểu hàng quý, hoặc khi có thay đổi mạnh về thesis ngành, legal/governance risk, thanh khoản, hoặc profile ML/drawdown
- BCTT feature-lift harness cũng chỉ là artifact offline; dữ liệu quý được cache riêng để đánh giá ảnh hưởng lên stock ranking, chưa được merge vào runtime engine.
- Macro-factor sensitivity cũng chỉ là artifact offline; mục tiêu là thêm ngữ cảnh `mã nào đang nhạy với dầu/vàng/risk-off` để hỗ trợ giải thích, chưa được nối vào prompt live.
- ~2-year ML range report hiện được tune riêng theo từng mã và vẫn tạo artifact offline đầy đủ; đồng thời hai file `ml_range_predictions_full_2y.csv` và `ml_range_predictions_recent_focus.csv` cũng đã quay lại contract live của Codex như lớp forecast giá/xu hướng bổ sung.

## Nguồn dữ liệu upstream

Để tránh trùng lặp tài liệu, mọi nguồn dữ liệu chính được mô tả ngắn gọn ngay trong thiết kế hệ thống:

- **Giá lịch sử (daily OHLCV)** – lấy từ VNDIRECT dchart API thông qua `scripts/data_fetching/fetch_ticker_data.py`. Engine gọi lớp `VndirectMarketDataService.load_history`, đảm bảo cache dưới `out/data/` đủ sâu (ít nhất `data.history_min_days`).
- **Giá và thanh khoản intraday** – lấy từ cùng API VNDIRECT (resolution phút) qua `scripts/data_fetching/collect_intraday.py`. Engine dùng `load_intraday` để bổ sung `Last`, `IntradayVol_shares`, `IntradayValue_kVND` và các chỉ báo ngắn hạn.
- **Dòng tiền khối ngoại / tự doanh (CafeF)** – module `scripts/data_fetching/cafef_flows.py` tải và cache CSV dưới `out/cafef_flows/{foreign,proprietary}/`. Khi `data.cafef_flow_enabled: true`, engine tự merge các trường như `ForeignFlowDate`, `ProprietaryFlowDate`, `NetBuySellForeign_*`, `NetBuySellProprietary_*`, `ForeignRoomRemaining_shares`, `ForeignHoldingPct` vào `out/universe.csv`. Tham số `data.cafef_flow_max_age_hours` > 0 (mặc định `4` giờ) cho phép refresh định kỳ khi cache cũ hơn TTL; giá trị 0 chỉ dùng cache hiện có và không fetch thêm.
- **Chỉ tiêu cơ bản tóm tắt (Vietstock overview)** – module `scripts/data_fetching/vietstock_overview_api.py` đọc HTML tĩnh `https://finance.vietstock.vn/<TICKER>-ctcp.htm`, cache JSON ở `out/vietstock_overview/` rồi trích xuất các field như `PE_fwd`, `PB`, `ROE`. Việc bật/tắt và TTL được điều khiển qua khối `data` trong `config/data_engine.yaml`; `vietstock_overview_max_age_hours` > 0 cho phép refresh định kỳ, còn 0 (mặc định) coi dữ liệu là non‑daily và chỉ fetch khi thiếu.
- **Báo cáo tài chính quý (Vietstock BCTT)** – module `scripts/data_fetching/vietstock_bctt_api.py` mở `https://finance.vietstock.vn/<TICKER>/tai-chinh.htm?tab=BCTT` bằng Playwright, trích xuất các bảng `KQKD`, `CĐKT`, `CSTC`, cache JSON ở `out/vietstock_bctt/`, rồi chuẩn hoá feature theo quý với độ trễ công bố bảo thủ để phục vụ harness `scripts/analysis/evaluate_bctt_feature_lift.py`.
- **Macro factors (FRED + Stooq)** – module `scripts/data_fetching/macro_factor_cache.py` đọc cấu hình `config/macro_factors.yaml`, tải/cached chuỗi thời gian vàng/dầu/USD/VIX/Nasdaq/SP500/US10Y dưới `out/macro_factors/`, rồi `scripts/analysis/evaluate_macro_factor_sensitivity.py` dùng chúng để chấm rolling correlation/beta và phản ứng khi factor có shock mạnh.
- **Danh mục TCBS** – `scripts/scrapers/tcbs.py` mở TCInvest bằng Chrome qua Playwright, đăng nhập bằng `TCBS_USERNAME`/`TCBS_PASSWORD` rồi trích xuất bảng danh mục về đúng một file canonical `data/portfolios/portfolio.csv` (schema tối thiểu `Ticker,Quantity,AvgPrice`). Engine chỉ đọc file này, không bao giờ sửa.

Các nguồn khác mang tính thử nghiệm hoặc probe (ví dụ các script khảo sát NEXT_STEPS) không còn nằm trong đường chạy chính của engine; nếu sử dụng, hãy coi chúng là công cụ ad-hoc, không thay đổi contract output chính (`out/universe.csv`, `out/positions.csv`).

## Thành phần chính

### EngineConfig

`EngineConfig.from_yaml(path)` đọc file YAML và chuẩn hoá:

- `universe.csv` – nguồn danh sách mã + sector.
- `technical_indicators` – cấu hình SMA/RSI/ATR/MACD.
- `portfolio.directory` – thư mục chứa danh mục duy nhất (`portfolio.csv`).
- `output` – vị trí ghi các file CSV chuẩn hoá (default `out/`).
- `execution` – tham số sizing (aggressiveness, max_order_pct_adv, slice_adv_ratio, min lot, max qty/order).
- `data.history_cache` – nơi cache dữ liệu lịch sử.

Mọi đường dẫn được chuẩn hoá thành `Path.resolve()`. Thiếu trường bắt buộc sẽ raise `ConfigurationError` (fail-fast).

Lưu ý về cache/output:
- Engine chỉ xoá các **artifact** đầu ra (ví dụ `out/universe.csv`, `out/positions.csv`, `out/market_summary.json`, `out/sector_summary.csv`) giữa các lần chạy.
- Các cache dữ liệu (ví dụ `out/data/`, `out/cafef_flows/`, `out/vietstock_overview/`) được giữ lại để tránh re-fetch không cần thiết.

### VndirectMarketDataService

- `load_history(tickers)` gọi `ensure_and_load_history_df` để đảm bảo cache đầy đủ rồi trả về DataFrame hợp nhất (cột `Date,Ticker,Open,High,Low,Close,Volume,t`).
- `load_intraday(tickers)` gọi `ensure_intraday_latest_df` để lấy giá phút gần nhất. Nếu API fail, engine vẫn fallback về giá đóng cửa gần nhất.
- Có thể cung cấp `data/reference_overrides` (CSV `Ticker,Ref`) trong cấu hình để ép giá tham chiếu khi tính `bands.csv` trong các phiên có điều chỉnh tham chiếu của sàn.

### TechnicalSnapshotBuilder

- Ghép dữ liệu lịch sử và intraday, tính snapshot kỹ thuật chuẩn hoá.
- Với mỗi ticker:
  - `Last` = giá intraday nếu có, fallback `Close` cuối cùng; `Ref` = `LastClose`.
  - `ChangePct` = (Last/Ref − 1) × 100 (đơn vị %).
  - `SMA_20/50/200`, `EMA_20`, `RSI_14`, `ATR_14`, `MACD`, `MACDSignal`, `MACD_Hist`.
  - `Return_5`/`Return_20` (đơn vị %, dùng trực tiếp để xuất `Ret5d`/`Ret20d` trong `universe.csv`).
  - `ADV_20`, `Hi_252`, `Lo_252`, Z-score (`Z_20`).
  - `Sector` lấy từ universe.
- Output trung gian giữ nguyên dưới dạng DataFrame và được dùng trực tiếp để dựng `universe.csv` (không còn ghi file riêng `technical.csv`).
- Sau khi ghép snapshot kỹ thuật, engine còn tính thêm market breadth/co-movement, sector leadership, và relative strength để làm giàu `universe.csv` và tạo hai output phụ `market_summary.json` + `sector_summary.csv`.

### PortfolioReporter

- Đọc danh mục `data/portfolios/portfolio.csv` (schema: `Ticker,Quantity,AvgPrice`).
- Hợp nhất với snapshot để xác định `Last`, `Sector`, tính `MarketValue_kVND`, `CostBasis_kVND`, `Unrealized_kVND`, `PNLPct`.
- Ghi `aggregate_positions` (phục vụ merge vào `universe.csv`) và `aggregate_sector` (dùng nội bộ cho thống kê, không xuất file riêng).
- Không chạm vào file danh mục gốc; chỉ đọc.

### TCBS Scraper

- Đọc `TCBS_USERNAME` và `TCBS_PASSWORD` từ `.env` hoặc biến môi trường.
- Wrapper `broker.sh` tự nạp `.env` ở repo root trước khi dispatch các subcommand, nên cùng một nguồn biến được dùng nhất quán cho `tcbs`, `codex`, `research`, `orders`, và các flow phụ trợ khác.
- Dùng Chrome qua Playwright (`channel="chrome"`) với persistent profile tại `.playwright/tcbs-user-data/default` để giữ fingerprint/session giữa các lần chạy.
- Mở `https://tcinvest.tcbs.com.vn/home`; nếu đã đăng nhập từ phiên trước, script sẽ mở menu tài khoản, bấm **Đăng xuất** rồi mới đăng nhập lại bằng credentials để đảm bảo trạng thái sạch cho mỗi lần chạy.
- Sau khi đăng nhập, điều hướng `my-asset` → tab `Cổ phiếu` → `Tài sản` → bảng danh mục, parse dữ liệu theo header (`Mã`, `SL Tổng`/`Được GD`, `Giá vốn`) và ghi `data/portfolios/portfolio.csv`.

### Codex scripted chat

- `broker.sh codex` gọi `scripts/codex/exec_resume.py` dùng `codex exec` + `resume` (không cần TTY) trong thư mục `codex_universe/`.
- Trước khi gọi `codex exec`, wrapper sẽ xoá file cũ trong `codex_universe` nhưng giữ lại đủ input live `bundle_manifest.json`, `universe.csv`, `market_summary.json`, `sector_summary.csv`, `ml_range_predictions_full_2y.csv`, `ml_range_predictions_recent_focus.csv`, `ml_cycle_forecast_ticker_matrix.csv`, `ml_cycle_forecast_best_horizon.csv`, `ticker_playbook_best_configs.csv`, `ml_ohlc_next_session.csv`, và nếu có thì `human_notes.md`, `ml_intraday_rest_of_session.csv`, `ml_single_name_timing.csv`, `ml_entry_ladder_eval.csv`, `strategy_buckets.csv`, cùng thư mục `research/`, rồi tạo/ghi đè `total_capital_kVND.txt` (kVND) để đúng contract của prompt. Nếu repo có file `human_notes.md` thì `tcbs` cũng sync file này sang `codex_universe/human_notes.md`; nếu repo có file `strategy_buckets.csv` thì wrapper synthesize file đích và chỉ giữ các mã nằm trong active ticker filter, đồng thời chỉ auto-add `exit_only` cho các mã portfolio còn nằm trong filter nhưng chưa có trong source; nếu repo có thư mục `research/` thì wrapper copy nguyên thư mục này sang `codex_universe/research/`; sau đó wrapper build `codex_universe/bundle_manifest.json`.
- Snapshot tại `codex_universe/` được engine đồng bộ tự động sau mỗi lần chạy, nên bước thủ công copy không còn cần thiết (chỉ cần đảm bảo thư mục sạch để guard ở trên hoạt động). Thư mục này bị bỏ qua khỏi git nên `broker.sh engine` sẽ tự tạo mới nếu chưa tồn tại.
- Prompt đầu tiên được đọc từ `prompts/PROMPT.txt`; bước mở đầu của prompt hiện tự kiểm tra file/cột theo `bundle_manifest.json` và nêu mismatch ngay trong câu trả lời của bước đó, nhưng không chặn flow. Prompt đã được rút gọn để phần mô tả file/contract nằm chủ yếu trong `bundle_manifest.json`; `PROMPT.txt` giữ workflow, hard rules, execution order, và các ràng buộc chiến lược. Nếu có `human_notes.md`, `research/manifest.json`, `ml_intraday_rest_of_session.csv`, `ml_single_name_timing.csv`, hoặc `strategy_buckets.csv`, prompt vẫn phải dùng chúng theo thứ tự ưu tiên và semantics được manifest mô tả. Prompt cũng hard-code giả định phí giao dịch `0.3%` mỗi chiều để Codex đánh giá reward/risk sau phí. Working universe đã được co upstream và phản ánh sẵn trong snapshot live.
- Sau khi Codex nhận prompt gốc, wrapper gọi `codex resume` để gửi kickoff `"Tiếp tục"` nhằm “mở khoá” luồng bước‑bước của prompt.
- Giá trị **số liệu** mà prompt yêu cầu phải đọc là từ `total_capital_kVND.txt` (wrapper đã tạo/ghi đè từ `CODEX_TOTAL_CAPITAL_KVND`, hoặc parse từ `CODEX_BUDGET_TEXT` nếu không set).
- Sau kickoff, wrapper nhắn `CODEX_CONTINUE_MESSAGE` (mặc định "Tiếp tục.") liên tục cho đến khi `CODEX_DONE_FILE` (mặc định `DONE.md`) xuất hiện trong thư mục; completion chỉ được check giữa các vòng `resume`, và wrapper dừng sớm nếu đạt `CODEX_MAX_CONTINUES` (nếu bật).
- Nếu một lượt `codex exec`/`resume` không phát ra stdout/stderr quá `CODEX_IDLE_TIMEOUT_SECONDS` (mặc định `900s`), wrapper sẽ coi là silent hang, kill tiến trình CLI local đó, rồi tiếp tục bằng `resume` trên chính thread đã có thay vì đứng vô hạn.
- Ngay khi wrapper lấy được `thread_id`, nó ghi sẵn `out/codex/resume_last_codex.sh` để người vận hành có thể resume thủ công ngay cả khi phiên hiện tại còn đang dang dở.
- Script theo dõi thư mục đến khi xuất hiện file completion đã yêu cầu ở ranh giới giữa các vòng `resume`, chờ thêm `CODEX_GRACE_AFTER_CSV` giây, fail-fast nếu chưa có `orders.csv`, archive `orders.csv`, `DONE.md`, và log Codex tương ứng vào `archives/codex_runs/<timestamp>/`, rồi copy toàn bộ `codex_universe/` sang `codex_universe_history/<TCBS account>/<YYYY>/<timestamp>/` trong repo chính. Nếu bật sync lịch sử lệnh thì wrapper tiếp tục clone/pull repo riêng, copy artifact theo `TCBS account / timestamp`, commit và push sang repo đó, rồi mới in nội dung CSV ra stdout. Nếu Codex không tạo file trong thời gian `CODEX_TIMEOUT_SECONDS` / `CODEX_MAX_WALL_SECONDS` (mặc định 7200 giây) sẽ raise lỗi.

## Quy trình chạy GitHub Action

Workflow (đã gỡ tạm thời):

1. Checkout mã nguồn (fetch đầy đủ lịch sử để có thể push).
2. Cài đặt Python 3.11 và dependencies (`pip install -r requirements.txt`).
3. Chạy `python -m scripts.engine.data_engine --config config/data_engine.yaml`.
4. (Trước đây) Nếu chạy theo lịch hoặc kích hoạt thủ công, workflow sẽ commit và push những thay đổi trong `out/*.csv` và `out/diagnostics`. Khi chạy trên Pull Request, bước commit được bỏ qua để workflow chỉ dùng cho việc xem log.

Không còn workflow chạy định kỳ trong repo hiện tại. Nếu cần bật lại, thêm file YAML workflow vào `.github/workflows/`.

## Danh mục

- Danh mục duy nhất: `data/portfolios/portfolio.csv` với schema tối thiểu `Ticker,Quantity,AvgPrice`.
- Khi engine chạy, file danh mục không bị sửa; các báo cáo luôn ghi ra `out/universe.csv` và có thể được ghi đè mỗi lần chạy.

## Kiểm thử

- `tests/test_data_engine.py` tạo dữ liệu giả, chạy engine và xác minh tất cả output tồn tại.
- `tests/test_tcbs_parser.py` xác minh bộ phân tích bảng TCBS với dữ liệu giả lập (không gọi mạng/trình duyệt).
- `tests/test_evaluate_deterministic_strategies.py` kiểm tra logic chọn market gate và tổng hợp replay cho harness offline.
- `tests/test_evaluate_ml_models.py` kiểm tra chọn model tốt nhất và tổng hợp metrics cho harness ML baseline.
- `tests/test_evaluate_vnindex_models.py` kiểm tra chọn model tốt nhất và tổng hợp metrics cho harness VNINDEX ML.

## Mở rộng

- Có thể bổ sung chỉ báo mới bằng cách thêm vào `scripts/indicators/` và cập nhật `TechnicalSnapshotBuilder`.
- Nếu cần nguồn dữ liệu khác, triển khai class mới implement `MarketDataService` rồi truyền vào `DataEngine` (ví dụ trong test).
- Để đồng bộ với hệ thống khác, bạn chỉ cần đọc các CSV trong `out/` (được commit sẵn) hoặc pull nhánh mới nhất từ repo.
- Đọc `TCBS_USERNAME` và `TCBS_PASSWORD` từ `.env` hoặc biến môi trường.
