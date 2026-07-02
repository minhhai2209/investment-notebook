# Start Here

Repo này chỉ chuẩn bị dữ liệu số cho ChatGPT đọc nhanh.

## Lệnh Nhanh

```bash
./broker.sh hub
```

Sau đó đọc:

1. `data-hub/latest/manifest.json`
2. `data-hub/latest/latest_metrics.csv`
3. `data-hub/latest/api_catalog.csv`
4. `data-hub/latest/calculation_catalog.csv`
5. `data-hub/latest/market/cross_section_latest.csv`
6. `data-hub/latest/market/breadth_daily.csv`
7. `data-hub/latest/daily/{TICKER}.csv`
8. `data-hub/latest/intraday/minute_profile/{TICKER}.csv`

## Khi Cần Refresh API

```bash
./broker.sh collect
```

Nguồn nào được refresh nằm trong `config/data_hub.yaml`.

Nếu cần financial statement quarterly cache:

```bash
./broker.sh refresh_bctt
./broker.sh hub
```

## Quy Tắc

- Không tìm tin tức.
- Không dựng model dự báo trong repo.
- Không đưa khuyến nghị mua/bán.
- Nếu thiếu cache thì nói thiếu dữ liệu, không tự suy diễn.
- Mọi output chính phải là dữ liệu số có thể audit bằng CSV/JSON.
