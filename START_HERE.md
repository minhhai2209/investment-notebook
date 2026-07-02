# Start Here

Repo này chỉ chuẩn bị dữ liệu số cho ChatGPT đọc nhanh.

## Lệnh Nhanh

```bash
./broker.sh hub
```

Sau đó đọc:

1. `data-hub/latest/START_HERE.json`
2. `data-hub/latest/bundles/source_audit.csv`
3. `data-hub/latest/bundles/market_snapshot.csv`
4. `data-hub/latest/bundles/symbol_latest.csv`
5. `data-hub/latest/index/ticker_catalog.csv`
6. `data-hub/latest/index/file_catalog.csv`
7. `data-hub/latest/manifest.json`

Khi cần soi sâu một mã, dùng `index/ticker_catalog.csv` để lấy đường dẫn `daily/{TICKER}.csv` và `intraday/minute_profile/{TICKER}.csv`.

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
