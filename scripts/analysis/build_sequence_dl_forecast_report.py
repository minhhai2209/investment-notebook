from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from scripts.analysis.evaluate_ohlc_models import (
    FEATURE_COLUMNS,
    build_multi_ticker_sample,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI environment checks.
    raise SystemExit(
        "PyTorch is required for true LSTM/Transformer sequence models. "
        "Install requirements first: ./broker.sh setup or pip install -r requirements.txt"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = REPO_ROOT / "out" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "analysis"
DEFAULT_TICKERS = ("VIC", "VHM")
DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 15, 20)
VN_TZ = timezone(timedelta(hours=7))


@dataclass(frozen=True)
class Normalizer:
    median: np.ndarray
    scale: np.ndarray
    target_mean: float
    target_scale: float


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(values)
        return self.head(output[:, -1, :]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, input_size: int, sequence_length: int, hidden_size: int, heads: int) -> None:
        super().__init__()
        usable_heads = max(1, min(int(heads), int(hidden_size)))
        while hidden_size % usable_heads != 0 and usable_heads > 1:
            usable_heads -= 1
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=usable_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(values) + self.position[:, : values.shape[1], :]
        encoded = self.encoder(encoded)
        return self.head(encoded[:, -1, :]).squeeze(-1)


def _normalise_ticker(value: object) -> str:
    return str(value or "").strip().upper()


def _parse_csv_values(values: Sequence[str] | None, default: Sequence[str]) -> List[str]:
    if not values:
        return [str(item) for item in default]
    parsed: List[str] = []
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if item:
                parsed.append(item)
    return parsed


def _parse_horizons(values: Sequence[str] | None) -> List[int]:
    horizons: List[int] = []
    for value in _parse_csv_values(values, [str(item) for item in DEFAULT_HORIZONS]):
        horizon = int(value)
        if horizon <= 0:
            raise ValueError("horizons must be positive integers")
        if horizon not in horizons:
            horizons.append(horizon)
    return sorted(horizons)


def _sequence_arrays(scoped: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scoped = scoped.sort_values("Date").reset_index(drop=True)
    features = scoped[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    target = scoped["TargetCloseRetPct"].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(scoped["Date"]).to_numpy()
    base_close = scoped["BaseClose"].to_numpy(dtype=np.float32)
    sequences: List[np.ndarray] = []
    targets: List[float] = []
    sequence_dates: List[np.datetime64] = []
    sequence_base_close: List[float] = []
    for end_idx in range(int(sequence_length) - 1, len(scoped)):
        sequences.append(features[end_idx - int(sequence_length) + 1 : end_idx + 1])
        targets.append(float(target[end_idx]) if not math.isnan(float(target[end_idx])) else np.nan)
        sequence_dates.append(dates[end_idx])
        sequence_base_close.append(float(base_close[end_idx]))
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(sequence_dates),
        np.asarray(sequence_base_close, dtype=np.float32),
    )


def _fit_normalizer(train_x: np.ndarray, train_y: np.ndarray) -> Normalizer:
    flat = train_x.reshape(-1, train_x.shape[-1])
    median = np.nanmedian(flat, axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    filled = np.where(np.isfinite(flat), flat, median)
    scale = np.nanstd(filled, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    target_mean = float(np.nanmean(train_y))
    target_scale = float(np.nanstd(train_y))
    if not math.isfinite(target_mean):
        target_mean = 0.0
    if not math.isfinite(target_scale) or target_scale <= 1e-6:
        target_scale = 1.0
    return Normalizer(median=median, scale=scale, target_mean=target_mean, target_scale=target_scale)


def _apply_normalizer(values: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    filled = np.where(np.isfinite(values), values, normalizer.median)
    return ((filled - normalizer.median) / normalizer.scale).astype(np.float32)


def _make_model(model_name: str, input_size: int, sequence_length: int, hidden_size: int, transformer_heads: int) -> nn.Module:
    if model_name == "lstm":
        return LSTMRegressor(input_size=input_size, hidden_size=hidden_size)
    if model_name == "transformer":
        return TransformerRegressor(
            input_size=input_size,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            heads=transformer_heads,
        )
    raise ValueError(f"Unsupported sequence model: {model_name}")


def _train_predict(
    *,
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    sequence_length: int,
    hidden_size: int,
    transformer_heads: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    normalizer = _fit_normalizer(train_x, train_y)
    x_train = _apply_normalizer(train_x, normalizer)
    y_train = ((train_y.astype(np.float32) - normalizer.target_mean) / normalizer.target_scale).astype(np.float32)
    x_predict = _apply_normalizer(predict_x, normalizer)

    model = _make_model(
        model_name,
        input_size=x_train.shape[-1],
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        transformer_heads=transformer_heads,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    for _ in range(int(epochs)):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x_predict)).detach().cpu().numpy()
    return (pred_scaled * normalizer.target_scale) + normalizer.target_mean


def _direction_hit(actual: pd.Series, predicted: pd.Series) -> float:
    if actual.empty:
        return float("nan")
    return float((np.sign(actual.astype(float)) == np.sign(predicted.astype(float))).mean() * 100.0)


def _evaluate_one_scope(
    *,
    scoped: pd.DataFrame,
    ticker: str,
    horizon: int,
    model_name: str,
    sequence_length: int,
    min_train_sequences: int,
    holdout_dates: int,
    retrain_every: int,
    hidden_size: int,
    transformer_heads: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    sequences, targets, dates, base_closes = _sequence_arrays(scoped, sequence_length)
    labeled_mask = np.isfinite(targets)
    labeled_dates = list(pd.Index(sorted(pd.to_datetime(dates[labeled_mask]).unique())))
    if len(labeled_dates) <= min_train_sequences + 2:
        raise RuntimeError(f"Not enough sequence dates for {ticker} T+{horizon}: {len(labeled_dates)}")

    eval_dates = labeled_dates[-int(holdout_dates) :]
    history_rows: List[pd.DataFrame] = []
    for start in range(0, len(eval_dates), int(retrain_every)):
        block_dates = eval_dates[start : start + int(retrain_every)]
        block_first = np.datetime64(block_dates[0])
        block_values = np.asarray(block_dates, dtype="datetime64[ns]")
        train_mask = labeled_mask & (dates < block_first)
        block_mask = labeled_mask & np.isin(dates.astype("datetime64[ns]"), block_values)
        if int(train_mask.sum()) < int(min_train_sequences) or not block_mask.any():
            continue
        predictions = _train_predict(
            model_name=model_name,
            train_x=sequences[train_mask],
            train_y=targets[train_mask],
            predict_x=sequences[block_mask],
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            transformer_heads=transformer_heads,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
        history_rows.append(
            pd.DataFrame(
                {
                    "Date": pd.to_datetime(dates[block_mask]),
                    "Ticker": ticker,
                    "Horizon": int(horizon),
                    "ForecastWindow": f"T+{horizon}",
                    "Model": model_name,
                    "ActualCloseRetPct": targets[block_mask],
                    "PredCloseRetPct": predictions,
                    "BaseClose": base_closes[block_mask],
                }
            )
        )

    if not history_rows:
        raise RuntimeError(f"No walk-forward rows for {ticker} T+{horizon} {model_name}")
    history = pd.concat(history_rows, ignore_index=True)
    close_mae = float(np.mean(np.abs(history["ActualCloseRetPct"] - history["PredCloseRetPct"])))
    dir_hit = _direction_hit(history["ActualCloseRetPct"], history["PredCloseRetPct"])

    latest_idx = len(dates) - 1
    all_train_mask = labeled_mask
    current_prediction = _train_predict(
        model_name=model_name,
        train_x=sequences[all_train_mask],
        train_y=targets[all_train_mask],
        predict_x=sequences[latest_idx : latest_idx + 1],
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        transformer_heads=transformer_heads,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )[0]
    current_base = float(base_closes[latest_idx])
    forecast_close = current_base * (1.0 + (float(current_prediction) / 100.0))
    current = {
        "Ticker": ticker,
        "Horizon": int(horizon),
        "ForecastWindow": f"T+{horizon}",
        "Model": model_name,
        "ModelFamily": "DL-sequence",
        "ModelClass": "LSTMRegressor" if model_name == "lstm" else "TransformerEncoderRegressor",
        "EvalRows": int(history.shape[0]),
        "CloseMAEPct": round(close_mae, 4),
        "CloseDirHitPct": round(dir_hit, 2),
        "SequenceLength": int(sequence_length),
        "Epochs": int(epochs),
        "BaseClose": round(current_base, 4),
        "ForecastCloseRetPct": round(float(current_prediction), 4),
        "ForecastClose": round(float(forecast_close), 4),
        "SelectionScore": round(float(close_mae - (0.01 * dir_hit)), 4),
    }
    return history, current


def build_sequence_dl_report(
    *,
    tickers: Sequence[str],
    horizons: Sequence[int],
    history_dir: Path,
    output_dir: Path,
    models: Sequence[str],
    sequence_length: int,
    min_train_sequences: int,
    holdout_dates: int,
    retrain_every: int,
    hidden_size: int,
    transformer_heads: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, Path]:
    tickers = [_normalise_ticker(ticker) for ticker in tickers]
    sample = build_multi_ticker_sample(tickers, history_dir, max_horizon=max(horizons))
    current_rows: List[Dict[str, object]] = []
    history_frames: List[pd.DataFrame] = []
    errors: List[Dict[str, object]] = []
    for ticker in tickers:
        for horizon in horizons:
            scoped = sample[(sample["Ticker"] == ticker) & (sample["Horizon"] == int(horizon))].copy()
            for model_name in models:
                try:
                    history, current = _evaluate_one_scope(
                        scoped=scoped,
                        ticker=ticker,
                        horizon=int(horizon),
                        model_name=model_name,
                        sequence_length=sequence_length,
                        min_train_sequences=min_train_sequences,
                        holdout_dates=holdout_dates,
                        retrain_every=retrain_every,
                        hidden_size=hidden_size,
                        transformer_heads=transformer_heads,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        seed=seed,
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "Ticker": ticker,
                            "Horizon": int(horizon),
                            "Model": model_name,
                            "Error": str(exc),
                        }
                    )
                    continue
                current_rows.append(current)
                history_frames.append(history)

    if not current_rows:
        raise RuntimeError(f"No sequence DL forecast rows were produced. Errors: {errors}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(VN_TZ).isoformat()
    forecast_df = pd.DataFrame(current_rows).sort_values(["Ticker", "Horizon", "SelectionScore"]).reset_index(drop=True)
    forecast_df.insert(0, "SnapshotDate", generated_at)
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    metrics_df = forecast_df[
        [
            "SnapshotDate",
            "Ticker",
            "Horizon",
            "ForecastWindow",
            "Model",
            "ModelFamily",
            "ModelClass",
            "EvalRows",
            "CloseMAEPct",
            "CloseDirHitPct",
            "SelectionScore",
            "SequenceLength",
            "Epochs",
        ]
    ].copy()
    best_df = forecast_df.sort_values(["Ticker", "Horizon", "SelectionScore"]).groupby(["Ticker", "Horizon"], as_index=False).head(1)

    forecast_path = output_dir / "ml_sequence_dl_forecasts.csv"
    metrics_path = output_dir / "ml_sequence_dl_model_metrics.csv"
    history_path = output_dir / "ml_sequence_dl_prediction_history.csv"
    best_path = output_dir / "ml_sequence_dl_best_by_horizon.csv"
    summary_path = output_dir / "ml_sequence_dl_summary.json"
    forecast_df.to_csv(forecast_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    history_df.to_csv(history_path, index=False)
    best_df.to_csv(best_path, index=False)
    summary = {
        "SchemaVersion": 1,
        "GeneratedAt": generated_at,
        "Tickers": tickers,
        "Horizons": [int(item) for item in horizons],
        "Models": list(models),
        "SequenceLength": int(sequence_length),
        "Epochs": int(epochs),
        "Artifacts": {
            "Forecasts": str(forecast_path),
            "Metrics": str(metrics_path),
            "PredictionHistory": str(history_path),
            "BestByHorizon": str(best_path),
        },
        "Errors": errors,
        "BestRows": best_df.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "forecast": forecast_path,
        "metrics": metrics_path,
        "history": history_path,
        "best": best_path,
        "summary": summary_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build true LSTM/Transformer sequence forecasts for ticker close returns.")
    parser.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS), help="Tickers, comma-separated or space-separated.")
    parser.add_argument("--horizons", nargs="*", default=[str(item) for item in DEFAULT_HORIZONS], help="Horizons, comma-separated or space-separated.")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="*", default=["lstm", "transformer"], help="Models: lstm transformer.")
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--min-train-sequences", type=int, default=160)
    parser.add_argument("--holdout-dates", type=int, default=20)
    parser.add_argument("--retrain-every", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [item.lower() for item in _parse_csv_values(args.models, ["lstm", "transformer"])]
    allowed = {"lstm", "transformer"}
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    paths = build_sequence_dl_report(
        tickers=_parse_csv_values(args.tickers, DEFAULT_TICKERS),
        horizons=_parse_horizons(args.horizons),
        history_dir=args.history_dir,
        output_dir=args.output_dir,
        models=models,
        sequence_length=int(args.sequence_length),
        min_train_sequences=int(args.min_train_sequences),
        holdout_dates=int(args.holdout_dates),
        retrain_every=int(args.retrain_every),
        hidden_size=int(args.hidden_size),
        transformer_heads=int(args.transformer_heads),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
    )
    for path in paths.values():
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
