import tempfile
import unittest
from pathlib import Path


class TestStrategyBuckets(unittest.TestCase):
    def test_build_strategy_buckets_appends_live_exit_only_rows(self):
        from scripts.codex.strategy_buckets import build_strategy_buckets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config" / "data_engine.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                "portfolio:\n"
                "  directory: data/portfolios\n",
                encoding="utf-8",
            )
            portfolio_csv = root / "data" / "portfolios" / "portfolio.csv"
            portfolio_csv.parent.mkdir(parents=True)
            portfolio_csv.write_text(
                "Ticker,Quantity,AvgPrice\n"
                "MBB,100,25.0\n"
                "VCI,200,30.0\n",
                encoding="utf-8",
            )
            source_path = root / "strategy_buckets.csv"
            source_path.write_text(
                "Ticker,StrategyBucket,AllowNewBuy,AllowAvgDown,TargetState\n"
                "MBB,invest_normal,1,1,hold_or_add\n"
                "HPG,invest_normal,1,1,hold_or_add\n",
                encoding="utf-8",
            )

            df = build_strategy_buckets(config_path, source_path)

            self.assertEqual(
                df.to_dict("records"),
                [
                    {
                        "Ticker": "MBB",
                        "StrategyBucket": "invest_normal",
                        "AllowNewBuy": 1,
                        "AllowAvgDown": 1,
                        "TargetState": "hold_or_add",
                    },
                    {
                        "Ticker": "HPG",
                        "StrategyBucket": "invest_normal",
                        "AllowNewBuy": 1,
                        "AllowAvgDown": 1,
                        "TargetState": "hold_or_add",
                    },
                    {
                        "Ticker": "VCI",
                        "StrategyBucket": "exit_only",
                        "AllowNewBuy": 0,
                        "AllowAvgDown": 0,
                        "TargetState": "exit_all",
                    },
                ],
            )

    def test_build_strategy_buckets_fails_on_missing_columns(self):
        from scripts.codex.strategy_buckets import build_strategy_buckets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config" / "data_engine.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("portfolio:\n  directory: data/portfolios\n", encoding="utf-8")
            source_path = root / "strategy_buckets.csv"
            source_path.write_text(
                "Ticker,StrategyBucket,AllowNewBuy,TargetState\n"
                "MBB,invest_normal,1,hold_or_add\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                build_strategy_buckets(config_path, source_path)

    def test_build_strategy_buckets_respects_active_ticker_filter(self):
        from scripts.codex.strategy_buckets import build_strategy_buckets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config" / "data_engine.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                "universe:\n"
                "  core_tickers:\n"
                "    - MBB\n"
                "    - HPG\n"
                "    - NVL\n"
                "portfolio:\n"
                "  directory: data/portfolios\n",
                encoding="utf-8",
            )
            portfolio_csv = root / "data" / "portfolios" / "portfolio.csv"
            portfolio_csv.parent.mkdir(parents=True)
            portfolio_csv.write_text(
                "Ticker,Quantity,AvgPrice\n"
                "MBB,100,25.0\n"
                "VCI,200,30.0\n",
                encoding="utf-8",
            )
            source_path = root / "strategy_buckets.csv"
            source_path.write_text(
                "Ticker,StrategyBucket,AllowNewBuy,AllowAvgDown,TargetState\n"
                "MBB,invest_normal,1,1,hold_or_add\n"
                "HPG,invest_normal,1,1,hold_or_add\n"
                "VCI,exit_only,0,0,exit_all\n",
                encoding="utf-8",
            )

            df = build_strategy_buckets(config_path, source_path)

            self.assertEqual(
                df.to_dict("records"),
                [
                    {
                        "Ticker": "MBB",
                        "StrategyBucket": "invest_normal",
                        "AllowNewBuy": 1,
                        "AllowAvgDown": 1,
                        "TargetState": "hold_or_add",
                    },
                    {
                        "Ticker": "HPG",
                        "StrategyBucket": "invest_normal",
                        "AllowNewBuy": 1,
                        "AllowAvgDown": 1,
                        "TargetState": "hold_or_add",
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
