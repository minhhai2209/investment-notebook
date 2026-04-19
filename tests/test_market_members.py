from __future__ import annotations

import unittest
from unittest import mock

from scripts.data_fetching.market_members import (
    fetch_hose_members,
    extract_vndirect_stock_codes,
    parse_investing_vn100_members,
    parse_vietstock_board_symbols,
)


class MarketMembersParsingTest(unittest.TestCase):
    def test_parse_vietstock_board_symbols(self) -> None:
        html = '<tr data-symbol="AAA"></tr><tr data-symbol="BBB"></tr>'
        self.assertEqual(parse_vietstock_board_symbols(html), {"AAA", "BBB"})

    def test_parse_investing_vn100_members(self) -> None:
        html = """
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"state":{"assetsCollectionStore":{"assetsCollection":{"_collection":[
          {"symbol":"AAA"},{"symbol":"BBB"}
        ]}}}}}}
        </script>
        """
        self.assertEqual(parse_investing_vn100_members(html), {"AAA", "BBB"})

    def test_extract_vndirect_stock_codes_filters_floor(self) -> None:
        payload = {
            "data": [
                {"code": "AAA", "floor": "HOSE"},
                {"code": "BBB", "floor": "HNX"},
                {"code": "CCC", "floor": "HOSE"},
            ]
        }
        self.assertEqual(extract_vndirect_stock_codes(payload, floor="HOSE"), {"AAA", "CCC"})

    def test_fetch_hose_members_uses_vndirect_api(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [
                {"code": "AAA", "floor": "HOSE"},
                {"code": "BBB", "floor": "HOSE"},
                {"code": "CCC", "floor": "HNX"},
            ]
        }

        with mock.patch("scripts.data_fetching.market_members.requests.get", return_value=response):
            self.assertEqual(fetch_hose_members(), {"AAA", "BBB"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
