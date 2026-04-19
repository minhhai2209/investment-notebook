from __future__ import annotations

import unittest

from scripts.data_fetching.vietstock_industry import (
    build_company_profile_url,
    parse_vietstock_sector_levels,
    select_vietstock_sector,
)


class VietstockIndustryParsingTest(unittest.TestCase):
    def test_build_company_profile_url_normalizes_ticker(self) -> None:
        self.assertEqual(
            build_company_profile_url("dpm"),
            "https://finance.vietstock.vn/DPM/ho-so-doanh-nghiep.htm",
        )

    def test_parse_vietstock_sector_levels_handles_minified_html(self) -> None:
        html = """
        <div class="m-b-xs sector-level">
          <span class=text>Ng&#224;nh: </span>
          <h3 class=title-x><a class="title-link text-bold" href=/nganh/15-nguyen-vat-lieu-.htm>Nguy&#234;n vật liệu</a></h3>
          <h3 class=title-x>/ <a class="title-link text-bold" href=/nganh/1510-nguyen-vat-lieu-.htm>Nguy&#234;n vật liệu</a></h3>
          <h3 class=title-x>/ <a class="title-link text-bold" href=/nganh/151010-hoa-chat-.htm>H&#243;a chất</a></h3>
        </div>
        """
        self.assertEqual(
            parse_vietstock_sector_levels(html),
            ["Nguyên vật liệu", "Nguyên vật liệu", "Hóa chất"],
        )

    def test_select_vietstock_sector_modes(self) -> None:
        levels = ["Nguyên vật liệu", "Nguyên vật liệu", "Hóa chất"]
        self.assertEqual(select_vietstock_sector(levels, "top"), "Nguyên vật liệu")
        self.assertEqual(select_vietstock_sector(levels, "leaf"), "Hóa chất")
        self.assertEqual(
            select_vietstock_sector(levels, "path"),
            "Nguyên vật liệu / Nguyên vật liệu / Hóa chất",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
