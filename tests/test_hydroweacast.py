"""
Author: Wenyu Ouyang
Date: 2024-04-09 16:31:35
LastEditTime: 2024-05-05 16:39:12
LastEditors: Wenyu Ouyang
Description: Tests for `hydroweacast` package.
FilePath: /hydroweacast/tests/test_hydroweacast.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest

from hydroweacast.run_graphcast import main


def test_main():
    main()
