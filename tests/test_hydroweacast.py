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
from hydroweacast.run_graphcast import GraphCast

def test_gc():
    gc = GraphCast()
    gc.param_path = 'params_GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz'
    gc.dataset_path = 'datasource_2024_steps_01.nc'
    gc.run()
