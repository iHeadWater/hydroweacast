<!--
 * @Author: Wenyu Ouyang
 * @Date: 2024-04-09 16:05:41
 * @LastEditTime: 2024-04-10 14:46:35
 * @LastEditors: Wenyu Ouyang
 * @Description: Simple description for weather forecast for hydrological modeling
 * @FilePath: /hydroweacast/README.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# hydroweacast

We try to use some state-of-the-art open-source AI-driven Weather forecast models to generate forcings for hydrological modeling.

**Note: A Linux system is required**

Basically, we will use the following models:

1. [GraphCast](https://github.com/google-deepmind/graphcast): there are 3 versions of model-params but we mainly use the first version -- source-ERA5 1979-2017 resolution-0.25, where source-ERA5 means the model is trained on ERA5 reanalysis data, and the resolution and levels represent the spatial and vertical resolution of the model.
2. [Pangu](https://github.com/198808xc/Pangu-Weather): it provides 4 versions for different forecast horizons -- 1, 3, 6, and 24 hours

## Usage

Install required packages:

```bash
mamba env create -f env-dev.yml
# for developers
conda activate hydroweacast
pip install -r requirements-dev.txt
```

Then you can try the test function:

```bash
pytest tests/test_hydroweacast.py
```

