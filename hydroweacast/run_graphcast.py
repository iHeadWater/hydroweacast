import dataclasses
import datetime
import functools

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast.config import ROOT_DIR
from graphcast import xarray_jax
from graphcast import xarray_tree

import haiku as hk
import jax
import numpy as np
import xarray
import time
import datetime
import hydrodatasource.configs.config as conf
import logging
from line_profiler import profile
import warnings
warnings.filterwarnings("ignore")


# @profile
def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig
):
    """Constructs and wraps the GraphCast Predictor."""
    with open(ROOT_DIR + f"dm_graphcast/stats/diffs_stddev_by_level.nc","rb",) as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open(ROOT_DIR + f"dm_graphcast/stats/mean_by_level.nc","rb",) as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open(ROOT_DIR + f"dm_graphcast/stats/stddev_by_level.nc","rb",) as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


def with_configs(fn, model_config, task_config):
    return functools.partial(fn, model_config=model_config, task_config=task_config)
def with_params(fn, params, state):
    return functools.partial(fn, params=params, state=state)
def drop_state(fn):
    return lambda **kw: fn(**kw)[0]


class GraphCast:
    def __init__(self):
        self.param_path = ""
        self.dataset_path = ""
        self.predict_path = ""
        self.eval_num = 1

    def run(self):
        # 配置日志记录器
        logging.basicConfig(
            filename="hydro_log/hydro.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        start_time = time.time()
        with open(
            ROOT_DIR + f"dm_graphcast/params/{self.param_path}",
            "rb",
        ) as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
        params = ckpt.params
        state = {}
        model_config = ckpt.model_config
        task_config = ckpt.task_config
        first_part_end_time = time.time()
        first_part_duration = first_part_end_time - start_time
        print("加载模型运行时间: {:.2f} 秒".format(first_part_duration))
        print("Model config:", model_config)
        
        with open(
            ROOT_DIR + f"dm_graphcast/dataset/{self.dataset_path}",
            "rb",
        ) as f:
            example_batch = xarray.load_dataset(f).compute()
        assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets
        second_part_end_time = time.time()
        second_part_duration = second_part_end_time - first_part_end_time
        print("加载数据运行时间: {:.2f} 秒".format(second_part_duration))
        logging.info("加载数据运行时间: {:.2f} 秒".format(second_part_duration))
        
        eval_steps = int(input("1 to " + str(example_batch.sizes["time"] - 2) + ": "))
        print("eval steps: {:.0f} ".format(eval_steps))
        logging.info("eval steps: {:.0f} ".format(eval_steps))
        eval_inputs, eval_targets, eval_forcings = (
            data_utils.extract_inputs_targets_forcings(
                example_batch,
                target_lead_times=slice("6h", f"{eval_steps*6}h"),
                **dataclasses.asdict(task_config),
            )
        )
        third_part_end_time = time.time()
        third_part_duration = third_part_end_time - second_part_end_time
        print("运行时间: {:.2f} 秒".format(third_part_duration))
        logging.info("运行时间: {:.2f} 秒".format(third_part_duration))
        
        run_forward_jitted = drop_state(
            with_params(
                jax.jit(with_configs(run_forward.apply, model_config, task_config)),
                params,
                state,
            )
        )
        assert model_config.resolution in (0, 360.0 / eval_inputs.sizes["lon"]), (
            "Model resolution doesn't match the data resolution. You likely want to "
            "re-filter the dataset list, and download the correct data."
        )
        predictions = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=eval_inputs,
            targets_template=eval_targets * np.nan,
            forcings=eval_forcings,
        )
        fourth_part_end_time = time.time()
        fourth_part_duration = fourth_part_end_time - third_part_end_time
        print("推理预测运行时间: {:.2f} 秒".format(fourth_part_duration))
        logging.info("推理预测运行时间: {:.2f} 秒".format(fourth_part_duration))
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{time_str}.nc"
        predictions.to_netcdf(filename)
        fifth_part_end_time = time.time()
        fifth_part_duration = fifth_part_end_time - fourth_part_end_time
        print("预测结果存储本地运行时间: {:.2f} 秒".format(fifth_part_duration))
        logging.info("预测结果存储本地运行时间: {:.2f} 秒".format(fifth_part_duration))
        conf.FS.put_file(filename, f"s3://foreign-model-predictions/{filename}")
        # os.remove(filename)
        print(f"{filename} saved!")
        sixth_part_end_time = time.time()
        sixth_part_duration = sixth_part_end_time - fifth_part_end_time
        print("预测结果存储minio运行时间: {:.2f} 秒".format(sixth_part_duration))
        logging.info("预测结果存储minio运行时间: {:.2f} 秒".format(sixth_part_duration))
        
        end_time = time.time()
        total_duration = end_time - start_time
        print("总运行时间: {:.2f} 秒".format(total_duration))
        logging.info("总运行时间: {:.2f} 秒".format(total_duration))
        return predictions

