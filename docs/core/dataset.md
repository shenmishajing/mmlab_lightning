## Introduction

A dataset adapter for datasets from mmlab repos to use them in [project-template](https://github.com/shenmishajing/project_template)

## Arguments

Compare to the base [DataModule](https://github.com/shenmishajing/lightning_template/blob/main/lightning_template/datasets/base.py) from [lightning-template](https://github.com/shenmishajing/lightning_template), it has two additional keys, called `evaluator_cfg` and `visualizer_cfg`. All of them support [deep update](https://github.com/shenmishajing/lightning_template/blob/main/docs/configs/deep_update.md) feature, which means they can work as `dataset_cfg` and `dataloader_cfg` in base [DataModule](https://github.com/shenmishajing/lightning_template/blob/main/docs/core/dataset.md#arguments-and-config)

### evaluator config

The `evaluator_cfg` param should indicate `mmengine.evaluator.Evaluator` objects for all splits, and use the `metrics` arguement to specify the `metrics` to run, which means it should be a dict with keys `train`, `val`, `test` and `predict`, and values as a dict follow lightning CLI instantiate_class arguments format (for more details, see [arguments with class type doc](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_advanced_3.html#trainer-callbacks-and-arguments-with-class-type)). But, you can use the [Deep update between split](https://github.com/shenmishajing/lightning_template/blob/main/docs/core/dataset.md#deep-update-between-split) feature to simplify this. Check the doc, if you want more details.

### visualizer config

The `visualizer_cfg` param should indicate `mmengine.visualization.visualizer` or objects inherited from it for all splits. But, generally, you only need specify just one `visualizer` for all splits.
