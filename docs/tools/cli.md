## Interface

`mmlab_lightnning` provides serval cli scripts for you to use. Generally, you can run scripts under `mmlab_lightnning/tools` by `python -m mmlab_lightnning.tools.<path.to.scripts>`, for example, use `python -m mmlab_lightnning.tools.config.get_mmconfig` to run `tools/config/get_mmconfig.py`. For convenience, we also provide some aliases for these scripts. All of them are as follows:

| alias        | scripts                                         |
| ------------ | ----------------------------------------------- |
| get_mmconfig | lightning_template/tools/config/get_mmconfig.py |

We recommmand you to use `alias` method to luanch scripts in terminal, and use `module` method to debug scripts in `vscode` etc.

For commands and options, you can get all available options and commands with `help` arguement, for example `get_mmconfig --help`.

## Get mmconfig

`mmlab_lightnning` provide a cli script, called `get_mmconfig`. You can use it to download the config files from mmlab repos, 

For example, if you want to download the whole config of `faster-rcnn_r50_fpn_1x_coco`, just use command as follow, by default it will download the config file, convert it to yaml, and save it to `config.yaml`:

```bash
get_mmconfig mmdet::_base_/models/faster-rcnn_r50_fpn_1x_coco.py
```

If you want to download the model config of `faster-rcnn_r50_fpn` and save it to `configs/models/faster-rcnn/faster-rcnn_r50_fpn.yaml`, use command as follow:

```bash
get_mmconfig mmdet::_base_/models/faster-rcnn_r50_fpn.py --save-path configs/models/faster-rcnn/faster-rcnn_r50_fpn.yaml
```

See [mmengine doc](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html#id13) for more details.
