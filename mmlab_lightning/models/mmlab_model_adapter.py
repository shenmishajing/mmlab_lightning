from abc import ABC
from typing import Any, Optional

from lightning_template.models import LightningModule
from mmengine import MessageHub
from mmengine.model import BaseModule


class MMLabModelAdapter(LightningModule, BaseModule, ABC):
    def __init__(self, visualizer_kwargs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if visualizer_kwargs is None:
            self.visualizer_kwargs = {}
        else:
            self.visualizer_kwargs = visualizer_kwargs

    def _dump_init_info(self, *args, **kwargs):
        pass

    def set_data_preprocessor_device(self):
        self.model.data_preprocessor.to(self.device)

    def on_fit_start(self):
        message_hub = MessageHub.get_current_instance()
        message_hub.update_info(
            "epoch", self.trainer.fit_loop.epoch_progress.current.completed
        )
        message_hub.update_info("iter", self.trainer.global_step)
        message_hub.update_info("max_epochs", self.trainer.max_epochs)
        message_hub.update_info("max_iters", self.trainer.max_steps)
        self.set_data_preprocessor_device()
        self.init_weights()

    def on_validation_start(self):
        self.set_data_preprocessor_device()

    def on_test_start(self):
        self.set_data_preprocessor_device()

    def on_predict_start(self):
        self.set_data_preprocessor_device()

    def forward(self, batch, mode="loss"):
        self.batch_size = len(batch["inputs"])
        batch = self.model.data_preprocessor(batch, mode != "predict")
        return self.model._run_forward(batch, mode=mode)

    def forward_step(self, batch, *args, split="val", **kwargs):
        outputs = self(batch, mode="predict")
        self.trainer.datamodule.evaluators[split].process(outputs, batch)
        return outputs

    def on_forward_epoch_end(self, *args, split="val", **kwargs):
        log_vars = self.trainer.datamodule.evaluators[split].evaluate(
            len(self.trainer.datamodule.datasets[split])
        )
        self.log_dict(self.flatten_dict(log_vars, split), sync_dist=True)
        return log_vars

    def on_train_epoch_start(self) -> None:
        message_hub = MessageHub.get_current_instance()
        message_hub.update_info(
            "epoch", self.trainer.fit_loop.epoch_progress.current.completed
        )
        return super().on_train_epoch_start()

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        message_hub = MessageHub.get_current_instance()
        message_hub.update_info("iter", self.trainer.global_step)
        return super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch, *args, **kwargs):
        _, log_vars = self.model.parse_losses(self(batch))
        self.log_dict(self.flatten_dict(log_vars))
        return log_vars

    def predict_forward(self, batch, *args, **kwargs):
        predict_result = {"predict_outputs": self(batch, mode="predict")}

        batch = self.model.data_preprocessor(batch)
        if isinstance(batch, (list, tuple)):
            batch = {"inputs": batch[0], "data_samples": batch[1]}
        predict_result.update(batch)

        return predict_result
