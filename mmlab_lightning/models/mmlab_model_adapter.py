from abc import ABC
from typing import Any

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

    def forward(self, batch, *args, mode="loss", **kwargs):
        self.batch_size = len(batch["inputs"])
        batch = self.model.data_preprocessor(batch, mode != "predict")
        return self.model._run_forward(batch, mode=mode)

    def _loss_step(self, batch, output, *args, **kwargs):
        _, log_dict = self.model.parse_losses(output)
        return log_dict

    def metric_step(self, batch, output, *args, split="val", **kwargs):
        if self.evaluators[split]:
            self.evaluators[split].process(output, batch)

    def on_metric_epoch_end(self, *args, split="val", **kwargs):
        return self.evaluators[split].evaluate(len(self.datasets[split]))

    def forward_step(self, batch, *args, split="val", **kwargs):
        return super().forward_step(
            batch,
            *args,
            split=split,
            mode="loss" if split == "train" else "predict",
            **kwargs
        )

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

    def predict_forward(self, batch, *args, **kwargs):
        predict_result = {"predict_outputs": self(batch, mode="predict")}

        batch = self.model.data_preprocessor(batch)
        if isinstance(batch, (list, tuple)):
            batch = {"inputs": batch[0], "data_samples": batch[1]}
        predict_result.update(batch)

        return predict_result
