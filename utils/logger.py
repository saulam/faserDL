import re
from typing import Any, Dict, Optional
import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger, TensorBoardLogger


class SplitTensorBoardLogger(Logger):
    """Route metrics ending in '/train' to one TB run and '/val' to another.
    Everything else can either go to 'train', 'val', or both (configurable).
    """

    def __init__(
        self,
        save_dir: str = "tb_logs",
        name: str = "exp",
        version: Optional[str] = None,
        train_run_suffix: str = "train",
        val_run_suffix: str = "val",
        other_target: str = "train",   # 'train' | 'val' | 'both'
        strip_suffix: bool = True,     # strip '/train' or '/val' from the tag before writing
        val_suffix: str = "",          # when writing val metrics, add this suffix to avoid clashing with train
     ):
        super().__init__()
        self.train_logger = TensorBoardLogger(save_dir, name=f"{name}_{train_run_suffix}", version=version)
        self.val_logger = TensorBoardLogger(save_dir, name=f"{name}_{val_run_suffix}", version=version)
        self.other_target = other_target
        self.strip_suffix = strip_suffix
        self.val_suffix = val_suffix
        self._suffix_re = re.compile(r"^(?P<prefix>.+)/(?P<split>train|val)(?P<agg>_(?:step|epoch))?$")

    @property
    def name(self) -> str:
        return f"{self.train_logger.name}+{self.val_logger.name}"

    @property
    def version(self) -> str:
        # Keep both versions in the path; TB will just use the directories
        return str(self.train_logger.version)

    @property
    def save_dir(self) -> Optional[str]:
        return self.train_logger.save_dir

    @property
    def experiment(self):
        # Default experiment handle (e.g., for add_graph). Choose train by default.
        return self.train_logger.experiment

    @property
    def train_experiment(self):
        return self.train_logger.experiment

    @property
    def val_experiment(self):
        return self.val_logger.experiment

    def log_hyperparams(self, params: Any) -> None:
        # Write hparams to both runs so they’re self-contained
        self.train_logger.log_hyperparams(params)
        self.val_logger.log_hyperparams(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        train_metrics, val_metrics, others_for_train, others_for_val = {}, {}, {}, {}

        for k, v in metrics.items():
            m = self._suffix_re.match(k)
            if m:
                prefix = m.group("prefix")       # e.g., "loss_total"
                split  = m.group("split")        # "train" or "val"
                agg    = m.group("agg") or ""    # "", "_step", or "_epoch"

                # Decide final tag
                if self.strip_suffix:
                    # strip '/train' or '/val' but optionally keep "_step"/"_epoch"
                    final_key = prefix + agg
                else:
                    # keep full original key
                    final_key = k

                if split == "train":
                    train_metrics[final_key] = v
                else:
                    val_metrics[final_key + self.val_suffix] = v  # adding suffix to overlap with train
            else:
                # no '/train' or '/val' — route per policy
                if self.other_target == "train":
                    others_for_train[k] = v
                elif self.other_target == "val":
                    others_for_val[k] = v
                elif self.other_target == "both":
                    # write to both runs below
                    others_for_train[k] = v
                    others_for_val[k] = v

        if others_for_train:
            self.train_logger.log_metrics(others_for_train, step=step)
        if others_for_val:
            self.val_logger.log_metrics(others_for_val, step=step)
        if train_metrics:
            self.train_logger.log_metrics(train_metrics, step=step)
        if val_metrics:
            self.val_logger.log_metrics(val_metrics, step=step)

    def finalize(self, status: str) -> None:
        self.train_logger.finalize(status)
        self.val_logger.finalize(status)
