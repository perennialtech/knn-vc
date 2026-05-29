import argparse
import itertools
import logging
import os
import subprocess
import time
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datamodules import create_dataloader
from .mel_utils import LogMelSpectrogram
from .models import (Generator, MultiPeriodDiscriminator,
                     MultiScaleDiscriminator, discriminator_loss, feature_loss,
                     generator_loss)
from .utils import load_checkpoint, scan_checkpoint

torch.backends.cudnn.benchmark = True


def get_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Trainer:
    def __init__(self, config: Any, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.tb_logger = SummaryWriter(config.ckpt_path)
        self.device = config.device

        # init models
        self.generator = Generator(config.hifigan).to(self.device)
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)

        # check if ckpt folder already exists and retrieve checkpoints
        os.makedirs(config.ckpt_path, exist_ok=True)
        logger.info(f"checkpoints directory: {config.ckpt_path}")

        cp_g = None
        cp_do = None
        if os.path.isdir(config.ckpt_path):
            cp_g = scan_checkpoint(config.ckpt_path, "g_")
            cp_do = scan_checkpoint(config.ckpt_path, "do_")

        # if ckpt folder is new, start training from scratch
        if cp_g is None or cp_do is None:
            if getattr(config, "resume", False):
                raise FileNotFoundError(
                    f"No complete generator/discriminator checkpoint pair found in {config.ckpt_path}"
                )

            self.steps = 0
            state_dict_do = None
            self.last_epoch = -1

        # otherwise, resume training from ckpt
        else:
            state_dict_g = load_checkpoint(cp_g, self.device)
            state_dict_do = load_checkpoint(cp_do, self.device)
            self.generator.load_state_dict(state_dict_g["generator"])
            self.mpd.load_state_dict(state_dict_do["mpd"])
            self.msd.load_state_dict(state_dict_do["msd"])
            self.steps = state_dict_do["steps"] + 1
            self.last_epoch = state_dict_do["epoch"]
            logger.info(f"Restored checkpoint from {cp_g} and {cp_do}")

        # setup optimizers
        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            config.adamw.learning_rate,
            betas=(config.adamw.adam_b1, config.adamw.adam_b2),
        )
        self.optim_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            config.adamw.learning_rate,
            betas=(config.adamw.adam_b1, config.adamw.adam_b2),
        )

        # load optimizer checkpoints if appropriate
        if state_dict_do is not None:
            self.optim_g.load_state_dict(state_dict_do["optim_g"])
            self.optim_d.load_state_dict(state_dict_do["optim_d"])

        # setup schedulers and gradient scalers
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=config.adamw.lr_decay, last_epoch=self.last_epoch
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=config.adamw.lr_decay, last_epoch=self.last_epoch
        )
        self.scaler_g = GradScaler(self.device, enabled=config.fp16)
        self.scaler_d = GradScaler(self.device, enabled=config.fp16)

        # create dataloaders
        self.train_loader = create_dataloader(config.train_tars, config, logger)
        self.valid_loader = create_dataloader(
            config.valid_tars, config, logger, shuffle=False
        )

        # initialize mel-spectrogram transform
        self.melspec = LogMelSpectrogram(
            config.mel.n_fft,
            config.mel.num_mels,
            config.sample_rate,
            config.hifigan.hop_size,
            config.mel.win_size,
            config.mel.fmin,
            config.mel.fmax,
        ).to(self.device)

    def train(self):
        self.generator.train()
        self.mpd.train()
        self.msd.train()

        for epoch in tqdm(range(max(0, self.last_epoch), self.config.training_epochs)):
            start_epoch = time.time()

            for batch in self.train_loader:
                start_batch = time.time()
                y_mel, y_g_hat_mel, loss_gen_all, loss_disc_all = self.train_batch(
                    batch
                )

                # logging and checkpointing
                if self.steps % self.config.checkpoint_interval == 0 and self.steps > 0:
                    self.store_checkpoint(epoch)

                if self.steps % self.config.log_interval == 0:
                    self.write_training_logs(
                        loss_gen_all, y_mel, y_g_hat_mel, loss_disc_all, start_batch
                    )

                # validation
                if self.steps % self.config.validation_interval == 0:
                    self.generator.eval()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    val_err_tot = 0
                    n_batches = 0
                    with torch.no_grad():
                        for batch in self.valid_loader:
                            val_err = self.valid_batch(batch)
                            val_err_tot += val_err
                            n_batches += 1

                    # log avg. mel-spech error on the validation set
                    val_err = val_err_tot / n_batches
                    self.tb_logger.add_scalar("val/mel_spec_error", val_err, self.steps)
                    self.logger.info(
                        f"val. done at {self.steps:,d} steps. mel spec error: {val_err:5.4f}"
                    )

                    # go back to training
                    self.generator.train()
                    if self.device == "cuda":
                        self.tb_logger.add_scalar(
                            "memory/max_allocated_gb",
                            torch.cuda.max_memory_allocated() / 1e9,
                            self.steps,
                        )
                        self.tb_logger.add_scalar(
                            "memory/max_reserved_gb",
                            torch.cuda.max_memory_reserved() / 1e9,
                            self.steps,
                        )
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.reset_accumulated_memory_stats()

                self.steps += 1

            # epoch is done; update schedulers and log epoch duration
            self.scheduler_g.step()
            self.scheduler_d.step()
            self.logger.info(
                "Time taken for epoch {} is {} sec".format(
                    epoch + 1, int(time.time() - start_epoch)
                )
            )

    def run_generator(self, batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = batch.ssl.to(self.device, non_blocking=True)
        y = batch.audio.to(self.device, non_blocking=True)
        y_mel = self.melspec(y.squeeze(1))

        with torch.amp.autocast(enabled=self.config.fp16, device_type=self.device):
            y_g_hat = self.generator(x)
            y_g_hat_mel = self.melspec(y_g_hat.squeeze(1))

        return y, y_mel, y_g_hat, y_g_hat_mel

    def train_batch(self, batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        y, y_mel, y_g_hat, y_g_hat_mel = self.run_generator(batch)

        # run discriminator and compute its losses
        self.optim_d.zero_grad()
        with torch.amp.autocast(enabled=self.config.fp16, device_type=self.device):
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )

            loss_disc_all = loss_disc_s + loss_disc_f

        # run the backward prop for the discriminators
        self.scaler_d.scale(loss_disc_all).backward()
        self.scaler_d.step(self.optim_d)
        self.scaler_d.update()

        # Compute generator losses
        self.optim_g.zero_grad()
        with torch.amp.autocast(enabled=self.config.fp16, device_type=self.device):
            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        # run the backward prop for the generator
        self.scaler_g.scale(loss_gen_all).backward()
        self.scaler_g.step(self.optim_g)
        self.scaler_g.update()

        return y_mel, y_g_hat_mel, loss_gen_all, loss_disc_all

    def valid_batch(self, batch) -> float:
        y, y_mel, y_g_hat, y_g_hat_mel = self.run_generator(batch)

        # pad the audio and compute the mel-spec again if shapes don't match
        # TODO: this shouldn't be needed
        if y_g_hat_mel.shape[-1] != y_mel.shape[-1]:
            self.logger.warning("Mismatching shapes between mel-spectrograms!")
            n_pad = self.config.hifigan.hop_size
            y_g_hat = F.pad(y_g_hat, (n_pad // 2, n_pad - n_pad // 2))
            y_g_hat_mel = self.melspec(y_g_hat.squeeze(1))

        return F.l1_loss(y_mel, y_g_hat_mel).item()

    def write_training_logs(
        self,
        loss_gen_all: Tensor,
        y_mel: Tensor,
        y_g_hat_mel: Tensor,
        loss_disc_all: Tensor,
        start_batch: float,
    ):
        with torch.no_grad():
            mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

        peak_memory_gb = (
            torch.cuda.max_memory_allocated() / 1e9 if self.device == "cuda" else 0.0
        )

        # TB logs
        self.tb_logger.add_scalar("train/gen_loss_total", loss_gen_all, self.steps)
        self.tb_logger.add_scalar("train/mel_spec_error", mel_error, self.steps)
        self.tb_logger.add_scalar("train/disc_loss_total", loss_disc_all, self.steps)

        # TXT logs
        self.logger.info(
            "Steps : {:,d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, sec/batch : {:4.3f}, peak mem: {:5.2f}GB".format(
                self.steps,
                loss_gen_all.item(),
                mel_error,
                time.time() - start_batch,
                peak_memory_gb,
            )
        )

    def store_checkpoint(self, epoch: int):
        self.logger.info(f"Storing checkpoints after {self.steps} steps")
        ckpt_path = "{}/g_{:08d}.pt".format(self.config.ckpt_path, self.steps)
        torch.save({"generator": (self.generator).state_dict()}, ckpt_path)
        ckpt_path = "{}/do_{:08d}.pt".format(self.config.ckpt_path, self.steps)
        torch.save(
            {
                "mpd": (self.mpd).state_dict(),
                "msd": (self.msd).state_dict(),
                "optim_g": self.optim_g.state_dict(),
                "optim_d": self.optim_d.state_dict(),
                "steps": self.steps,
                "epoch": epoch,
            },
            ckpt_path,
        )


def override_with_args(config: Any, args: list[str]) -> None:
    """
    Update existing config keys from key-value CLI pairs.

    Values are parsed by OmegaConf, so booleans, numbers, nulls, lists, and strings
    follow the same rules as config files.
    """

    def check_key_existence(sub_config: DictConfig, key: str, full_key: str) -> None:
        if not isinstance(sub_config, DictConfig) or key not in sub_config:
            raise KeyError(
                f"Subkey '{key}' not found in config for override '{full_key}'"
            )

    if len(args) % 2 != 0:
        raise RuntimeError(
            "The number of config arguments must be even (key-value pairs)"
        )

    for key, value in zip(args[0::2], args[1::2]):
        keys = key.split(".")

        sub_config = config
        for sub_key in keys[:-1]:
            check_key_existence(sub_config, sub_key, key)
            sub_config = sub_config[sub_key]

        check_key_existence(sub_config, keys[-1], key)

        parsed_override = OmegaConf.from_dotlist([f"{key}={value}"])
        parsed_value = OmegaConf.select(parsed_override, key)
        OmegaConf.update(config, key, parsed_value, merge=False)


def resolve_run_directory(config: Any, run_dir: str | None, resume: bool) -> str:
    if resume:
        if run_dir is None:
            raise RuntimeError("--resume requires --run-dir")

        if not os.path.isdir(run_dir):
            raise FileNotFoundError(
                f"Cannot resume because run directory does not exist: {run_dir}"
            )

        return run_dir

    if run_dir is None:
        return os.path.join(config.checkpoint_dir, str(int(time.time())))

    if os.path.isfile(run_dir) or (os.path.isdir(run_dir) and os.listdir(run_dir)):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}"
        )

    return run_dir


def main(argv: list[str] | None = None):

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")

    # load args and config overrides
    args, config_overrides = parser.parse_known_args(argv)
    config = OmegaConf.load(args.config)
    override_with_args(config, config_overrides)

    # define logging directory
    config.ckpt_path = os.path.join(config.checkpoint_dir, str(int(time.time())))
    config.ckpt_path = resolve_run_directory(config, args.run_dir, args.resume)
    os.makedirs(config.ckpt_path, exist_ok=True)

    # add args to config
    for key, value in vars(args).items():
        if isinstance(config, dict):
            config[key] = value
        else:
            OmegaConf.update(config, key, value, merge=False)

    # dump config with commit hash
    config.commit_hash = get_commit_hash()
    OmegaConf.save(config, os.path.join(config.ckpt_path, "config.yaml"))

    # create logger
    logger = logging.getLogger("train")
    logger.handlers.clear()
    handler = logging.FileHandler(os.path.join(config.ckpt_path, "train.log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("Initializing training")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        logger.info(f"Batch size: {config.batch_size}")
        config.device = "cuda"
    else:
        logger.info("Batch size set to 1 for CPU")
        config.batch_size = 1
        config.device = "cpu"
        config.fp16 = False

    config.commit_hash = get_commit_hash()
    OmegaConf.save(config, os.path.join(config.ckpt_path, "config.yaml"))

    trainer = Trainer(config, logger)
    trainer.train()


if __name__ == "__main__":
    main()
