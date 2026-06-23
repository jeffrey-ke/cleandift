"""Isaac ObsMask render dirs → CleanDIFT training batches.

Reads isaac_datagen on-disk layout via vision_core serialize/deserialize
(ObsMask per frame, ObsMaskDescriptorMetadata at idx 0). Returns the same
batch keys as DummyDataset: {"x", "caption"}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

from vision_core.datastructs import ObsMask, ObsMaskDescriptorMetadata, count_samples
from vision_core.transforms import StripAlpha


def alpha_crop(obs: torch.Tensor) -> torch.Tensor:
    """Crop RGBA (4, H, W) to the bounding box of non-transparent pixels."""
    ys, xs = np.nonzero(np.asarray(obs[3]) > 0)
    if ys.size == 0:
        raise ValueError("fully transparent observation")
    rows = slice(int(ys.min()), int(ys.max()) + 1)
    cols = slice(int(xs.min()), int(xs.max()) + 1)
    return obs[:, rows, cols]


def _discover_render_dirs(root: Path, render_dirs: list[str] | None) -> list[Path]:
    if render_dirs is not None:
        dirs = [root / name for name in render_dirs]
        for d in dirs:
            if not (d / "obs").is_dir():
                raise FileNotFoundError(f"render dir missing obs/: {d}")
        return sorted(dirs)
    dirs = sorted(d for d in root.glob("render*") if (d / "obs").is_dir())
    if not dirs:
        raise FileNotFoundError(f"No render dirs (render*/ with obs/) under {root}")
    return dirs


def _caption_from_classes(om: ObsMask, md: ObsMaskDescriptorMetadata) -> str:
    cids = np.unique(om.cid_mask.numpy())
    names = sorted(md.cid_to_class[int(c)] for c in cids if int(c) != 0)
    return ", ".join(names)


def _dataset_kwargs(
    img_size: int,
    alpha_crop: bool,
    caption_mode: Literal["empty", "classes"],
    render_dirs: list[str] | None,
    train: bool,
) -> dict:
    return dict(
        img_size=img_size,
        alpha_crop=alpha_crop,
        caption_mode=caption_mode,
        render_dirs=render_dirs,
        train=train,
    )


def make_isaac_obsmask_dataset(
    *,
    dataset_dir: str | Path | None = None,
    dataset_dirs: list[str | Path] | None = None,
    img_size: int = 768,
    alpha_crop: bool = True,
    caption_mode: Literal["empty", "classes"] = "empty",
    render_dirs: list[str] | None = None,
    train: bool = True,
) -> data.Dataset:
    """One umbrella dir, or concatenate several (e.g. expanded-refseg + mixed-persp)."""
    kw = _dataset_kwargs(img_size, alpha_crop, caption_mode, render_dirs, train)
    if dataset_dirs is not None:
        if dataset_dir is not None:
            raise ValueError("pass dataset_dir or dataset_dirs, not both")
        parts = [IsaacObsMaskDataset(root, **kw) for root in dataset_dirs]
        if not parts:
            raise ValueError("dataset_dirs is empty")
        return parts[0] if len(parts) == 1 else data.ConcatDataset(parts)
    if dataset_dir is None:
        raise ValueError("need dataset_dir or dataset_dirs")
    return IsaacObsMaskDataset(dataset_dir, **kw)


class IsaacObsMaskDataset(data.Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        img_size: int = 768,
        *,
        alpha_crop: bool = True,
        caption_mode: Literal["empty", "classes"] = "empty",
        render_dirs: list[str] | None = None,
        train: bool = True,
    ):
        del train  # same index for train/val unless DataModule holds out render dirs
        self.img_size = img_size
        self.alpha_crop = alpha_crop
        self.caption_mode = caption_mode
        self.root = Path(dataset_dir)
        self.render_dirs = _discover_render_dirs(self.root, render_dirs)
        self._md: dict[Path, ObsMaskDescriptorMetadata] = {}
        self.index: list[tuple[Path, int]] = []
        for rd in self.render_dirs:
            self._md[rd] = ObsMaskDescriptorMetadata.deserialize(0, rd)
            for f in range(count_samples(rd, field="obs")):
                self.index.append((rd, f))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rd, f = self.index[idx]
        om = ObsMask.deserialize(f, rd)
        obs = om.obs
        if self.alpha_crop:
            obs = alpha_crop(obs)
        rgb = StripAlpha()(obs).float() / 255.0
        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        x = rgb * 2.0 - 1.0
        if self.caption_mode == "empty":
            caption = ""
        elif self.caption_mode == "classes":
            caption = _caption_from_classes(om, self._md[rd])
        else:
            raise ValueError(f"unknown caption_mode: {self.caption_mode}")
        return {"x": x, "caption": caption}


class IsaacObsMaskDataModule:
    def __init__(
        self,
        batch_size: int = 1,
        img_size: int = 768,
        *,
        dataset_dir: str | None = None,
        dataset_dirs: list[str] | None = None,
        alpha_crop: bool = True,
        caption_mode: Literal["empty", "classes"] = "empty",
        render_dirs: list[str] | None = None,
        val_render_dirs: list[str] | None = None,
    ):
        if dataset_dir is None and dataset_dirs is None:
            raise ValueError("need dataset_dir or dataset_dirs")
        if dataset_dirs is not None and val_render_dirs is not None:
            raise ValueError("val_render_dirs applies only to a single dataset_dir")

        self.batch_size = batch_size
        train_render_dirs = render_dirs
        if val_render_dirs is not None:
            held_out = set(val_render_dirs)
            if render_dirs is None:
                all_dirs = _discover_render_dirs(Path(dataset_dir), None)
                train_render_dirs = [d.name for d in all_dirs if d.name not in held_out]
            else:
                train_render_dirs = [d for d in render_dirs if d not in held_out]

        ds_kw = dict(
            img_size=img_size,
            alpha_crop=alpha_crop,
            caption_mode=caption_mode,
            render_dirs=train_render_dirs,
            train=True,
        )
        train_dataset = make_isaac_obsmask_dataset(
            dataset_dir=dataset_dir,
            dataset_dirs=dataset_dirs,
            **ds_kw,
        )
        self.train_loader = data.DataLoader(train_dataset, batch_size=batch_size)

        if val_render_dirs is not None:
            val_dataset = IsaacObsMaskDataset(
                dataset_dir=dataset_dir,
                img_size=img_size,
                alpha_crop=alpha_crop,
                caption_mode=caption_mode,
                render_dirs=val_render_dirs,
                train=False,
            )
        else:
            val_dataset = make_isaac_obsmask_dataset(
                dataset_dir=dataset_dir,
                dataset_dirs=dataset_dirs,
                img_size=img_size,
                alpha_crop=alpha_crop,
                caption_mode=caption_mode,
                render_dirs=train_render_dirs,
                train=False,
            )
        self.val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
