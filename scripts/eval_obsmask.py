"""Full-dataset validation and correspondence visualizations for ObsMask fine-tunes."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import einops
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from cleandift.isaac_dataloader import IsaacObsMaskDataset
from cleandift.utils import dict_to

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
logger = logging.getLogger(__name__)

DATASET_DIRS = [
    "/ocean/projects/cis260205p/jke2/refseg-workspace/segmentation/datasets/expanded-refseg",
    "/ocean/projects/cis260205p/jke2/refseg-workspace/segmentation/datasets/mixed-persp",
    "/ocean/projects/cis260205p/jke2/refseg-workspace/UFM-train/datasets/shelf-optflow",
]

CORRESPONDENCE_PAIRS = [
    {
        "name": "mixed-persp_render000_vs_render001_f84",
        "dataset_root": DATASET_DIRS[1],
        "source_render": "render000",
        "target_render": "render001",
        "frame_idx": 84,
    },
    {
        "name": "expanded-refseg_render000_vs_render001_f150",
        "dataset_root": DATASET_DIRS[0],
        "source_render": "render000",
        "target_render": "render001",
        "frame_idx": 150,
    },
    {
        "name": "shelf-optflow_render000_vs_render001_f50",
        "dataset_root": DATASET_DIRS[2],
        "source_render": "render000",
        "target_render": "render001",
        "frame_idx": 50,
    },
]


@dataclass
class ValMetrics:
    n_samples: int
    n_batches: int
    total_loss: float
    per_layer: dict[str, float]


def load_model(checkpoint: Path, device: torch.device):
    cfg = OmegaConf.load(_CONFIG_DIR / "sd21_isaac_obsmask.yaml")
    cfg.data.dataset_dirs = DATASET_DIRS
    cfg.data.dataset_dir = None
    cfg.data.batch_size = 2
    OmegaConf.resolve(cfg)
    cfg = hydra.utils.instantiate(cfg)
    model = cfg.model.to(device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cfg.data


def batch_loss(model, batch, device) -> tuple[float, dict[str, float]]:
    with torch.no_grad():
        losses = model(**dict_to(batch, device=device))
    per_layer = {k: float(v.mean().cpu()) for k, v in losses.items()}
    total = float(sum(per_layer.values()))
    return total, per_layer


def run_full_validation(model, dataloader, device) -> ValMetrics:
    total_loss = 0.0
    per_layer_sum: dict[str, float] = {}
    n_samples = 0

    for batch in tqdm(dataloader, desc="Full validation"):
        batch_total, batch_layers = batch_loss(model, batch, device)
        bs = batch["x"].shape[0]
        total_loss += batch_total * bs
        n_samples += bs
        for k, v in batch_layers.items():
            per_layer_sum[k] = per_layer_sum.get(k, 0.0) + v * bs

    per_layer = {k: v / n_samples for k, v in per_layer_sum.items()}
    return ValMetrics(
        n_samples=n_samples,
        n_batches=len(dataloader),
        total_loss=total_loss / n_samples,
        per_layer=per_layer,
    )


def load_isaac_frame(dataset_root: str, render_dir: str, frame_idx: int, img_size: int = 768):
    ds = IsaacObsMaskDataset(
        dataset_root,
        img_size=img_size,
        render_dirs=[render_dir],
    )
    for i, (rd, f) in enumerate(ds.index):
        if rd.name == render_dir and f == frame_idx:
            return ds[i]
    available = sorted({f for rd, f in ds.index if rd.name == render_dir})
    raise KeyError(
        f"frame {frame_idx} not in {dataset_root}/{render_dir}; "
        f"available frames (sample): {available[:5]}...{available[-3:]}"
    )


def tensor_to_display(x: torch.Tensor) -> np.ndarray:
    rgb = ((x.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)
    return einops.rearrange(rgb, "c h w -> h w c").numpy()


def source_points_from_foreground(x: torch.Tensor, n_points: int = 5) -> list[list[int]]:
    rgb = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    fg = rgb.mean(dim=0) > 0.05
    ys, xs = fg.nonzero(as_tuple=True)
    if len(xs) == 0:
        h, w = fg.shape
        return [[w // 2, h // 2]]

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    candidates = [
        (cx, cy),
        (x0 + (x1 - x0) // 4, y0 + (y1 - y0) // 4),
        (x0 + 3 * (x1 - x0) // 4, y0 + (y1 - y0) // 4),
        (x0 + (x1 - x0) // 4, y0 + 3 * (y1 - y0) // 4),
        (x0 + 3 * (x1 - x0) // 4, y0 + 3 * (y1 - y0) // 4),
    ]
    return [[x, y] for x, y in candidates[:n_points]]


@torch.no_grad()
def correspondence_maps(
    model,
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    source_points: list[list[int]],
    feat_key: str = "us6",
    device: torch.device | None = None,
):
    device = device or source_x.device
    caption = [""]
    source = source_x.unsqueeze(0).to(device)
    target = target_x.unsqueeze(0).to(device)

    source_features = model.get_features(source, caption, t=None, feat_key=feat_key)
    target_features = model.get_features(target, caption, t=None, feat_key=feat_key)

    h, w = source.shape[-2], source.shape[-1]
    source_features_up = F.interpolate(source_features, (h, w), mode="bilinear")
    target_features_up = F.interpolate(target_features, (h, w), mode="bilinear")

    pts = torch.tensor(source_points, device=device, dtype=torch.long)
    source_point_feats = source_features_up[0, :, pts[:, 1], pts[:, 0]].T[:, None]
    target_norm_flat = einops.rearrange(
        target_features_up / target_features_up.norm(p=2, dim=1, keepdim=True),
        "1 c h w -> c (h w)",
    )
    source_norm = source_point_feats / source_point_feats.norm(p=2, dim=-1, keepdim=True)
    sims = einops.rearrange(
        source_norm @ target_norm_flat,
        "b 1 (h w) -> b h w",
        h=target_features_up.shape[-2],
    )
    matches = torch.stack(
        torch.unravel_index(
            einops.rearrange(sims, "b h w -> b (h w)").argmax(dim=-1),
            sims.shape[1:],
        )
    ).T
    return sims, matches


def save_correspondence_figure(
    source_img: np.ndarray,
    target_img: np.ndarray,
    source_points: list[list[int]],
    sims: torch.Tensor,
    matches: torch.Tensor,
    out_path: Path,
):
    n_pts = len(source_points)
    fig, axes = plt.subplots(n_pts, 3, figsize=(12, 3.5 * n_pts), squeeze=False)
    for pt_idx in range(n_pts):
        sx, sy = source_points[pt_idx]
        ty, tx = matches[pt_idx].tolist()

        axes[pt_idx, 0].imshow(source_img)
        axes[pt_idx, 0].scatter(sx, sy, edgecolor="white", linewidth=1, color="C2", s=50)
        axes[pt_idx, 0].set_title("Source")
        axes[pt_idx, 0].axis("off")

        axes[pt_idx, 1].imshow(sims[pt_idx].cpu().float(), cmap="magma")
        axes[pt_idx, 1].set_title("Similarity heatmap")
        axes[pt_idx, 1].axis("off")

        axes[pt_idx, 2].imshow(target_img)
        axes[pt_idx, 2].scatter(tx, ty, edgecolor="white", linewidth=1, color="C1", s=50)
        axes[pt_idx, 2].set_title("Target match")
        axes[pt_idx, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_correspondence_viz(model, output_dir: Path, device: torch.device, feat_key: str = "us6"):
    viz_dir = output_dir / "correspondence"
    viz_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for pair in CORRESPONDENCE_PAIRS:
        source_item = load_isaac_frame(
            pair["dataset_root"], pair["source_render"], pair["frame_idx"]
        )
        target_item = load_isaac_frame(
            pair["dataset_root"], pair["target_render"], pair["frame_idx"]
        )
        source_x = source_item["x"]
        target_x = target_item["x"]
        source_points = source_points_from_foreground(source_x)
        sims, matches = correspondence_maps(
            model, source_x, target_x, source_points, feat_key=feat_key, device=device
        )

        out_path = viz_dir / f"{pair['name']}.png"
        save_correspondence_figure(
            tensor_to_display(source_x),
            tensor_to_display(target_x),
            source_points,
            sims,
            matches,
            out_path,
        )
        manifest.append(
            {
                **pair,
                "feat_key": feat_key,
                "source_points": source_points,
                "matches_yx": matches.cpu().tolist(),
                "image_path": str(out_path),
            }
        )
        logger.info("Saved correspondence viz: %s", out_path)

    with open(viz_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/ocean/projects/cis260205p/jke2/cleandift-checkpoints/sd21-obsmask-multi/step_400.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/ocean/projects/cis260205p/jke2/cleandift-checkpoints/sd21-obsmask-multi/eval_step400"
        ),
    )
    parser.add_argument("--feat-key", default="us6")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-correspondence", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint %s", args.checkpoint)
    model, data_module = load_model(args.checkpoint, device)

    results = {"checkpoint": str(args.checkpoint), "feat_key": args.feat_key}

    if not args.skip_val:
        val_loader = data_module.val_dataloader()
        logger.info("Running full validation on %d samples", len(val_loader.dataset))
        metrics = run_full_validation(model, val_loader, device)
        results["full_validation"] = asdict(metrics)
        logger.info("Full validation loss: %.4f (n=%d)", metrics.total_loss, metrics.n_samples)
        with open(args.output_dir / "full_validation.json", "w") as f:
            json.dump(results["full_validation"], f, indent=2)

    if not args.skip_correspondence:
        run_correspondence_viz(model, args.output_dir, device, feat_key=args.feat_key)
        results["correspondence_dir"] = str(args.output_dir / "correspondence")

    with open(args.output_dir / "eval_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote results to %s", args.output_dir)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
