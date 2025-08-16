# mqbench_ptq_full.py
import os
import re
import time
import logging
import warnings
from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Tuple
from types import SimpleNamespace

import torch
import torchvision as tv
import torchvision.transforms as T
from torchvision.models import get_model, get_model_weights

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.convert_deploy import convert_deploy

# Optional advanced PTQ (AdaRound / BRECQ / QDrop)
try:
    from mqbench.advanced_ptq import ptq_reconstruction
    HAS_ADV = True
except Exception:
    HAS_ADV = False

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def make_adv_cfg(user_cfg=None):
    # defaults work for BRECQ/QDrop-style reconstruction
    defaults = dict(
        pattern="block",                 # "layer" -> AdaRound, "block" -> BRECQ/QDrop
        scale_lr=4e-5,
        warm_up=0.2,
        weight=0.01,
        max_count=20000,
        b_range=[20, 2],
        keep_gpu=True,                   # keep stacked calibration tensors on GPU
        round_mode="learned_hard_sigmoid",
        prob=1.0,                        # 1.0 => BRECQ/AdaRound; <1.0 => QDrop
    )
    if isinstance(user_cfg, dict):
        defaults.update(user_cfg)
    elif user_cfg is not None:
        # already a namespace/object; return as-is
        return user_cfg
    return SimpleNamespace(**defaults)


# =========================
# Logging & utilities
# =========================
def setup_logger(log_file: Optional[str], verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    logging.captureWarnings(True)


@contextmanager
def log_section(name: str):
    logging.info(f"▶ START: {name}")
    t0 = time.time()
    try:
        yield
    except Exception:
        logging.exception(f"✖ ERROR in section: {name}")
        raise
    finally:
        dt = time.time() - t0
        logging.info(f"✔ END: {name} (elapsed {dt:.2f}s)")


def log_cuda_mem(tag: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        logging.info(f"[CUDA] {tag}: alloc={alloc:.1f}MB, reserved={reserved:.1f}MB")


def env_info():
    info = {
        "torch": torch.__version__,
        "torchvision": tv.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update({
            "device_name": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "cudnn_version": torch.backends.cudnn.version(),
        })
    return info


# =========================
# Dataset checks & loaders
# =========================
def assert_imagenet_val_structure(val_root: str):
    """Ensure ILSVRC2012 val/ is split into 1,000 synset folders."""
    classes = [d for d in os.listdir(val_root) if os.path.isdir(os.path.join(val_root, d))]
    if len(classes) != 1000:
        raise RuntimeError(
            f"Expected 1000 class folders, found {len(classes)} at {val_root}. "
            "Reorganize ILSVRC2012 val into synset subfolders."
        )
    syn_pat = re.compile(r"^n\d{8}$")
    bad = [c for c in classes if not syn_pat.match(c)]
    if bad:
        raise RuntimeError(f"Found non-synset folder names: {bad[:5]} ...")
    logging.info("Val structure looks OK (1000 synset folders).")


def build_loaders(
    imagenet_val_root: str,
    weights,
    img_size: int = 224,
    calib_batches: int = 32,
    batch_size: int = 64,
    workers: int = 8,
):
    """
    Use the weights' own eval transforms for correctness.
    """
    if weights is not None and hasattr(weights, "transforms"):
        tfm = weights.transforms()
    else:
        tfm = T.Compose([
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    ds_val = tv.datasets.ImageFolder(imagenet_val_root, transform=tfm)
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )

    def calib_iter(dl, n_batches):
        it = iter(dl)
        for _ in range(n_batches):
            try:
                images, _ = next(it)
            except StopIteration:
                it = iter(dl)  # cycle if calib > dataset
                images, _ = next(it)
            yield images

    return val_loader, calib_iter


def log_first_batch_stats(loader):
    try:
        images, _ = next(iter(loader))
        m, s, mn, mx = images.mean().item(), images.std().item(), images.min().item(), images.max().item()
        logging.info(f"[SANITY] Batch[0] stats: mean={m:.4f}, std={s:.4f}, min={mn:.3f}, max={mx:.3f}")
    except Exception as e:
        logging.warning(f"Could not log first-batch stats: {e}")


# =========================
# Model loading
# =========================
def load_model_with_weights(
    model_name: str,
    weights_arg: Optional[str],
    device: str
) -> Tuple[torch.nn.Module, Optional[object], Optional[float]]:
    """
    Loads a torchvision model using the modern weights API.
    weights_arg: "DEFAULT", a specific enum name (e.g., "IMAGENET1K_V1"), or "NONE".
    Returns (model, weights_enum_or_None, ref_top1_or_None)
    """
    weights = None
    try:
        WeightsEnum = get_model_weights(model_name)
        if weights_arg is None or weights_arg.upper() == "DEFAULT":
            weights = getattr(WeightsEnum, "DEFAULT", None)
        elif weights_arg.upper() != "NONE":
            weights = getattr(WeightsEnum, weights_arg, None)
            if weights is None:
                logging.warning(f"Unknown weights '{weights_arg}' for {model_name}. Falling back to DEFAULT.")
                weights = getattr(WeightsEnum, "DEFAULT", None)
    except Exception as e:
        logging.warning(f"Could not resolve weights enum for {model_name}: {e}. Using random init.")

    model = get_model(model_name, weights=weights).eval().to(device)
    ref_top1 = None
    if weights is not None:
        ref_top1 = weights.meta.get("acc@1", None)
    return model, weights, ref_top1


# =========================
# Eval helper
# =========================
@torch.no_grad()
def top1(model, loader, device, log_prefix="EVAL", log_every=50):
    model.eval()
    total, correct = 0, 0
    t0 = time.time()
    for i, (images, targets) in enumerate(loader, 1):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
        if i % log_every == 0:
            logging.info(f"[{log_prefix}] progress: {i} batches, running top1={100.0*correct/max(total,1):.2f}%")
    dt = time.time() - t0
    acc = 100.0 * correct / max(total, 1)
    logging.info(f"[{log_prefix}] done: {i} batches in {dt:.2f}s, top1={acc:.2f}%")
    return acc


# =========================
# PTQ core
# =========================
def count_quantish_modules(model):
    n_all = sum(1 for _ in model.modules())
    n_quantish = sum(
        1 for m in model.modules()
        if ("quant" in m.__class__.__name__.lower())
        or ("fake" in m.__class__.__name__.lower())
        or ("observer" in m.__class__.__name__.lower())
        or hasattr(m, "quant_min")
    )
    return n_all, n_quantish


def run_ptq(
    model, backend, calib_images_fn, device="cuda",
    do_advanced=False, adv_cfg=None, calib_steps=None,
    log_interval=10, profile_mem=False,
):
    # You can keep this, but it’s not sufficient because prepare_by_platform creates new CPU modules
    model.to(device).eval()

    with log_section(f"prepare_by_platform({backend.name})"):
        pre_all, _ = count_quantish_modules(model)
        model = prepare_by_platform(model, backend)
        # ★★★ IMPORTANT: move AFTER prepare_by_platform ★★★
        model = model.to(device).eval()
        post_all, post_quantish = count_quantish_modules(model)
        logging.info(f"Modules (total): {pre_all} -> {post_all}")
        logging.info(f"'Quantish' modules detected after prepare: {post_quantish}")
        if profile_mem: log_cuda_mem("after prepare_by_platform")

    with log_section("calibration (enable_calibration + forward)"):
        enable_calibration(model)
        seen_imgs, t0 = 0, time.time()
        iterator = calib_images_fn()
        for step, images in enumerate(iterator, 1):
            images = images.to(device, non_blocking=True)
            _ = model(images)  # all on same device now
            seen_imgs += images.size(0)
            if (step % log_interval == 0) or (step == 1):
                elapsed = time.time() - t0
                ips = seen_imgs / max(elapsed, 1e-6)
                logging.info(f"[CALIB] step={step}/{calib_steps or '?'} seen={seen_imgs} ({ips:.1f} img/s)")
                if profile_mem: log_cuda_mem(f"calib step {step}")
        logging.info(f"[CALIB] total images seen: {seen_imgs}")

    if do_advanced:
        assert HAS_ADV, "mqbench.advanced_ptq not available."
    with log_section("advanced PTQ reconstruction"):
        # build config object with attributes (no more dict)
        adv_ns = make_adv_cfg(adv_cfg)

        # stack calibration mini-batches (respect keep_gpu)
        stacked, total_imgs = [], 0
        iterator = calib_images_fn()
        if tqdm is not None and calib_steps:
            iterator = tqdm(iterator, total=calib_steps, desc="Stack", leave=False)
        with torch.no_grad():
            for images in iterator:
                img_dev = device if adv_ns.keep_gpu else "cpu"
                stacked.append(images.to(img_dev, non_blocking=True))
                total_imgs += images.size(0)

        logging.info(f"[ADV] cfg={adv_ns.__dict__}")
        logging.info(f"[ADV] stacked tensors: {len(stacked)} | total calib images: {total_imgs}")

        if profile_mem: log_cuda_mem("before ptq_reconstruction")
        # Convert SimpleNamespace to dict for ptq_reconstruction compatibility
        adv_dict = vars(adv_ns)
        model = ptq_reconstruction(model, stacked, adv_dict)  # <-- pass dict instead of namespace
        # ensure model is on the right device after reconstruction
        model = model.to(device).eval()
        if profile_mem: log_cuda_mem("after ptq_reconstruction")

    with log_section("enable_quantization (simulate INT8)"):
        enable_quantization(model)
        if profile_mem: log_cuda_mem("after enable_quantization")

    return model



# =========================
# Script entry
# =========================
def main():
    import argparse, random, numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--val_root", required=True, help="Path to ImageNet val (folder with class subdirs).")
    p.add_argument("--backend", default="Tensorrt",
                   help="BackendType enum name, e.g., Tensorrt, OPENVINO, Vitis, PPLCUDA, SNPE, ONNX_QNN, Tengine_u8, Tensorrt_NLP, etc.")
    p.add_argument("--model", default="resnet18", help="torchvision model name (e.g., resnet18, mobilenet_v2, vit_b_16)")
    p.add_argument("--weights", default="DEFAULT", help="Weights enum for the model (DEFAULT | NONE | e.g., IMAGENET1K_V1)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--calib_batches", type=int, default=32, help="How many batches to calibrate with")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--advanced", action="store_true", help="Use advanced PTQ reconstruction (AdaRound/BRECQ/QDrop)")
    p.add_argument("--export_onnx", action="store_true")
    p.add_argument("--log_file", default=None, help="Write logs to this file (in addition to console)")
    p.add_argument("--verbose", action="store_true", help="DEBUG logging")
    p.add_argument("--log_interval", type=int, default=10, help="Log every N calibration steps")
    p.add_argument("--profile_mem", action="store_true", help="Log CUDA memory at key steps")
    p.add_argument("--no_structure_check", action="store_true", help="Skip ImageNet val/ structure check")
    p.add_argument("--seed", type=int, default=123, help="Set random seed for reproducibility")
    args = p.parse_args()

    setup_logger(args.log_file, args.verbose)

    # Repro
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.info(f"Environment: {env_info()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with log_section("load fp32 model (torchvision weights API)"):
        fp, weights, ref_top1 = load_model_with_weights(args.model, args.weights, device)
        n_params = sum(p.numel() for p in fp.parameters())
        logging.info(f"Model: {args.model} | Weights: {weights} | Params: {n_params/1e6:.2f}M | Ref acc@1={ref_top1}")

    with log_section("build & check loaders"):
        if not args.no_structure_check:
            assert_imagenet_val_structure(args.val_root)

        val_loader, calib_iter_builder = build_loaders(
            args.val_root, weights=weights, img_size=args.img_size,
            calib_batches=args.calib_batches, batch_size=args.batch_size, workers=args.workers
        )
        logging.info(f"Val dataset size: {getattr(val_loader.dataset, '__len__', lambda: 'unknown')()} | "
                     f"batch_size={args.batch_size} | calib_batches={args.calib_batches}")
        log_first_batch_stats(val_loader)

    # Fresh iterator factory for each pass (two passes if advanced is enabled)
    def calib_images_fn():
        return calib_iter_builder(val_loader, args.calib_batches)

    # Backend enum
    try:
        backend = getattr(BackendType, args.backend)
    except AttributeError:
        raise ValueError(f"Unknown BackendType '{args.backend}'. Check mqbench.prepare_by_platform.BackendType.")

    # Baseline FP32 eval
    '''try:
        with log_section("evaluate FP32 baseline"):
            acc_fp = top1(fp, val_loader, device, log_prefix="EVAL_FP32")
            logging.info(f"[FP32] Top-1 = {acc_fp:.2f}% "
                         f"(expected ~{ref_top1} if weights & transforms match)")
    except Exception:
        logging.exception("FP32 eval failed/skipped")'''

    # Run PTQ
    qmodel = run_ptq(
        fp,
        backend,
        calib_images_fn=calib_images_fn,
        device=device,
        do_advanced=args.advanced,
        calib_steps=args.calib_batches,
        log_interval=args.log_interval,
        profile_mem=args.profile_mem,
    )

    # INT8-sim eval
    try:
        with log_section("evaluate INT8-sim"):
            acc_q = top1(qmodel, val_loader, device, log_prefix="EVAL_INT8")
            logging.info(f"[PTQ][{args.model}][{args.backend}]{' [ADV]' if args.advanced else ''} Top-1 = {acc_q:.2f}%")
    except Exception:
        logging.exception("INT8 eval failed/skipped")

    # Optional export
    if args.export_onnx:
        with log_section(f"convert_deploy/export ({args.backend})"):
            input_shape = {"data": [1, 3, args.img_size, args.img_size]}
            out = convert_deploy(qmodel, backend, input_shape)
            logging.info(f"Exported deployable model/artifacts: {out}")


if __name__ == "__main__":
    main()
