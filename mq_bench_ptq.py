# mqbench_ptq_full.py  (Academic-only)
import os
import re
import time
import logging
from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Tuple

import torch
import torchvision as tv
import torchvision.transforms as T
from torchvision.models import get_model, get_model_weights

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization

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


# -------------------------
# Config ns for advanced PTQ
# -------------------------
class ConfigNamespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def make_adv_cfg(user_cfg=None, **overrides):
    defaults = dict(
        pattern="block",                 # 'layer' -> AdaRound, 'block' -> BRECQ/QDrop
        scale_lr=4e-5,
        warm_up=0.2,
        weight=0.01,
        max_count=20000,
        b_range=[20, 2],
        keep_gpu=True,
        round_mode="learned_hard_sigmoid",
        prob=1.0,                        # 1.0 => BRECQ/AdaRound; <1.0 => QDrop
    )
    if isinstance(user_cfg, dict):
        defaults.update(user_cfg)
    elif isinstance(user_cfg, ConfigNamespace):
        defaults.update(user_cfg.__dict__)
    defaults.update({k: v for k, v in overrides.items() if v is not None})
    return ConfigNamespace(**defaults)

# =========================
# Logging & utilities
# =========================
def setup_logger(log_file: Optional[str], verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers)
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
    tfm = weights.transforms() if (weights is not None and hasattr(weights, "transforms")) else T.Compose([
        T.Resize(256), T.CenterCrop(img_size),
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
                it = iter(dl)
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
    ref_top1 = weights.meta.get("acc@1", None) if weights is not None else None
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
        pred = model(images).argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
        if i % log_every == 0:
            logging.info(f"[{log_prefix}] progress: {i} batches, running top1={100.0*correct/max(total,1):.2f}%")
    dt = time.time() - t0
    acc = 100.0 * correct / max(total, 1)
    logging.info(f"[{log_prefix}] done: {i} batches in {dt:.2f}s, top1={acc:.2f}%")
    return acc


# =========================
# PTQ core (Academic only)
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

def build_academic_qconfig(
    w_bits=8,
    a_bits=8,
    w_per_channel=True,     # weights per-channel
    a_per_channel=False,    # activations per-tensor
    w_sym=True,             # symmetric weights
    a_sym=False,            # asymmetric activations (zero-point)
    w_observer="MinMaxObserver",
    a_observer="EMAMinMaxObserver",
    pot_scale=False,        # set True if you want power-of-two scales
):
    """
    Returns the exact dict MQBench expects for BackendType.Academic in this repo version.
    """
    return {
        "w_qscheme": {
            "bit": w_bits,
            "per_channel": w_per_channel,
            "sym": w_sym,
            "observer": w_observer,
            "pot_scale": pot_scale,
        },
        "a_qscheme": {
            "bit": a_bits,
            "per_channel": a_per_channel,
            "sym": a_sym,
            "observer": a_observer,
            "pot_scale": pot_scale,
        },
    }

def run_ptq(
    model,
    calib_images_fn,
    device="cuda",
    do_advanced=False,
    adv_cfg=None,
    calib_steps=None,
    log_interval=10,
    profile_mem=False,
    quant_model="fixed",
    w_bits=8,
    a_bits=8,
):
    model.to(device).eval()

    backend = BackendType.Academic
    with log_section(f"prepare_by_platform({backend.name})"):
        pre_all, _ = count_quantish_modules(model)

        # Resolve fake-quant classes from CLI
        w_fq, a_fq = resolve_fakequant_names(quant_model)  # <-- needs args in scope; see note below

        extra_config = {
            "extra_qconfig_dict": {
                "w_observer": "MinMaxObserver",
                "a_observer": "EMAMinMaxObserver",

                # Use the selected fakequant
                "w_fakequantize": w_fq,
                "a_fakequantize": a_fq,

                # Schemes (this branch expects 'symmetry' key)
                "w_qscheme": {
                    "bit": w_bits,
                    "symmetry": True,
                    "per_channel": True,
                    "pot_scale": False,
                },
                "a_qscheme": {
                    "bit": a_bits,
                    "symmetry": False,
                    "per_channel": False,
                    "pot_scale": False,
                },
            }
        }
        logging.info(f"[Academic extra_config] {extra_config}")

        model = prepare_by_platform(
            model,
            backend,
            prepare_custom_config_dict=extra_config,
            is_qat=False,        # PTQ path
            freeze_bn=True,
        )

        model = model.to(device).eval()
        post_all, post_quantish = count_quantish_modules(model)
        logging.info(f"Modules (total): {pre_all} -> {post_all}")
        logging.info(f"'Quantish' modules detected after prepare: {post_quantish}")
        if profile_mem:
            log_cuda_mem("after prepare_by_platform")


        

        with log_section("calibration (enable_calibration + forward)"):
            enable_calibration(model)
            seen_imgs, t0 = 0, time.time()
            iterator = calib_images_fn()
            if tqdm is not None and calib_steps:
                iterator = tqdm(iterator, total=calib_steps, desc="Calib", leave=False)
            for step, images in enumerate(iterator, 1):
                images = images.to(device, non_blocking=True)
                _ = model(images)
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
                adv_ns = make_adv_cfg(adv_cfg)
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
                model = ptq_reconstruction(model, stacked, adv_ns)
                model = model.to(device).eval()
                if profile_mem: log_cuda_mem("after ptq_reconstruction")

        with log_section("enable_quantization (simulate INT8)"):
            enable_quantization(model)
            if profile_mem: log_cuda_mem("after enable_quantization")

        return model

def resolve_fakequant_names(quant_model: str):
    """
    Map CLI-friendly names to MQBench fake quant class strings.
    """
    qm = quant_model.lower()
    if qm == "fixed":
        return "FixedFakeQuantize", "FixedFakeQuantize"
    if qm == "learnable":
        return "LearnableFakeQuantize", "LearnableFakeQuantize"
    if qm == "lsq":
        return "LSQFakeQuantize", "LSQFakeQuantize"
    if qm == "lsqplus":
        return "LSQPlusFakeQuantize", "LSQPlusFakeQuantize"
    # fallback
    return "FixedFakeQuantize", "FixedFakeQuantize"


# =========================
# Script entry
# =========================
def main():
    import argparse, random, numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--val_root", default="/home/alz07xz/imagenet/val",
                   help="Path to ImageNet val (folder with class subdirs).")
    p.add_argument("--model", default="resnet18",
                   help="torchvision model name (e.g., resnet18, mobilenet_v2, vit_b_16)")
    p.add_argument("--weights", default="DEFAULT",
                   help="Weights enum for the model (DEFAULT | NONE | e.g., IMAGENET1K_V1)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--calib_batches", type=int, default=32, help="How many batches to calibrate with")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--advanced", action="store_true",
                   help="Use advanced PTQ reconstruction (AdaRound/BRECQ/QDrop)")
    p.add_argument("--log_file", default=None, help="Write logs to this file (in addition to console)")
    p.add_argument("--verbose", action="store_true", help="DEBUG logging")
    p.add_argument("--log_interval", type=int, default=10, help="Log every N calibration steps")
    p.add_argument("--profile_mem", action="store_true", help="Log CUDA memory at key steps")
    p.add_argument("--no_structure_check", action="store_true", help="Skip ImageNet val/ structure check")
    p.add_argument("--seed", type=int, default=123, help="Set random seed for reproducibility")
    p.add_argument("--w_bits", type=int, default=8)
    p.add_argument("--a_bits", type=int, default=8)
    p.add_argument("--w_per_channel", action="store_true", default=True)
    p.add_argument("--a_per_channel", action="store_true", default=False)
    p.add_argument("--w_sym", action="store_true", default=True)
    p.add_argument("--a_sym", action="store_true", default=False)
    p.add_argument("--quant_model",
               choices=["fixed", "learnable", "lsq", "lsqplus"],
               default="fixed",
               help="FakeQuant type for weights/activations")
    p.add_argument("--adv_mode", choices=["adaround", "brecq", "qdrop"], default="adaround",
               help="Choose advanced PTQ method. If set, --advanced is implied.")
    p.add_argument("--adv_steps", type=int, default=20000, help="Optimization steps (max_count)")
    p.add_argument("--adv_warmup", type=float, default=0.2, help="Warm-up ratio in [0,1]")
    p.add_argument("--adv_lambda", type=float, default=0.01, help="Reconstruction loss weight")
    p.add_argument("--adv_prob", type=float, default=1.0,
                help="Drop prob for QDrop (set <1.0). Ignored for AdaRound/BRECQ.")
    p.add_argument("--keep_gpu", action="store_true",
                help="Keep stacked calibration tensors on GPU during reconstruction")

    # p.add_argument("--fp32_eval", action="store_true")  # optional switch if you want baseline eval
    args = p.parse_args()

    # If user picked a mode, enable advanced automatically
    if args.adv_mode is not None:
        args.advanced = True


    setup_logger(args.log_file, args.verbose)

    # Repro
    random.seed(args.seed)
    import numpy as np_alias  # avoid shadowing above
    np_alias.random.seed(args.seed)
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
        try:
            ds_len = len(val_loader.dataset)
        except Exception:
            ds_len = "unknown"
        logging.info(f"Val dataset size: {ds_len} | batch_size={args.batch_size} | calib_batches={args.calib_batches}")
        log_first_batch_stats(val_loader)

    def calib_images_fn():
        return calib_iter_builder(val_loader, args.calib_batches)

    # Optional FP32 eval (uncomment if you want it every run)
    # with log_section("evaluate FP32 baseline"):
    #     acc_fp = top1(fp, val_loader, device, log_prefix="EVAL_FP32")
    #     logging.info(f"[FP32] Top-1 = {acc_fp:.2f}% (expected ~{ref_top1})")

    # Run Academic PTQ
    adv_cfg = None
    if args.advanced:
        pattern = "layer" if args.adv_mode == "adaround" else "block"
        prob = args.adv_prob if args.adv_mode == "qdrop" else 1.0
        adv_cfg = make_adv_cfg(
            pattern=pattern,
            max_count=args.adv_steps,
            warm_up=args.adv_warmup,
            weight=args.adv_lambda,
            prob=prob,
            keep_gpu=args.keep_gpu,
        )

    qmodel = run_ptq(
        fp,
        calib_images_fn=calib_images_fn,
        device=device,
        do_advanced=args.advanced,
        calib_steps=args.calib_batches,
        log_interval=args.log_interval,
        profile_mem=args.profile_mem,
        quant_model=args.quant_model,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
    )

    with log_section("evaluate INT8-sim"):
        acc_q = top1(qmodel, val_loader, device, log_prefix="EVAL_INT8")
        logging.info(f"[PTQ][{args.model}][Academic]{' [ADV]' if args.advanced else ''} Top-1 = {acc_q:.2f}%")

if __name__ == "__main__":
    main()
