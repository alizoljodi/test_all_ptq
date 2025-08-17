# mqbench_ptq_full.py  (Academic-only)
import os
import re
import time
import logging
import csv
from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Tuple

import torch
import torchvision as tv
import torchvision.transforms as T
from torchvision.models import get_model, get_model_weights
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

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
# --- replace your ConfigNamespace & make_adv_cfg with this ---
class AttrDict(dict):
    """Dict that also supports attribute access: d.x <-> d['x']"""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v
    def __delattr__(self, k):
        try:
            del self[k]
        except KeyError:
            raise AttributeError(k)
    def copy(self):
        return AttrDict(**self)

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
    cfg = {}
    if isinstance(user_cfg, dict):
        cfg.update(user_cfg)
    elif isinstance(user_cfg, AttrDict):
        cfg.update(dict(user_cfg))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return AttrDict(**{**defaults, **cfg})

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
def assert_imagenet_train_structure(train_root: str):
    """Ensure ImageNet train/ is split into 1,000 synset folders."""
    classes = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
    if len(classes) != 1000:
        raise RuntimeError(
            f"Expected 1000 class folders, found {len(classes)} at {train_root}. "
            "Reorganize ImageNet train into synset subfolders."
        )
    syn_pat = re.compile(r"^n\d{8}$")
    bad = [c for c in classes if not syn_pat.match(c)]
    if bad:
        raise RuntimeError(f"Found non-synset folder names: {bad[:5]} ...")
    logging.info("Train structure looks OK (1000 synset folders).")

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
    imagenet_root: str,
    weights,
    img_size: int = 224,
    calib_batches: int = 32,
    batch_size: int = 64,
    workers: int = 8,
    split="val",
):
    """
    Build data loaders for ImageNet train/val splits.
    
    Args:
        imagenet_root: Path to ImageNet root directory containing 'train' and 'val' folders
        weights: Model weights for transforms
        img_size: Input image size
        calib_batches: Number of batches for calibration
        batch_size: Batch size for data loading
        workers: Number of worker processes
        split: Which split to load ('train', 'val', or 'both')
    """
    # Determine which splits to load
    if split == "both":
        splits = ["train", "val"]
    else:
        splits = [split]
    
    loaders = {}
    
    for current_split in splits:
        split_path = os.path.join(imagenet_root, current_split)
        
        if not os.path.exists(split_path):
            raise RuntimeError(f"ImageNet {current_split} directory not found at {split_path}")
        
        # Use weights' transforms if available, otherwise use default
        if weights is not None and hasattr(weights, "transforms"):
            tfm = weights.transforms()
        else:
            # Default ImageNet transforms
            tfm = T.Compose([
                T.Resize(256),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Create dataset
        dataset = tv.datasets.ImageFolder(split_path, transform=tfm)
        
        # Create data loader
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(current_split == "train"),  # Shuffle training data
            num_workers=workers, 
            pin_memory=True
        )
        
        loaders[current_split] = loader
        
        # Log dataset info
        try:
            ds_len = len(dataset)
            num_classes = len(dataset.classes)
            logging.info(f"ImageNet {current_split}: {ds_len} images, {num_classes} classes")
        except Exception as e:
            logging.warning(f"Could not get {current_split} dataset info: {e}")
    
    # Create calibration iterator function
    def calib_iter(dl, n_batches):
        it = iter(dl)
        for _ in range(n_batches):
            try:
                images, _ = next(it)
            except StopIteration:
                it = iter(dl)  # cycle if calib > dataset
                images, _ = next(it)
            yield images
    
    # Return appropriate loaders based on split
    if split == "both":
        return loaders["train"], loaders["val"], calib_iter
    elif split == "train":
        return loaders["train"], calib_iter
    else:  # val
        return loaders["val"], calib_iter

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
        w_fq, a_fq = resolve_fakequant_names(quant_model, adv_enabled=do_advanced)  # <-- needs args in scope; see note below

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
                logging.info(f"[ADV] cfg={dict(adv_ns)}")  # instead of adv_ns.__dict__
                logging.info(f"[ADV] stacked tensors: {len(stacked)} | total calib images: {total_imgs}")
                if profile_mem: log_cuda_mem("before ptq_reconstruction")
                model = ptq_reconstruction(model, stacked, adv_ns)
                model = model.to(device).eval()
                if profile_mem: log_cuda_mem("after ptq_reconstruction")

        with log_section("enable_quantization (simulate INT8)"):
            enable_quantization(model)
            if profile_mem: log_cuda_mem("after enable_quantization")

        return model

def resolve_fakequant_names(quant_model: str, adv_enabled: bool):
    """
    Pick (w_fakequantize, a_fakequantize) class names for MQBench.
    In this commit, advanced PTQ requires a weight FQ that has .init(...).
    """
    if adv_enabled:
        # Your grep shows only AdaRoundFakeQuantize has `init(...)`
        return "AdaRoundFakeQuantize", "FixedFakeQuantize"

    qm = (quant_model or "fixed").lower()
    table = {
        "fixed":     ("FixedFakeQuantize",     "FixedFakeQuantize"),
        "learnable": ("LearnableFakeQuantize", "LearnableFakeQuantize"),
        "lsq":       ("LSQFakeQuantize",       "LSQFakeQuantize"),
        "lsqplus":   ("LSQPlusFakeQuantize",   "LSQPlusFakeQuantize"),
        "pact":      ("PACTFakeQuantize",      "PACTFakeQuantize"),
        "dsq":       ("DSQFakeQuantize",       "DSQFakeQuantize"),
        "tqt":       ("TqtFakeQuantize",       "TqtFakeQuantize"),
    }
    return table.get(qm, ("FixedFakeQuantize", "FixedFakeQuantize"))

def extract_model_logits(qmodel, fp_model, train_loader, device, max_batches: int = 10):
    """
    Extract logits from both quantized and full-precision models on training data.
    
    Args:
        qmodel: Quantized model
        fp_model: Full-precision model
        train_loader: Training data loader
        device: Device to run inference on
        max_batches: Maximum number of batches to extract
    
    Returns:
        Tuple of (quantized_logits, full_precision_logits)
    """
    qmodel.eval()
    fp_model.eval()
    
    all_q_logits = []
    all_fp_logits = []
    
    logging.info("Extracting logits from both models...")
    
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i >= max_batches:  # Limit to first max_batches batches for efficiency
                break
                
            images = images.to(device, non_blocking=True)
            
            # Get logits from quantized model
            q_logits = qmodel(images)
            all_q_logits.append(q_logits.cpu())
            
            # Get logits from full-precision model
            fp_logits = fp_model(images)
            all_fp_logits.append(fp_logits.cpu())
            
            if (i + 1) % 5 == 0:
                logging.info(f"Processed {i + 1} batches")
    
    # Concatenate all logits
    all_q = torch.cat(all_q_logits, dim=0)
    all_fp = torch.cat(all_fp_logits, dim=0)
    
    logging.info(f"Extracted logits: Q={all_q.shape}, FP={all_fp.shape}")
    
    return all_q, all_fp


# =========================
# Script entry
# =========================
def build_cluster_affine(all_q, all_fp, num_clusters=64, pca_dim=None):
        """
        Build cluster affine correction model from pre-extracted logits.
        """
        # Optional PCA for clustering only
        pca = None
        if pca_dim is not None and pca_dim < all_q.shape[1]:
            pca = PCA(n_components=pca_dim, random_state=42)
            q_features = pca.fit_transform(all_q.numpy())
        else:
            q_features = all_q.numpy()

        # Cluster quantized outputs
        cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_ids = cluster_model.fit_predict(q_features)

        # For each cluster: learn gamma, beta (per-class)
        gamma_dict = {}
        beta_dict = {}

        for cid in range(num_clusters):
            idxs = (cluster_ids == cid)
            if idxs.sum() == 0:
                # Empty cluster, default to identity
                gamma_dict[cid] = torch.ones(all_q.shape[1])
                beta_dict[cid] = torch.zeros(all_q.shape[1])
                continue

            q_c = all_q[idxs]  # [Nc, C]
            fp_c = all_fp[idxs]  # [Nc, C]

            # Closed-form least squares: fp ≈ gamma * q + beta
            mean_q = q_c.mean(dim=0)
            mean_fp = fp_c.mean(dim=0)

            # Compute variance, avoid div by zero
            var_q = q_c.var(dim=0, unbiased=False)
            var_q[var_q < 1e-8] = 1e-8

            gamma = ((q_c - mean_q) * (fp_c - mean_fp)).mean(dim=0) / var_q
            beta = mean_fp - gamma * mean_q

            gamma_dict[cid] = gamma
            beta_dict[cid] = beta

        return cluster_model, gamma_dict, beta_dict, pca

def check_existing_results(args, output_dir="results"):
    """
    Check for existing results and return the last saved state.
    
    Args:
        args: Command line arguments
        output_dir: Directory to check for existing results
    
    Returns:
        Tuple of (existing_results, last_timestamp, resume_from)
    """
    if not os.path.exists(output_dir):
        return [], None, None
    
    # Look for existing result files
    result_files = []
    for filename in os.listdir(output_dir):
        if filename.startswith("ptq_results_") and filename.endswith(".csv"):
            result_files.append(filename)
    
    if not result_files:
        return [], None, None
    
    # Get the most recent results file
    latest_file = max(result_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
    latest_path = os.path.join(output_dir, latest_file)
    
    # Extract timestamp from filename
    timestamp = latest_file.replace("ptq_results_", "").replace(".csv", "")
    
    # Read existing results
    existing_results = []
    try:
        with open(latest_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert string values back to appropriate types
                result = {
                    'alpha': float(row['alpha']),
                    'num_clusters': int(row['num_clusters']),
                    'pca_dim': int(row['pca_dim']),
                    'top1_accuracy': float(row['top1_accuracy']),
                    'top5_accuracy': float(row['top5_accuracy'])
                }
                existing_results.append(result)
        
        print(f"📁 Found existing results: {len(existing_results)} combinations")
        print(f"📄 Last results file: {latest_file}")
        print(f"⏰ Timestamp: {timestamp}")
        
        return existing_results, timestamp, latest_path
        
    except Exception as e:
        print(f"⚠️  Warning: Could not read existing results: {e}")
        return [], None, None


def get_remaining_combinations(args, existing_results):
    """
    Determine which parameter combinations still need to be run.
    
    Args:
        args: Command line arguments
        existing_results: List of already completed results
    
    Returns:
        List of remaining parameter combinations to run
    """
    # Get parameter lists
    alpha_list = args.alpha_list if args.alpha_list else [args.alpha]
    num_clusters_list = args.num_clusters_list if args.num_clusters_list else [args.num_clusters]
    pca_dim_list = args.pca_dim_list if args.pca_dim_list else [args.pca_dim]
    
    # Create set of completed combinations
    completed = set()
    for result in existing_results:
        completed.add((result['alpha'], result['num_clusters'], result['pca_dim']))
    
    # Find remaining combinations
    remaining = []
    for alpha in alpha_list:
        for num_clusters in num_clusters_list:
            for pca_dim in pca_dim_list:
                if (alpha, num_clusters, pca_dim) not in completed:
                    remaining.append({
                        'alpha': alpha,
                        'num_clusters': num_clusters,
                        'pca_dim': pca_dim
                    })
    
    return remaining


def save_results_to_csv(results, args, fp32_acc, baseline_ptq_acc, output_dir="results", append=False, existing_file=None):
    """
    Save all results to a CSV file with setup parameters and accuracies.
    
    Args:
        results: List of result dictionaries
        args: Command line arguments
        fp32_acc: Full-precision model accuracy
        baseline_ptq_acc: Baseline PTQ accuracy
        output_dir: Directory to save the CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"ptq_results_{timestamp}.csv")
    
    # Define CSV headers
    headers = [
        'timestamp', 'model', 'weights', 'batch_size', 'calib_batches', 'img_size',
        'w_bits', 'a_bits', 'quant_model', 'advanced', 'adv_mode', 'adv_steps',
        'adv_warmup', 'adv_lambda', 'adv_prob', 'keep_gpu', 'fp32_accuracy', 'baseline_ptq_acc',
        'alpha', 'num_clusters', 'pca_dim',
        'top1_accuracy', 'top5_accuracy', 'extract_logits', 'logits_batches'
    ]
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for result in results:
            # Create row with all arguments and results
            row = {
                'timestamp': timestamp,
                'model': args.model,
                'weights': args.weights,
                'batch_size': args.batch_size,
                'calib_batches': args.calib_batches,
                'img_size': args.img_size,
                'w_bits': args.w_bits,
                'a_bits': args.a_bits,
                'quant_model': args.quant_model,
                'advanced': args.advanced,
                'adv_mode': args.adv_mode,
                'adv_steps': args.adv_steps,
                'adv_warmup': args.adv_warmup,
                'adv_lambda': args.adv_lambda,
                'adv_prob': args.adv_prob,
                'keep_gpu': args.keep_gpu,
                'fp32_accuracy': fp32_acc, # Add baseline FP32 accuracy
                'baseline_ptq_acc': baseline_ptq_acc, # Add baseline PTQ accuracy
                'alpha': result['alpha'],
                'num_clusters': result['num_clusters'],
                'pca_dim': result['pca_dim'],
                'top1_accuracy': result['top1_accuracy'],
                'top5_accuracy': result['top5_accuracy'],
                'extract_logits': args.extract_logits,
                'logits_batches': args.logits_batches
            }
            writer.writerow(row)
    
    print(f"Results saved to: {csv_filename}")
    return csv_filename

def save_summary_csv(results, args, fp32_acc, baseline_ptq_acc, output_dir="results"):
    """
    Save a summary CSV with the best results for each parameter combination.
    
    Args:
        results: List of result dictionaries
        args: Command line arguments
        fp32_acc: Full-precision model accuracy
        baseline_ptq_acc: Baseline PTQ accuracy
        output_dir: Directory to save the CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_filename = os.path.join(output_dir, f"ptq_summary_{timestamp}.csv")
    
    # Group results by parameter combinations and find best for each
    summary_data = {}
    
    for result in results:
        # Create key for grouping (excluding alpha which we want to optimize)
        key = (result['num_clusters'], result['pca_dim'])
        
        if key not in summary_data or result['top1_accuracy'] > summary_data[key]['top1_accuracy']:
            summary_data[key] = result.copy()
    
    # Define summary CSV headers
    headers = [
        'timestamp', 'model', 'weights', 'batch_size', 'calib_batches', 'img_size',
        'w_bits', 'a_bits', 'quant_model', 'advanced', 'adv_mode', 'adv_steps',
        'adv_warmup', 'adv_lambda', 'adv_prob', 'keep_gpu', 'fp32_accuracy', 'baseline_ptq_acc',
        'best_alpha', 'num_clusters', 'pca_dim',
        'best_top1_accuracy', 'best_top5_accuracy', 'improvement_over_baseline', 'improvement_over_fp32'
    ]
    
    with open(summary_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for (num_clusters, pca_dim), best_result in summary_data.items():
            improvement_over_baseline = best_result['top1_accuracy'] - baseline_ptq_acc
            improvement_over_fp32 = best_result['top1_accuracy'] - fp32_acc
            
            row = {
                'timestamp': timestamp,
                'model': args.model,
                'weights': args.weights,
                'batch_size': args.batch_size,
                'calib_batches': args.calib_batches,
                'img_size': args.img_size,
                'w_bits': args.w_bits,
                'a_bits': args.a_bits,
                'quant_model': args.quant_model,
                'advanced': args.advanced,
                'adv_mode': args.adv_mode,
                'adv_steps': args.adv_steps,
                'adv_warmup': args.adv_warmup,
                'adv_lambda': args.adv_lambda,
                'adv_prob': args.adv_prob,
                'keep_gpu': args.keep_gpu,
                'fp32_accuracy': fp32_acc,
                'baseline_ptq_acc': baseline_ptq_acc,
                'best_alpha': best_result['alpha'],
                'num_clusters': num_clusters,
                'pca_dim': pca_dim,
                'best_top1_accuracy': best_result['top1_accuracy'],
                'best_top5_accuracy': best_result['top5_accuracy'],
                'improvement_over_baseline': improvement_over_baseline,
                'improvement_over_fp32': improvement_over_fp32
            }
            writer.writerow(row)
    
    print(f"Summary saved to: {summary_filename}")
    return summary_filename

def apply_cluster_affine(q_logits, cluster_model, gamma_dict, beta_dict, pca=None, alpha=0.4):
        """
        Apply per-cluster affine correction with optional PCA and alpha blending.
        """
        q_np = q_logits.cpu().numpy()

        # Apply same PCA as used during LUT building
        if pca is not None:
            q_np = pca.transform(q_np)

        cluster_ids = cluster_model.predict(q_np)

        corrected = []
        for i, q in enumerate(q_logits):
            cid = int(cluster_ids[i])
            gamma = gamma_dict[cid].to(q.device)
            beta = beta_dict[cid].to(q.device)
            affine_corrected = q * gamma + beta
            blended = q + alpha * (affine_corrected - q)
            corrected.append(blended)
        return torch.stack(corrected)

def evaluate_cluster_affine_with_alpha(q_model, fp_model, cluster_model, gamma_dict, beta_dict, dataloader, device, pca=None, alpha=0.4):
        q_model.eval()
        fp_model.eval()
        total_top1, total_top5, total = 0, 0, 0
        
        # Store logits for plotting
        all_q_logits = []
        all_fp_logits = []
        all_corrected_logits = []
        all_cluster_ids = []

        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                q_logits = q_model(images)
                fp_logits = fp_model(images)

                corrected_logits = apply_cluster_affine(q_logits, cluster_model, gamma_dict, beta_dict, pca=pca, alpha=alpha)


                acc1, acc5 = accuracy(corrected_logits, targets, topk=(1, 5))
                total_top1 += acc1.item() * images.size(0)
                total_top5 += acc5.item() * images.size(0)
                total += images.size(0)

        print(f"[Alpha={alpha:.2f}] Top-1 Accuracy: {total_top1 / total:.2f}%")
        print(f"[Alpha={alpha:.2f}] Top-5 Accuracy: {total_top5 / total:.2f}%")
        
        
        # Save logits data as CSV files
        
        
        return total_top1 / total, total_top5 / total
  
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def create_recovery_checkpoint(args, results, output_dir="results"):
    """
    Create a recovery checkpoint file to track progress.
    
    Args:
        args: Command line arguments
        results: Current results list
        output_dir: Directory to save checkpoint
    """
    checkpoint_file = os.path.join(output_dir, "recovery_checkpoint.json")
    checkpoint_data = {
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        'total_combinations': len(results),
        'completed_combinations': [f"{r['alpha']}_{r['num_clusters']}_{r['pca_dim']}" for r in results],
        'args': {
            'model': args.model,
            'w_bits': args.w_bits,
            'a_bits': args.a_bits,
            'quant_model': args.quant_model,
            'adv_mode': args.adv_mode,
            'alpha_list': args.alpha_list,
            'num_clusters_list': args.num_clusters_list,
            'pca_dim_list': args.pca_dim_list
        }
    }
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"💾 Recovery checkpoint saved: {checkpoint_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save recovery checkpoint: {e}")


def check_recovery_checkpoint(output_dir="results"):
    """
    Check if a recovery checkpoint exists and return its data.
    
    Args:
        output_dir: Directory to check for checkpoint
    
    Returns:
        Checkpoint data if exists, None otherwise
    """
    checkpoint_file = os.path.join(output_dir, "recovery_checkpoint.json")
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            print(f"📋 Found recovery checkpoint from {checkpoint_data['timestamp']}")
            print(f"📊 Completed combinations: {checkpoint_data['total_combinations']}")
            return checkpoint_data
        except Exception as e:
            print(f"⚠️  Warning: Could not read recovery checkpoint: {e}")
    return None

def main():
    import argparse, random, numpy as np_alias  # avoid shadowing above

    p = argparse.ArgumentParser()
    p.add_argument("--val_root", default="/home/alz07xz/imagenet",
                   help="Path to ImageNet root directory containing 'train' and 'val' folders")
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
    p.add_argument("--check_train_structure", action="store_true",default=True, help="Also check ImageNet train/ structure")
    p.add_argument("--extract_logits", action="store_true",default=True, help="Extract and save model logits for analysis")
    p.add_argument("--logits_batches", type=int, default=10, help="Number of batches to use for logits extraction")
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

    # PCA analysis arguments
    p.add_argument("--alpha", type=float, default=0.5,
                 help="Alpha parameter for PCA analysis (default: 0.5)")
    p.add_argument("--alpha_list", nargs='+', type=float, default=None,
                 help="List of alpha values for PCA analysis (overrides --alpha)")
    p.add_argument("--num_clusters", type=int, default=10,
                 help="Number of clusters for PCA analysis (default: 10)")
    p.add_argument("--num_clusters_list", nargs='+', type=int, default=None,
                 help="List of cluster numbers for PCA analysis (overrides --num_clusters)")
    p.add_argument("--pca_dim", type=int, default=50,
                 help="PCA dimension for analysis (default: 50)")
    p.add_argument("--pca_dim_list", nargs='+', type=int, default=None,
                 help="List of PCA dimensions for analysis (overrides --pca_dim)")
    
    # Output arguments
    p.add_argument("--output_dir", default="results", help="Directory to save CSV results (default: results)")
    p.add_argument("--save_csv", action="store_true", default=True, help="Save results to CSV files (default: True)")
    p.add_argument("--recover", action="store_true", help="Recover from existing results and resume incomplete experiments")
    
    # p.add_argument("--fp32_eval", action="store_true")  # optional switch if you want baseline eval
    args = p.parse_args()

    # If user picked a mode, enable advanced automatically
    if args.adv_mode is not None:
        args.advanced = True


    setup_logger(args.log_file, args.verbose)

    # Repro
    random.seed(args.seed)
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
            # Check val structure in the val subdirectory
            val_path = os.path.join(args.val_root, "val")
            assert_imagenet_val_structure(val_path)
            
            # Optionally check train structure
            if args.check_train_structure:
                train_path = os.path.join(args.val_root, "train")
                assert_imagenet_train_structure(train_path)
        
        # Build both train and val loaders
        train_loader, val_loader, calib_iter_builder = build_loaders(
            args.val_root,  # This should be the ImageNet root directory
            weights=weights, 
            img_size=args.img_size,
            calib_batches=args.calib_batches, 
            batch_size=args.batch_size, 
            workers=args.workers,
            split="both"  # Load both train and val splits
        )
        
        try:
            ds_len = len(val_loader.dataset)
        except Exception:
            ds_len = "unknown"
        logging.info(f"Val dataset size: {ds_len} | batch_size={args.batch_size} | calib_batches={args.calib_batches}")
        log_first_batch_stats(val_loader)

    def calib_images_fn():
        return calib_iter_builder(val_loader, args.calib_batches)

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
    
    # Store baseline PTQ accuracy
    baseline_ptq_acc = acc_q
    
    # Evaluate full-precision model for comparison
    with log_section("evaluate FP32 baseline"):
        acc_fp = top1(fp, val_loader, device, log_prefix="EVAL_FP32")
        logging.info(f"[FP32] Top-1 = {acc_fp:.2f}% (expected ~{ref_top1})")
    
    # Print baseline accuracies
    print(f"\n{'='*60}")
    print("BASELINE ACCURACIES (Before Clustering)")
    print(f"{'='*60}")
    print(f"  FP32 Model: {acc_fp:.2f}%")
    print(f"  Baseline PTQ: {baseline_ptq_acc:.2f}%")
    print(f"  PTQ Degradation: {acc_fp - baseline_ptq_acc:.2f}%")
    print(f"{'='*60}")
    
    # Optional logits extraction for analysis
    if args.extract_logits:
        with log_section("extract model logits"):
            print("Extracting logits from quantized and full-precision models...")
            all_q, all_fp = extract_model_logits(qmodel, fp, train_loader, device, max_batches=args.logits_batches)
            print("Logits extraction complete.")
            print(f"Quantized logits shape: {all_q.shape}")
            print(f"Full-precision logits shape: {all_fp.shape}")
        
            # Check for existing results if recovery mode is enabled
            existing_results, last_timestamp, resume_from = [], None, None
            if args.recover:
                existing_results, last_timestamp, resume_from = check_existing_results(args, args.output_dir)
                checkpoint_data = check_recovery_checkpoint(args.output_dir)
                if checkpoint_data:
                    print(f"📋 Checkpoint shows {checkpoint_data['total_combinations']} completed combinations")
            
            # Get remaining combinations to run
            remaining_combinations = get_remaining_combinations(args, existing_results)
            
            if args.recover and existing_results:
                print(f"\n🔄 RECOVERY MODE")
                print(f"📊 Existing results: {len(existing_results)} combinations")
                print(f"⏳ Remaining to run: {len(remaining_combinations)} combinations")
                print(f"📁 Resuming from: {resume_from}")
                
                if len(remaining_combinations) == 0:
                    print("✅ All combinations already completed! No need to run additional experiments.")
                    results = existing_results
                else:
                    print(f"🚀 Running remaining {len(remaining_combinations)} combinations...")
            else:
                print(f"🚀 Running all {len(remaining_combinations)} combinations...")
            
            # Initialize results list with existing results
            results = existing_results.copy()
            
            # Run remaining combinations
            try:
                for i, comb in enumerate(remaining_combinations, 1):
                    alpha = comb['alpha']
                    num_clusters = comb['num_clusters']
                    pca_dim = comb['pca_dim']
                    print(f"\n🔄 [{i}/{len(remaining_combinations)}] Running with alpha={alpha}, num_clusters={num_clusters}, pca_dim={pca_dim}")
                    cluster_model, gamma_dict, beta_dict, pca = build_cluster_affine(
                        all_q, all_fp, num_clusters=num_clusters, pca_dim=pca_dim)

                    # Evaluate with current parameters
                    top1_acc, top5_acc = evaluate_cluster_affine_with_alpha(
                        qmodel, fp, cluster_model, gamma_dict, beta_dict, val_loader, device, 
                        pca=pca, alpha=alpha
                    )
                    
                    # Store results
                    result = {
                        'alpha': alpha,
                        'num_clusters': num_clusters,
                        'pca_dim': pca_dim,
                        'top1_accuracy': top1_acc,
                        'top5_accuracy': top5_acc
                    }
                    results.append(result)
                    
                    print(f"✅ Result: Top-1: {top1_acc:.2f}%, Top-5: {top5_acc:.2f}%")
                    
                    # Save intermediate results every few combinations
                    if i % 5 == 0 or i == len(remaining_combinations):
                        print(f"💾 Saving intermediate results... ({len(results)} total combinations)")
                        if args.save_csv:
                            save_results_to_csv(results, args, acc_fp, baseline_ptq_acc, args.output_dir)
                            create_recovery_checkpoint(args, results, args.output_dir)
                            
            except KeyboardInterrupt:
                print(f"\n⚠️  Experiment interrupted by user (Ctrl+C)")
                print(f"💾 Saving current progress... ({len(results)} combinations completed)")
                if args.save_csv:
                    save_results_to_csv(results, args, acc_fp, baseline_ptq_acc, args.output_dir)
                    create_recovery_checkpoint(args, results, args.output_dir)
                print(f"🔄 You can resume later with: --recover")
                return
                
            except Exception as e:
                print(f"\n❌ Experiment crashed with error: {e}")
                print(f"💾 Saving current progress... ({len(results)} combinations completed)")
                if args.save_csv:
                    save_results_to_csv(results, args, acc_fp, baseline_ptq_acc, args.output_dir)
                    create_recovery_checkpoint(args, results, args.output_dir)
                print(f"🔄 You can resume later with: --recover")
                raise  # Re-raise the exception for debugging
        # Print summary of all results
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL RESULTS")
        print(f"{'='*80}")
        print(f"{'Alpha':<8} {'Clusters':<10} {'PCA_dim':<10} {'Top-1':<10} {'Top-5':<10}")
        print(f"{'-'*50}")
        
        for result in results:
            print(f"{result['alpha']:<8.2f} {result['num_clusters']:<10} {result['pca_dim']:<10} "
                f"{result['top1_accuracy']:<10.2f} {result['top5_accuracy']:<10.2f}")
        
        # Find best result
        best_result = max(results, key=lambda x: x['top1_accuracy'])
        print(f"\nBEST RESULT:")
        print(f"  Alpha: {best_result['alpha']}")
        print(f"  Clusters: {best_result['num_clusters']}")
        print(f"  PCA_dim: {best_result['pca_dim']}")
        print(f"  Top-1 Accuracy: {best_result['top1_accuracy']:.2f}%")
        print(f"  Top-5 Accuracy: {best_result['top5_accuracy']:.2f}%")
        
        # Print accuracy comparison
        print(f"\nACCURACY COMPARISON:")
        print(f"  FP32 Model: {acc_fp:.2f}%")
        print(f"  Baseline PTQ: {baseline_ptq_acc:.2f}%")
        print(f"  Best Clustering: {best_result['top1_accuracy']:.2f}%")
        print(f"  PTQ Degradation: {acc_fp - baseline_ptq_acc:.2f}%")
        print(f"  Clustering Recovery: {best_result['top1_accuracy'] - baseline_ptq_acc:.2f}%")
        print(f"  Final Gap to FP32: {acc_fp - best_result['top1_accuracy']:.2f}%")

    # Save results to CSV
    if args.save_csv:
        save_results_to_csv(results, args, acc_fp, baseline_ptq_acc, args.output_dir)
        save_summary_csv(results, args, acc_fp, baseline_ptq_acc, args.output_dir)
        
        # Create final recovery checkpoint
        if args.recover:
            create_recovery_checkpoint(args, results, args.output_dir)
            print(f"💾 Final recovery checkpoint saved with {len(results)} total combinations")

if __name__ == "__main__":
    main()
