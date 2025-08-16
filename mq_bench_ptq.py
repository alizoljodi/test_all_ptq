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
            stacked = []
            with torch.no_grad():
                for images in calib_images_fn():
                    stacked.append(images.to(device, non_blocking=True))
            adv_cfg = adv_cfg or {
                "pattern": "block", "scale_lr": 4e-5, "warm_up": 0.2, "weight": 0.01,
                "max_count": 20000, "b_range": [20, 2], "keep_gpu": True,
                "round_mode": "learned_hard_sigmoid", "prob": 1.0,
            }
            model = ptq_reconstruction(model, stacked, adv_cfg)
            # ★★★ ensure it’s still on device ★★★
            model = model.to(device).eval()
            if profile_mem: log_cuda_mem("after ptq_reconstruction")

    with log_section("enable_quantization (simulate INT8)"):
        enable_quantization(model)
        if profile_mem: log_cuda_mem("after enable_quantization")

    return model
