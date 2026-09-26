import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description="UIE prior-utility experiments (CPU / CUDA-compatible PPU)")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Probe runtime and run a complete model forward/backward")
    doctor.add_argument("--device", default="auto")
    doctor.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    smoke = commands.add_parser("smoke", help="Synthetic end-to-end integration test")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    smoke.add_argument("--sampler", choices=["flow", "ddim"], default="flow")
    dl = commands.add_parser("download", help="Download author-linked data, without model/runtime installation")
    dl.add_argument("source", nargs="?")
    dl.add_argument("--list", action="store_true")
    dl.add_argument("--output", default="downloads")
    dl.add_argument("--extract-to")
    dl.add_argument("--url", help="Explicit authorized alternative Google Drive URL")
    extract = commands.add_parser("extract", help="Safely extract an already downloaded ZIP/TAR")
    extract.add_argument("archive")
    extract.add_argument("--output", required=True)
    prep = commands.add_parser("prepare", help="Pair by filename, create immutable portable split manifest")
    prep.add_argument("--layout", choices=["wwe", "pairs", "nonref"], required=True)
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--root", required=True)
    prep.add_argument("--input")
    prep.add_argument("--target")
    prep.add_argument("--output", required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--val-fraction", type=float, default=0.1)
    prep.add_argument("--test-fraction", type=float, default=0.1)
    prep.add_argument("--calibration-fraction", type=float, default=0.05)
    prep.add_argument("--validation-from-train", action="store_true")
    prep.add_argument("--allow-target-resize", action="store_true")
    prep.add_argument("--groups", help="JSON: relative image stem -> scene/sequence id")
    prep.add_argument("--depth", help="Aligned depth NPY directory; farther must mean larger")
    audit = commands.add_parser("audit")
    audit.add_argument("manifest")
    audit.add_argument("--target-manifest")
    tr = commands.add_parser("train")
    tr.add_argument("--config")
    for name in ["stage", "manifest", "output", "device", "precision", "init", "resume", "patch-mode", "flow-gate", "sampler"]:
        tr.add_argument("--"+name)
    for name in ["steps", "schedule-steps", "batch-size", "grad-accum", "workers", "size", "width", "seed", "nfe"]:
        tr.add_argument("--"+name, type=int)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--split", choices=["val", "calibration", "test"], default="test")
    ev.add_argument("--device", default="auto")
    ev.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ev.add_argument("--size", type=int, default=256, help="0 preserves original dimensions")
    ev.add_argument("--steps", type=int, default=4)
    ev.add_argument("--gate", choices=["auto", "utility", "learned", "all", "none", "fixed"], default="auto")
    ev.add_argument("--fixed-gate", type=float, default=0.5)
    ev.add_argument("--corruption", choices=["clean", "shift", "color", "invert", "missing"], default="clean")
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--diagnostics", action="store_true")
    ev.add_argument("--save-images", action="store_true")
    ev.add_argument("--source-manifest")
    ev.add_argument("--limit", type=int, default=0, help="debug subset only; 0 means entire split")
    cal = commands.add_parser("calibrate")
    cal.add_argument("--checkpoint", required=True)
    cal.add_argument("--manifest", required=True)
    cal.add_argument("--output", required=True)
    cal.add_argument("--device", default="auto")
    cal.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    cal.add_argument("--size", type=int, default=256)
    cal.add_argument("--steps", type=int, default=4)
    cal.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "download":
        from .download import SOURCES, download
        if args.list:
            print(json.dumps(SOURCES, indent=2))
        elif args.source:
            print(json.dumps(download(args.source, args.output, args.extract_to, args.url), indent=2))
        else:
            parser.error("download requires a source or --list")
    elif args.command == "extract":
        from .download import safe_extract
        safe_extract(args.archive, args.output)
    elif args.command == "prepare":
        from .prepare import prepare
        prepare(args)
    elif args.command == "audit":
        from .data import audit_manifest, audit_cross_manifests
        print(json.dumps(audit_manifest(args.manifest), indent=2))
        if args.target_manifest:
            print(json.dumps(audit_cross_manifests(args.manifest, args.target_manifest), indent=2))
    elif args.command == "train":
        from .engine import train, configuration
        values = vars(args).copy()
        values.pop("command")
        path = values.pop("config")
        print(train(configuration(path, values)))
    elif args.command == "evaluate":
        from .evaluate import evaluate
        values = vars(args).copy()
        values.pop("command")
        values["device_name"] = values.pop("device")
        values["gate_mode"] = values.pop("gate")
        print(json.dumps(evaluate(**values), indent=2))
    elif args.command == "calibrate":
        from .evaluate import calibrate
        values = vars(args).copy()
        values.pop("command")
        values["device_name"] = values.pop("device")
        print(json.dumps(calibrate(**values), indent=2))
    elif args.command == "smoke":
        from .smoke import smoke
        smoke(args.output, args.device, args.precision, args.sampler)
    elif args.command == "doctor":
        import torch
        from .models import RestorationSystem
        from .runtime import device_for, environment_report, autocast
        torch.set_num_threads(2)
        report = environment_report()
        device = device_for(args.device)
        model = RestorationSystem(width=8).to(device)
        x = torch.rand(1, 3, 32, 40, device=device)
        with autocast(device, args.precision):
            output, _ = model.restore(x, steps=2, gate_mode="utility")
            loss = (output-x).square().mean()
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError("Doctor model loss is non-finite")
        report.update(forward_backward_passed=True, tested_precision=args.precision,
                      tested_device=str(device), loss=float(loss))
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
