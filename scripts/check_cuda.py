"""Validate the exact interpreter used for training; no third-party imports before version check."""
import re
import sys


def check():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required; found {sys.version.split()[0]}")
    import torch
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__)
    if not match or not (2, 14, 1) <= tuple(map(int, match.groups())) < (2, 16, 0):
        raise RuntimeError(f"PyTorch 2.14.1–2.15.x required; found {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable in torch {torch.__version__}")
    a = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    (a @ a).sum().item()
    print(f"CUDA works: {torch.__version__}, {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    try:
        check()
    except Exception as exc:
        print(f"CUDA check failed for {sys.executable}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
