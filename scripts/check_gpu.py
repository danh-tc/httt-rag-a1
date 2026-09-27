"""Post-install sanity check: torch sees the GPU, the wheel ships kernels for its arch, and a matmul runs.

    python scripts/check_gpu.py
"""

import sys

import torch


def main() -> int:
    print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False (CPU-only wheel, or NVIDIA driver too old/missing)")
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    print(f"GPU 0: {name} ({arch}), {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

    # A wheel without kernels for this arch imports fine but fails on the first CUDA op.
    if arch not in torch.cuda.get_arch_list():
        print(f"FAIL: this torch build has no kernels for {arch}; supported: {torch.cuda.get_arch_list()}")
        return 1

    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    print(f"matmul OK (checksum {(x @ x).sum().item():.1f})")

    import sentence_transformers  # noqa: F401  (heavy import; confirms the rest of requirements resolved)

    print("OK: environment ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
