"""Stage 01 — preflight.

Checks:
  * MPS available and a matmul works
  * Model weights can be downloaded / are cached
  * All Python deps importable
  * Writes results/local_pipeline/config_hash.txt if missing
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_pipeline import load_config, config_hash, results_dir, write_checkpoint


def main() -> None:
    cfg = load_config()
    import torch
    assert torch.backends.mps.is_available(), "MPS not available"
    x = torch.randn(64, 64, device="mps")
    _ = (x @ x.T).sum().item()
    print(f"MPS OK. torch={torch.__version__}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid = cfg["model"]["id"]
    print(f"Downloading / loading {mid}...")
    t0 = time.time()
    AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model.to("mps")
    print(f"Model on MPS. load_time={time.time() - t0:.1f}s  params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # Stash config hash for drift detection
    hp = results_dir() / "config_hash.txt"
    hp.write_text(config_hash())
    print(f"config_hash={config_hash()}  → {hp}")

    write_checkpoint("stage_01_preflight", "ok")
    print("stage 01 preflight: OK")


if __name__ == "__main__":
    main()
