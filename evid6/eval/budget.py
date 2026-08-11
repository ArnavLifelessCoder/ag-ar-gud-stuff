"""EVID-6 compute accounting.

The plan makes a point of this: "4 GPU-hours on free-tier T4s" is a good line
in a workshop paper, but only if the number is measured rather than
remembered. This logs wall-clock GPU time per stage into a JSON file that
survives a Save Version, and accumulates across sessions.

Usage
-----
    from budget import stage
    with stage("qwen_cause", n_items=1500):
        run(...)

    print(report())
"""

import os
import json
import time
from contextlib import contextmanager

LOG_PATH = os.environ.get("EVID6_BUDGET_LOG", "/kaggle/working/gpu_budget.json")


def _load(path=None):
    path = path or LOG_PATH
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {"stages": []}


def _gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


@contextmanager
def stage(name: str, n_items: int = None, path: str = None):
    """Time one pipeline stage and append it to the budget log."""
    path = path or LOG_PATH
    t0 = time.time()
    device = _gpu_name()
    err = None
    try:
        yield
    except BaseException as e:              # record partial runs too
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        dt = time.time() - t0
        log = _load(path)
        log["stages"].append({
            "name": name,
            "device": device,
            "seconds": round(dt, 1),
            "hours": round(dt / 3600, 4),
            "n_items": n_items,
            "sec_per_item": round(dt / n_items, 3) if n_items else None,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": err,
        })
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
        except OSError:
            pass
        print(f"[budget] {name}: {dt/60:.1f} min on {device}"
              + (f" ({dt/n_items:.2f} s/item)" if n_items else ""))


def report(path: str = None) -> dict:
    """Totals, split by GPU and CPU. The GPU number is the one for the paper."""
    log = _load(path)
    gpu_h = sum(s["hours"] for s in log["stages"] if s["device"] != "cpu")
    cpu_h = sum(s["hours"] for s in log["stages"] if s["device"] == "cpu")
    by_stage = {}
    for s in log["stages"]:
        by_stage[s["name"]] = round(by_stage.get(s["name"], 0) + s["hours"], 4)
    return {
        "gpu_hours": round(gpu_h, 3),
        "cpu_hours": round(cpu_h, 3),
        "n_stages": len(log["stages"]),
        "by_stage": dict(sorted(by_stage.items(), key=lambda x: -x[1])),
        "weekly_quota_used": f"{gpu_h / 30:.1%}" if gpu_h else "0%",
        "paper_line": (f"The full pipeline consumed {gpu_h:.1f} GPU-hours on "
                       f"free-tier T4s." if gpu_h else None),
    }


def print_report(path: str = None):
    r = report(path)
    print("=" * 52)
    print("COMPUTE BUDGET")
    print("=" * 52)
    print(f"  GPU hours: {r['gpu_hours']:.2f}  ({r['weekly_quota_used']} of a 30 h week)")
    print(f"  CPU hours: {r['cpu_hours']:.2f}")
    for k, v in r["by_stage"].items():
        print(f"    {k:<28} {v:6.3f} h")
    if r["paper_line"]:
        print(f"\n  {r['paper_line']}")
    return r
