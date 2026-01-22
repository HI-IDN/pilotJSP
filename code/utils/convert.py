import re
from pathlib import Path
from typing import Optional

from utils.instance import ProblemInstance


def pfsp_txt_to_instance(txt_path: str | Path) -> ProblemInstance:
    """
    Convert a PFSP txt file into a ProblemInstance.
    Keeps your existing parsing logic.
    """
    txt_path = Path(txt_path)

    LB_TAILLARD_RE = re.compile(r"Taillard LB\s*:\s*(\d+)")
    LB_PROP_RE     = re.compile(r"Proportionate LB\s*:\s*(\d+)")
    LB_MIN_RE      = re.compile(r"Lower bound:\s*(\d+)")
    SEED_RE        = re.compile(r"Random seed:\s*(\d+)")

    text = txt_path.read_text(encoding="utf-8")

    # Split non-empty lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    header = lines[0].split()
    n_jobs, n_machines = map(int, header[:2])

    # -----------------------
    # Processing-time matrix (machines-first)
    # -----------------------
    processing = []
    idx = 1
    for _ in range(n_machines):
        toks = lines[idx].split()
        idx += 1

        if len(toks) == 2 * n_jobs:
            row = [int(toks[2 * j + 1]) for j in range(n_jobs)]
        else:
            row = [int(t) for t in toks]

        processing.append(row)

    # -----------------------
    # Footer metadata
    # -----------------------
    meta = {
        "lower_bounds": {
            "taillard": None,
            "proportionate": None,
            "min": None,
        },
        "seed": None,
        "source_file": str(txt_path),
    }

    for l in lines[idx:]:
        m = LB_TAILLARD_RE.search(l)
        if m:
            meta["lower_bounds"]["taillard"] = int(m.group(1))
        m = LB_PROP_RE.search(l)
        if m:
            meta["lower_bounds"]["proportionate"] = int(m.group(1))
        m = LB_MIN_RE.search(l)
        if m:
            meta["lower_bounds"]["min"] = int(m.group(1))
        m = SEED_RE.search(l)
        if m:
            meta["seed"] = int(m.group(1))

    return ProblemInstance(
        problem_type="pfsp",
        n_jobs=n_jobs,
        n_machines=n_machines,
        processing_times=processing,  # machines-first
        meta=meta,
    )


def pfsp_txt_to_json(txt_path: str | Path, json_path: Optional[str | Path] = None):
    """
    Backwards-compatible convenience:
      - parse txt -> instance
      - if json_path given: save
      - else return dict
    """
    inst = pfsp_txt_to_instance(txt_path)
    if json_path:
        inst.save_json(json_path)
        return str(json_path)
    return inst.to_dict()
