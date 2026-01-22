
import json
import re
from pathlib import Path


def pfsp_txt_to_json(txt_path, json_path=None):
    """
    Convert a PFSP txt file into a JSON-compatible dict.
    """

    LB_TAILLARD_RE = re.compile(r"Taillard LB\s*:\s*(\d+)")
    LB_PROP_RE     = re.compile(r"Proportionate LB\s*:\s*(\d+)")
    LB_MIN_RE      = re.compile(r"Lower bound:\s*(\d+)")
    SEED_RE        = re.compile(r"Random seed:\s*(\d+)")

    text = Path(txt_path).read_text()

    # Split non-empty lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    header = lines[0].split()
    n_jobs, n_machines = map(int, header[:2])

    # -----------------------
    # Processing-time matrix
    # -----------------------
    processing = []
    idx = 1
    for _ in range(n_machines):
        toks = lines[idx].split()
        idx += 1

        if len(toks) == 2 * n_jobs:
            row = [int(toks[2*j + 1]) for j in range(n_jobs)]
        else:
            row = [int(t) for t in toks]

        processing.append(row)

    # -----------------------
    # Footer metadata
    # -----------------------
    meta = {
        "taillard_lb": None,
        "proportionate_lb": None,
        "lower_bound": None,
        "seed": None,
    }

    for l in lines[idx:]:
        for regex, key in [
            (LB_TAILLARD_RE, "taillard_lb"),
            (LB_PROP_RE, "proportionate_lb"),
            (LB_MIN_RE, "lower_bound"),
            (SEED_RE, "seed"),
        ]:
            m = regex.search(l)
            if m:
                meta[key] = int(m.group(1))

    data = {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "processing_times": processing,
        "lower_bounds": {
            "taillard": meta["taillard_lb"],
            "proportionate": meta["proportionate_lb"],
            "min": meta["lower_bound"],
        },
        "seed": meta["seed"],
    }

    if json_path:
        Path(json_path).parent.mkdir(exist_ok=True, parents=True)
        Path(json_path).write_text(json.dumps(data, indent=2))
        return json_path

    return data
