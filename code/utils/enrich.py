
def extend_json(data, methods):
    """
    Apply one or more extension methods to the JSON dictionary.

    `methods` is a list of strings identifying desired enrichments.
    """
    for m in methods:
        if m == "add_totals":
            add_totals(data)
        elif m == "transpose":
            transpose_to_jobs_first(data)
        else:
            raise ValueError(f"Unknown enrichment method: {m}")
    return data


def add_totals(data):
    """ Add total processing time per job. """
    machines = data["processing_times"]
    n_jobs = data["n_jobs"]
    totals = [sum(machines[m][j] for m in range(len(machines)))
              for j in range(n_jobs)]
    data["job_totals"] = totals


def transpose_to_jobs_first(data):
    """ Convert machines-first to jobs-first representation. """
    machines = data["processing_times"]
    data["processing_times_jobs_first"] = list(map(list, zip(*machines)))
