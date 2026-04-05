from copy import deepcopy


WORKLOAD_PROFILES = {
    "timeslice30_openloop_icclz1": {
        "profile_type": "timeslice",
        "validated_on": {
            "node": "icclz1",
            "gpu_name": "NVIDIA GeForce GTX 1080 Ti",
        },
        "args": {
            "run_style": "timeslice",
            "size": 2560,
            "period_ms": 100.0,
            "compute_budget_ms": 27.0,
            "warmup_seconds": 2.0,
            "cycle_report_every": 100,
        },
        "validation": {
            "metric": "avg_gpu_util_percent",
            "measured_mean": 29.667,
            "sampling_scope": "1s average utilization",
        },
    },
}


def get_workload_profile(name: str) -> dict:
    if name not in WORKLOAD_PROFILES:
        raise KeyError(f"Unknown workload profile: {name}")
    return deepcopy(WORKLOAD_PROFILES[name])
