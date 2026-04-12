#!/usr/bin/env python3
import csv
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

from gpu_supervisor_80 import (
    read_gpu_metrics,
    activate_mode,
    wait_for_mode_effect,
    gpu_state,
    choose_mode,
    start_workload,
    stop_process,
    summarize_rows,
)
from gpu_scenario_runner_auto import (
    summarize_phase,
    make_plot,
    cc_get_devices,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", type=float, default=80.0)
    ap.add_argument("--band", type=float, default=5.0)
    ap.add_argument("--crit-temp", type=float, default=95.0)

    ap.add_argument("--warmup-seconds", type=int, default=120)
    ap.add_argument("--normal-hold-seconds", type=int, default=600)
    ap.add_argument("--fault-hold-seconds", type=int, default=300)
    ap.add_argument("--stable-seconds", type=int, default=30)
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--min-dwell-seconds", type=int, default=15)

    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--period-ms", type=int, default=100)

    ap.add_argument("--baseline-mode", default="GPU_DEFAULT")
    ap.add_argument("--fault-start-mode", default="GPU_FAULT_5")
    ap.add_argument("--restore-mode", default="GPU_DEFAULT")

    ap.add_argument("--cc-base", default="http://localhost:11987")
    ap.add_argument("--cc-token", default="")

    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("fan_control_lab/logs") / args.tag
    if run_dir.exists():
        raise FileExistsError(f"run_dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "tag": args.tag,
        "started_at": ts,
        "target": args.target,
        "band": args.band,
        "crit_temp": args.crit_temp,
        "warmup_seconds": args.warmup_seconds,
        "normal_hold_seconds": args.normal_hold_seconds,
        "fault_hold_seconds": args.fault_hold_seconds,
        "stable_seconds": args.stable_seconds,
        "sample_interval": args.sample_interval,
        "min_dwell_seconds": args.min_dwell_seconds,
        "size": args.size,
        "duty": args.duty,
        "period_ms": args.period_ms,
        "baseline_mode": args.baseline_mode,
        "fault_start_mode": args.fault_start_mode,
        "restore_mode": args.restore_mode,
        "cc_base": args.cc_base,
        "has_cc_token": bool(args.cc_token),
        "choice": "A_simple_cycle",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    thermal_fieldnames = [
        "timestamp",
        "elapsed_s",
        "phase",
        "binary_label",
        "phase_elapsed_s",
        "gpu_temp_c",
        "gpu_state",
        "gpu_util_pct",
        "gpu_power_w",
        "gpu_fan_pct",
        "gpu_clock_mhz",
        "gpu_mem_clock_mhz",
        "current_mode",
        "desired_mode",
        "reason",
        "infeasible",
        "target_temp_c",
        "band_c",
        "stable_counter_s",
    ]

    event_fieldnames = [
        "timestamp",
        "elapsed_s",
        "phase",
        "event",
        "detail",
    ]

    thermal_fp = (run_dir / "thermal.csv").open("w", newline="", encoding="utf-8")
    thermal_writer = csv.DictWriter(thermal_fp, fieldnames=thermal_fieldnames)
    thermal_writer.writeheader()
    thermal_fp.flush()

    events_fp = (run_dir / "events.csv").open("w", newline="", encoding="utf-8")
    events_writer = csv.DictWriter(events_fp, fieldnames=event_fieldnames)
    events_writer.writeheader()
    events_fp.flush()

    total_runtime = (
        args.warmup_seconds
        + args.normal_hold_seconds
        + args.fault_hold_seconds
        + 1800
    )

    workload_proc, workload_stdout, workload_stderr = start_workload(
        run_dir,
        total_runtime,
        args.size,
        args.duty,
        args.period_ms,
    )

    rows = []
    events = []

    current_mode = None
    entered_band_once = False

    def log_event(phase, elapsed_s, event, detail=""):
        evt = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(elapsed_s, 3),
            "phase": phase,
            "event": event,
            "detail": detail,
        }
        events.append(evt)
        events_writer.writerow(evt)
        events_fp.flush()
        print(evt)

    def maybe_log_cc_devices(phase, row):
        if not args.cc_token:
            return
        try:
            devices = cc_get_devices(args.cc_base, args.cc_token)
            with (run_dir / "cc_devices.jsonl").open("a", encoding="utf-8") as rf:
                rf.write(json.dumps({
                    "timestamp": row["timestamp"],
                    "phase": phase,
                    "devices": devices
                }, ensure_ascii=False) + "\n")
        except Exception as e:
            with (run_dir / "cc_devices.jsonl").open("a", encoding="utf-8") as rf:
                rf.write(json.dumps({
                    "timestamp": row["timestamp"],
                    "phase": phase,
                    "error": str(e)
                }, ensure_ascii=False) + "\n")

    def collect_row(phase, binary_label, phase_start_ts, current_mode, desired_mode, reason, stable_counter_s, infeasible=""):
        g = read_gpu_metrics()
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - start_time, 3),
            "phase": phase,
            "binary_label": binary_label,
            "phase_elapsed_s": round(time.time() - phase_start_ts, 3),
            "gpu_temp_c": g["gpu_temp_c"],
            "gpu_state": gpu_state(g["gpu_temp_c"], args.target, args.band),
            "gpu_util_pct": g["gpu_util_pct"],
            "gpu_power_w": g["gpu_power_w"],
            "gpu_fan_pct": g["gpu_fan_pct"],
            "gpu_clock_mhz": g["gpu_clock_mhz"],
            "gpu_mem_clock_mhz": g["gpu_mem_clock_mhz"],
            "current_mode": current_mode,
            "desired_mode": desired_mode,
            "reason": reason,
            "infeasible": infeasible,
            "target_temp_c": args.target,
            "band_c": args.band,
            "stable_counter_s": round(stable_counter_s, 3),
        }
        rows.append(row)
        thermal_writer.writerow(row)
        thermal_fp.flush()
        print(row)
        maybe_log_cc_devices(phase, row)
        return row

    start_time = time.time()

    try:
        # ---------------------------
        # Phase A: warmup
        # ---------------------------
        phase = "warmup"
        binary_label = "exclude"
        activate_mode(args.baseline_mode)
        ok = wait_for_mode_effect(args.baseline_mode, timeout=20)
        current_mode = args.baseline_mode
        log_event(phase, 0.0, "activate_mode", f"{args.baseline_mode}, ok={ok}")
        log_event(phase, 0.0, "phase_start", "")

        phase_start = time.time()
        while time.time() - phase_start < args.warmup_seconds:
            collect_row(
                phase=phase,
                binary_label=binary_label,
                phase_start_ts=phase_start,
                current_mode=current_mode,
                desired_mode=current_mode,
                reason="warmup_default",
                stable_counter_s=0.0,
            )
            time.sleep(args.sample_interval)
        log_event(phase, time.time() - start_time, "phase_end", "")

        # ---------------------------
        # Phase B: normal_hold
        # ---------------------------
        phase = "normal_hold"
        binary_label = "normal"
        activate_mode(args.baseline_mode)
        ok = wait_for_mode_effect(args.baseline_mode, timeout=20)
        current_mode = args.baseline_mode
        log_event(phase, time.time() - start_time, "activate_mode", f"{args.baseline_mode}, ok={ok}")
        log_event(phase, time.time() - start_time, "phase_start", "")

        phase_start = time.time()
        while time.time() - phase_start < args.normal_hold_seconds:
            collect_row(
                phase=phase,
                binary_label=binary_label,
                phase_start_ts=phase_start,
                current_mode=current_mode,
                desired_mode=current_mode,
                reason="normal_default",
                stable_counter_s=0.0,
            )
            time.sleep(args.sample_interval)
        log_event(phase, time.time() - start_time, "phase_end", "")

        # ---------------------------
        # Phase C: fault_ramp_up
        # ---------------------------
        phase = "fault_ramp_up"
        binary_label = "transition"
        activate_mode(args.fault_start_mode)
        ok = wait_for_mode_effect(args.fault_start_mode, timeout=20)
        current_mode = args.fault_start_mode
        last_switch_ts = time.time()
        log_event(phase, time.time() - start_time, "activate_mode", f"{args.fault_start_mode}, ok={ok}")
        log_event(phase, time.time() - start_time, "phase_start", "")

        phase_start = time.time()
        stable_counter_s = 0.0
        in_band_prev = False

        while True:
            g = read_gpu_metrics()
            temp = g["gpu_temp_c"]

            desired_mode, reason = choose_mode(
                temp,
                current_mode,
                args.target,
                args.band,
                args.crit_temp,
            )

            since_switch = time.time() - last_switch_ts
            infeasible = ""
            if current_mode == "GPU_Cool_Max" and temp is not None and temp > args.target + args.band:
                infeasible = "infeasible_high_load"
            elif current_mode == "GPU_FAULT_5" and temp is not None and temp < args.target - args.band:
                infeasible = "infeasible_low_load"

            if desired_mode != current_mode and since_switch >= args.min_dwell_seconds:
                activate_mode(desired_mode)
                ok = wait_for_mode_effect(desired_mode, timeout=20)
                log_event(
                    phase,
                    time.time() - start_time,
                    "activate_mode",
                    f"{desired_mode}, reason={reason}, ok={ok}",
                )
                if ok:
                    current_mode = desired_mode
                    last_switch_ts = time.time()
                else:
                    log_event(
                        phase,
                        time.time() - start_time,
                        "mode_switch_failed",
                        desired_mode,
                    )

            state = gpu_state(temp, args.target, args.band)
            in_band_now = (state == "within_band")

            if in_band_now and not in_band_prev:
                log_event(phase, time.time() - start_time, "enter_band", "")
                entered_band_once = True
            if (not in_band_now) and in_band_prev:
                log_event(phase, time.time() - start_time, "exit_band", "")

            if in_band_now:
                stable_counter_s += args.sample_interval
            else:
                stable_counter_s = 0.0

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(time.time() - start_time, 3),
                "phase": phase,
                "binary_label": binary_label,
                "phase_elapsed_s": round(time.time() - phase_start, 3),
                "gpu_temp_c": g["gpu_temp_c"],
                "gpu_state": state,
                "gpu_util_pct": g["gpu_util_pct"],
                "gpu_power_w": g["gpu_power_w"],
                "gpu_fan_pct": g["gpu_fan_pct"],
                "gpu_clock_mhz": g["gpu_clock_mhz"],
                "gpu_mem_clock_mhz": g["gpu_mem_clock_mhz"],
                "current_mode": current_mode,
                "desired_mode": desired_mode,
                "reason": reason,
                "infeasible": infeasible,
                "target_temp_c": args.target,
                "band_c": args.band,
                "stable_counter_s": round(stable_counter_s, 3),
            }
            rows.append(row)
            thermal_writer.writerow(row)
            thermal_fp.flush()
            print(row)
            maybe_log_cc_devices(phase, row)

            in_band_prev = in_band_now

            if stable_counter_s >= args.stable_seconds:
                log_event(
                    phase,
                    time.time() - start_time,
                    "fault_hold_start",
                    f"stable_counter_s={stable_counter_s}",
                )
                break

            time.sleep(args.sample_interval)

        log_event(phase, time.time() - start_time, "phase_end", "")

        # ---------------------------
        # Phase D: fault_hold
        # ---------------------------
        phase = "fault_hold"
        binary_label = "abnormal"
        log_event(phase, time.time() - start_time, "phase_start", "")

        phase_start = time.time()
        stable_counter_s = 0.0
        in_band_prev = False

        while time.time() - phase_start < args.fault_hold_seconds:
            g = read_gpu_metrics()
            temp = g["gpu_temp_c"]

            desired_mode, reason = choose_mode(
                temp,
                current_mode,
                args.target,
                args.band,
                args.crit_temp,
            )

            since_switch = time.time() - last_switch_ts
            infeasible = ""
            if current_mode == "GPU_Cool_Max" and temp is not None and temp > args.target + args.band:
                infeasible = "infeasible_high_load"
            elif current_mode == "GPU_FAULT_5" and temp is not None and temp < args.target - args.band:
                infeasible = "infeasible_low_load"

            if desired_mode != current_mode and since_switch >= args.min_dwell_seconds:
                activate_mode(desired_mode)
                ok = wait_for_mode_effect(desired_mode, timeout=20)
                log_event(
                    phase,
                    time.time() - start_time,
                    "activate_mode",
                    f"{desired_mode}, reason={reason}, ok={ok}",
                )
                if ok:
                    current_mode = desired_mode
                    last_switch_ts = time.time()
                else:
                    log_event(
                        phase,
                        time.time() - start_time,
                        "mode_switch_failed",
                        desired_mode,
                    )

            state = gpu_state(temp, args.target, args.band)
            in_band_now = (state == "within_band")

            if in_band_now:
                stable_counter_s += args.sample_interval
            else:
                stable_counter_s = 0.0

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(time.time() - start_time, 3),
                "phase": phase,
                "binary_label": binary_label,
                "phase_elapsed_s": round(time.time() - phase_start, 3),
                "gpu_temp_c": g["gpu_temp_c"],
                "gpu_state": state,
                "gpu_util_pct": g["gpu_util_pct"],
                "gpu_power_w": g["gpu_power_w"],
                "gpu_fan_pct": g["gpu_fan_pct"],
                "gpu_clock_mhz": g["gpu_clock_mhz"],
                "gpu_mem_clock_mhz": g["gpu_mem_clock_mhz"],
                "current_mode": current_mode,
                "desired_mode": desired_mode,
                "reason": reason,
                "infeasible": infeasible,
                "target_temp_c": args.target,
                "band_c": args.band,
                "stable_counter_s": round(stable_counter_s, 3),
            }
            rows.append(row)
            thermal_writer.writerow(row)
            thermal_fp.flush()
            print(row)
            maybe_log_cc_devices(phase, row)

            in_band_prev = in_band_now
            time.sleep(args.sample_interval)

        log_event(phase, time.time() - start_time, "phase_end", "")

    except KeyboardInterrupt:
        log_event("interrupted", time.time() - start_time, "keyboard_interrupt", "")
    finally:
        stop_process(workload_proc)
        workload_stdout.close()
        workload_stderr.close()

        try:
            activate_mode(args.restore_mode)
            ok = wait_for_mode_effect(args.restore_mode, timeout=20)
            log_event("finalize", time.time() - start_time, "restore_mode", f"{args.restore_mode}, ok={ok}")
        except Exception as e:
            log_event("finalize", time.time() - start_time, "restore_mode_failed", str(e))

    thermal_fp.close()
    events_fp.close()

    summary = {
        "warmup": summarize_phase(rows, "warmup"),
        "normal_hold": summarize_phase(rows, "normal_hold"),
        "fault_ramp_up": summarize_phase(rows, "fault_ramp_up"),
        "fault_hold": summarize_phase(rows, "fault_hold"),
        "overall": summarize_rows(rows),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if rows:
        make_plot(rows, run_dir / "scenario_plot.png", args.target, args.band)

    (run_dir / "report.txt").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    print(f"\nRun complete: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
