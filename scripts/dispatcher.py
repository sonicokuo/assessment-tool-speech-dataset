#!/usr/bin/env python3
"""dispatcher.py — autonomous job queue for the PSC nodes. Keeps nothing idle.

WHY THIS EXISTS
Nodes were sitting idle between jobs because every launch was manual: a job would finish, and
nothing started until someone noticed. This runs ON PSC in a loop, so progress continues whether
or not any interactive session is alive, and it survives disconnects.

HOW IT WORKS
Each job declares what it PRODUCES (a marker file) and what it NEEDS (marker files that must
already exist). The loop repeatedly:
  1. drops jobs whose `produces` already exists  (idempotent — safe to restart any time)
  2. finds jobs whose `needs` are all satisfied
  3. launches them onto any allocation with spare capacity
  4. records state to a status JSON that an outside watcher can read

Prerequisites are FILES, not job IDs, so a job that was run by hand still satisfies its
dependents, and a crashed job simply re-runs on the next pass.

BRANCHING ON RESULTS
Some jobs are gated on a NUMERIC outcome, not merely on a file existing (e.g. "only train MDN if
the oracle error ceiling is >= 0.30"). Those declare a `gate` callable that reads the upstream
JSON and returns True/False. A gate that returns False parks the job as SKIPPED with its reason
recorded, so the plan's decision tree executes itself instead of waiting for a human.

SAFETY
* never scancels an allocation — those are the author's.
* one launch per job per pass; a job already running (tracked by marker-in-progress) is not
  relaunched.
* every command is wrapped so its stdout lands in a per-job log under $SH/logs/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

SH = "/ocean/projects/cis260125p/shared"
RV = f"{SH}/repo_verify"
PY = f"{SH}/envs/project/bin/python"
ENV = f"export HF_HOME={SH}/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH={RV}/src"


def sh(cmd: str, timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:                                            # noqa: BLE001
        return ""


def live_jobids() -> list[str]:
    out = sh("squeue -u slin32 -h -o '%i %T' ")
    return [ln.split()[0] for ln in out.splitlines()
            if len(ln.split()) > 1 and ln.split()[1] == "RUNNING"]


def gpu_procs(jid: str) -> int:
    """How many processes are ACTUALLY on this allocation's GPU.

    Counting only the dispatcher's own launches is not enough: jobs started by hand (the
    6000-clip evals) are invisible to it, so it over-subscribed one allocation to 47 GB while
    another sat completely idle. Ask the GPU instead of trusting bookkeeping.
    """
    out = sh(f"srun --jobid={jid} --overlap nvidia-smi "
             f"--query-compute-apps=pid --format=csv,noheader 2>/dev/null", timeout=45)
    return len([ln for ln in out.splitlines() if ln.strip()])


# ---------------------------------------------------------------- result gates
def gate_ceiling_high(_):
    """Q9 (MDN / residual head) runs only if per-claim ranking is ACHIEVABLE.

    Reads the oracle error-ceiling result. >= 0.30 within-condition on any feature means the
    information exists and sigma is merely weak -> the upgrades are justified. Below that they
    would be chasing noise, so the job parks itself with the reason recorded.
    """
    p = f"{SH}/oracle_error_ceiling.json"
    if not os.path.exists(p):
        return None                                   # upstream not ready yet
    try:
        d = json.load(open(p))
    except Exception:                                 # noqa: BLE001
        return None
    best = max((v.get("within") or 0.0) for v in d.values() if isinstance(v, dict))
    return (best >= 0.30, f"max within-condition ceiling = {best:.3f}")


JOBS = [
    # ---- CPU, no GPU needed; these must never wait on a node ----
    dict(name="zeroed_overlap_recapture", gpu=True,
         produces=f"{SH}/saliency_zeroovl_test.npz", needs=[],
         cmd=f"{PY} -u scripts/capture_saliency.py "
             f"--checkpoint {SH}/checkpoints/full/l7audioonly_attnconcat_seed73/best.pt "
             f"--test_dir {SH}/data/processed_layer7/test "
             f"--out {SH}/saliency_zeroovl_test.npz --features f0_sd,f0_mean "
             f"--zero_overlap --limit 600"),

    dict(name="aux_sigma_dev", gpu=True,
         produces=f"{SH}/aux_sigma_s73_dev.json", needs=[],
         cmd=f"{PY} -u scripts/dump_aux_sigma.py "
             f"--checkpoint {SH}/checkpoints/full/l7audioonly_attnconcat_seed73/best.pt "
             f"--test_dir {SH}/data/processed_layer7/val "
             f"--out {SH}/aux_sigma_s73_dev.json"),

    dict(name="aux_readout_L24_audioonly", gpu=True,
         produces=f"{SH}/aux_l24audioonly_s73.json", needs=[],
         cmd=f"cd {SH} && {PY} -u aux_repool.py "
             f"{SH}/checkpoints/full/audioonly_attnconcat_seed73/best.pt "
             f"{SH}/data/processed_corrected/test "
             f"{SH}/data/features_corrected_merged/test.csv "
             f"{SH}/aux_l24audioonly_s73.json 0 zero_overlap"),

    # ---- gated on a NUMERIC upstream result ----
    dict(name="mdn_or_residual_head", gpu=True, gate=gate_ceiling_high,
         produces=f"{SH}/residual_head_s73.json",
         needs=[f"{SH}/oracle_error_ceiling.json", f"{SH}/aux_sigma_s73.json"],
         cmd=f"{PY} -u scripts/residual_error_head.py "
             f"--aux_sigma {SH}/aux_sigma_s73.json "
             f"--features_csv {SH}/data/features_corrected_merged/test.csv "
             f"--pt_dir {SH}/data/processed_layer7/test "
             f"--out {SH}/residual_head_s73.json"),

    # ---- Q1 METHODS SWEEP: the genre invariant. Every accepted comparator swept the METHODS
    # axis (min ~5, typically 8-11); we own the instrument and have never pointed it at a
    # population of methods. capture_saliency already implements three with the D1 sign fix
    # (signed sum for contribution-type methods, norm only for the sensitivity variant).
    dict(name="sweep_grad", gpu=True,
         produces=f"{SH}/sweep_grad_test.npz", needs=[],
         cmd=f"{PY} -u scripts/capture_saliency.py "
             f"--checkpoint {SH}/checkpoints/full/l7audioonly_attnconcat_seed73/best.pt "
             f"--test_dir {SH}/data/processed_layer7/test --method grad --zero_overlap "
             f"--features f0_sd,f0_mean,hnr,shimmer,jitter --limit 600 "
             f"--out {SH}/sweep_grad_test.npz"),

    dict(name="sweep_gradxinput", gpu=True,
         produces=f"{SH}/sweep_gradxinput_test.npz", needs=[],
         cmd=f"{PY} -u scripts/capture_saliency.py "
             f"--checkpoint {SH}/checkpoints/full/l7audioonly_attnconcat_seed73/best.pt "
             f"--test_dir {SH}/data/processed_layer7/test --method gradxinput --zero_overlap "
             f"--features f0_sd,f0_mean,hnr,shimmer,jitter --limit 600 "
             f"--out {SH}/sweep_gradxinput_test.npz"),

    dict(name="sweep_ig", gpu=True,
         produces=f"{SH}/sweep_ig_test.npz", needs=[],
         cmd=f"{PY} -u scripts/capture_saliency.py "
             f"--checkpoint {SH}/checkpoints/full/l7audioonly_attnconcat_seed73/best.pt "
             f"--test_dir {SH}/data/processed_layer7/test --method ig --zero_overlap "
             f"--features f0_sd,f0_mean,hnr,shimmer,jitter --limit 600 "
             f"--out {SH}/sweep_ig_test.npz"),

    # Scoring is gated on the captures: each map is scored against the EXACT oracle map with
    # trivial-null flooring and the resolution ceiling, so a number is only interpretable
    # relative to (ceiling - worst_null).
    dict(name="score_sweep_gradxinput", gpu=False,
         produces=f"{SH}/score_sweep_gradxinput.txt",
         needs=[f"{SH}/sweep_gradxinput_test.npz"],
         cmd=f"{PY} -u scripts/score_attribution.py --oracle {SH}/oracle_maps_test.npz "
             f"--model_maps {SH}/sweep_gradxinput_test.npz --feature f0_sd "
             f"> {SH}/score_sweep_gradxinput.txt"),

    dict(name="score_sweep_ig", gpu=False,
         produces=f"{SH}/score_sweep_ig.txt",
         needs=[f"{SH}/sweep_ig_test.npz"],
         cmd=f"{PY} -u scripts/score_attribution.py --oracle {SH}/oracle_maps_test.npz "
             f"--model_maps {SH}/sweep_ig_test.npz --feature f0_sd "
             f"> {SH}/score_sweep_ig.txt"),

    # ---- n=1 -> n=2 on the causal result. The f0_mean CAUSAL finding (b_int 0.377 vs b_within
    # 0.052) is currently ONE seed; the calibration says no accepted comparator had n=1 on its
    # flagship. Replicating on seed 42 is the cheapest breadth purchase available.
    # ---- THE MOST DANGEROUS UN-RUN BASELINE. Our AURC gains are measured against emit-always
    # only. The honest comparison is a ridge EQUIPPED with its own error predictor (a bare ridge
    # has 100% coverage and cannot abstain). If it wins the ill-posed panel, contribution III is
    # not a contribution on this corpus. A reviewer builds this in an afternoon — have it first.
    dict(name="uq_ridge_baseline", gpu=False,
         produces=f"{SH}/uq_ridge_baseline.json",
         needs=[f"{SH}/aux_sigma_s73.json"],
         cmd=f"{PY} -u scripts/uq_ridge_baseline.py "
             f"--aux_sigma {SH}/aux_sigma_s73.json "
             f"--features_csv {SH}/data/features_corrected_merged/test.csv "
             f"--pt_dir {SH}/data/processed_layer7/test "
             f"--out {SH}/uq_ridge_baseline.json"),

    # ---- CRITICAL PATH as of the G.2j retraction. All three shipping arms trained on `fw`
    # (checkpoint-verified), whose clean clips carry ZERO f0/hnr/shimmer supervision. Every
    # LM-side voice-panel number is therefore structurally untestable, not negative. This retrain
    # is the fix — and the SAME run also repairs the aux-MSE masking (train.py:1068) that leaves
    # those five heads trained on clean clips only, which is why they lose to the ridge 11/11.
    # The `.fw2.yaml` config points at the corrected targets for TRAINING, not just eval.
    dict(name="retrain_fw2_s73", gpu=True,
         produces=f"{SH}/checkpoints/full/l7audioonly_fw2_seed73/TRAINING_COMPLETE", needs=[],
         # RESUME-AWARE. 4 epochs is ~8h but allocations here are ~6-8h, so this WILL hit a
         # wall. train.py checkpoints `last.pt` per epoch; without --resume_from the dispatcher
         # would restart it from scratch on every relaunch and never finish. The shell picks the
         # flag at launch time, so the first run starts fresh and every relaunch continues.
         cmd=(f"L={SH}/checkpoints/full/l7audioonly_fw2_seed73/last.pt; "
              f"R=\"\"; [ -f \"$L\" ] && R=\"--resume_from $L\"; "
              f"{PY} -u src/train.py "
              f"--config {RV}/configs/config.l7audioonly.s73.fw2RETRAIN.yaml $R "
              f"&& touch {SH}/checkpoints/full/l7audioonly_fw2_seed73/TRAINING_COMPLETE")),

    # ---- DECIDES WHERE THE RETRAIN SHOULD AIM. Is the ridge deficit caused by SUPERVISION
    # MASKING (train.py:1068 drops 5 features on overlapped clips) or by POOLING (the ridge uses
    # mean+std; our head is mean-only, and f0_sd IS a variability statistic)? A 2x2 linear probe
    # on the same frozen features answers it with no training. CPU-only.
    dict(name="ridge_probe_2x2", gpu=False,
         produces=f"{SH}/ridge_probe_2x2.json", needs=[],
         cmd=f"{PY} -u scripts/ridge_probe_2x2.py "
             f"--train_dir {SH}/data/processed_layer7/train "
             f"--test_dir {SH}/data/processed_layer7/test "
             f"--train_csv {SH}/data/features_corrected_merged/train-100.csv "
             f"--test_csv {SH}/data/features_corrected_merged/test.csv "
             f"--out {SH}/ridge_probe_2x2.json"),

    # Second seed on the CORRECTED targets. Two seeds is the minimum for any stability claim,
    # and this fills the otherwise-idle allocation: the 2x2 ridge probe is CPU-only sklearn, so
    # it leaves that node's H100 completely unused.
    dict(name="retrain_fw2_s42", gpu=True,
         produces=f"{SH}/checkpoints/full/l7audioonly_fw2_seed42/TRAINING_COMPLETE", needs=[],
         cmd=(f"L={SH}/checkpoints/full/l7audioonly_fw2_seed42/last.pt; "
              f"R=\"\"; [ -f \"$L\" ] && R=\"--resume_from $L\"; "
              f"{PY} -u src/train.py "
              f"--config {RV}/configs/config.l7audioonly.s42.fw2RETRAIN.yaml $R "
              f"&& touch {SH}/checkpoints/full/l7audioonly_fw2_seed42/TRAINING_COMPLETE")),

    # ---- THE PAYOFF EVALS. Both fw2 retrains are done; these answer the two questions the
    # G.2j retraction opened: (1) do f0/hnr/shimmer now EMIT (they had ZERO clean-clip
    # supervision under fw), and (2) does the ILL-POSED panel close on the tuned L7 ridge
    # (0.5372) now that those heads are no longer trained on clean clips only?
    # aux_repool is the fast readout — minutes, and it gives the ridge comparison directly.
    dict(name="aux_readout_fw2_s73", gpu=True,
         produces=f"{SH}/aux_l7_fw2_s73.json",
         needs=[f"{SH}/checkpoints/full/l7audioonly_fw2_seed73/TRAINING_COMPLETE"],
         cmd=f"cd {SH} && {PY} -u aux_repool.py "
             f"{SH}/checkpoints/full/l7audioonly_fw2_seed73/best.pt "
             f"{SH}/data/processed_layer7/test "
             f"{SH}/data/features_corrected_merged/test.csv "
             f"{SH}/aux_l7_fw2_s73.json 0 zero_overlap"),

    dict(name="aux_readout_fw2_s42", gpu=True,
         produces=f"{SH}/aux_l7_fw2_s42.json",
         needs=[f"{SH}/checkpoints/full/l7audioonly_fw2_seed42/TRAINING_COMPLETE"],
         cmd=f"cd {SH} && {PY} -u aux_repool.py "
             f"{SH}/checkpoints/full/l7audioonly_fw2_seed42/best.pt "
             f"{SH}/data/processed_layer7/test "
             f"{SH}/data/features_corrected_merged/test.csv "
             f"{SH}/aux_l7_fw2_s42.json 0 zero_overlap"),

    # Emission is an LM-path question, so it needs the generation pass. 60 clean clips is
    # enough to separate 0% from ~100%; the full 6000-clip eval follows only if this shows
    # the features returning.
    dict(name="emission_check_fw2", gpu=True,
         produces=f"{SH}/temperature_redecode_fw2.json",
         needs=[f"{SH}/checkpoints/full/l7audioonly_fw2_seed73/TRAINING_COMPLETE"],
         cmd=f"{PY} -u scripts/temperature_redecode.py "
             f"--checkpoint {SH}/checkpoints/full/l7audioonly_fw2_seed73/best.pt "
             f"--config {RV}/configs/config.l7audioonly.s73.fw2RETRAIN.yaml "
             f"--test_dir {SH}/data/processed_layer7/test "
             f"--out {SH}/temperature_redecode_fw2.json --n 60 --temps 0.0"),

    # ---- THE ARM-ISOLATING COMPARISON. residual_error_head showed a post-hoc GBM beating our
    # LEARNED sigma head on 10/11 features (0.214 -> 0.300 within-condition), which CLOSES the
    # per-claim gap but moves the credit from the sigma head to the post-hoc predictor. The
    # reviewer's next question is immediate: give the RIDGE the same post-hoc predictor. Arms B
    # and C share clips, folds and GBM hyperparameters, so B-C isolates our REPRESENTATION.
    dict(name="abstention_3arm", gpu=False,
         produces=f"{SH}/abstention_3arm_s73.json",
         needs=[f"{SH}/aux_sigma_s73.json"],
         cmd=f"{PY} -u scripts/abstention_3arm.py "
             f"--aux_sigma {SH}/aux_sigma_s73.json "
             f"--features_csv {SH}/data/features_corrected_merged/test.csv "
             f"--pt_dir {SH}/data/processed_layer7/test "
             f"--out {SH}/abstention_3arm_s73.json"),

    # ---- G.2q: THE DECISIVE LEVER FOR THE ILL-POSED PANEL. The aux-MSE hedge mask supervises
    # f0_mean/f0_sd/jitter/shimmer/hnr on CLEAN CLIPS ONLY (overlap>=0.5 covers 99.1% of
    # mixtures), to avoid "mix-noisy" values that are BYTE-IDENTICAL to the clean twin's GT.
    # The ridge trains on 100% of clips and beats us on exactly those five (0.5153 vs 0.4605).
    # fw2 could not test this — the mask is overlap-driven, not target-driven. Prose/nums
    # hedging and the sigma head are untouched, so contribution 2's abstention is preserved.
    dict(name="retrain_unmasked_s73", gpu=True,
         produces=f"{SH}/checkpoints/full/l7audioonly_unmasked_seed73/TRAINING_COMPLETE",
         needs=[],
         cmd=(f"L={SH}/checkpoints/full/l7audioonly_unmasked_seed73/last.pt; "
              f"R=\"\"; [ -f \"$L\" ] && R=\"--resume_from $L\"; "
              f"{PY} -u src/train.py "
              f"--config {RV}/configs/config.l7audioonly.s73.unmasked.yaml $R "
              f"&& touch {SH}/checkpoints/full/l7audioonly_unmasked_seed73/TRAINING_COMPLETE")),

    # Matched readout: same command as the fw/fw2 rows, so the ill-posed panel is comparable
    # against the tuned L7 ridge (0.5153) on the identical 6000 clips.
    dict(name="aux_readout_unmasked_s73", gpu=True,
         produces=f"{SH}/aux_l7_unmasked_s73.json",
         needs=[f"{SH}/checkpoints/full/l7audioonly_unmasked_seed73/TRAINING_COMPLETE"],
         cmd=f"cd {SH} && {PY} -u aux_repool.py "
             f"{SH}/checkpoints/full/l7audioonly_unmasked_seed73/best.pt "
             f"{SH}/data/processed_layer7/test "
             f"{SH}/data/features_corrected_merged/test.csv "
             f"{SH}/aux_l7_unmasked_s73.json 0 zero_overlap"),

    # ---- CONTRIBUTION 1's PANEL, MEASURABLE FOR THE FIRST TIME. Under `fw` the LM emitted
    # NOTHING for hnr/f0_mean/f0_sd/shimmer (0.0%, G.2o), so a free-decode coverage column on
    # the ill-posed panel was meaningless — it measured a target bug. fw2 restores all four to
    # 98.3% (G.2r), so band-free SRCC + nMAE + COVERAGE vs INSTRUMENT GT is now well-defined on
    # all 11 features. Sorted order interleaves each mixture with its _s1clean twin, so a
    # 1500-clip prefix is ~750 of each rather than one condition.
    # ⚠️ THE MARKER, NOT THE JSON. inference.py flushes every 50 clips (atomic tmp->rename), so
    # the .json EXISTS while the run is 50/1500 done. Depending on it meant the scorer would fire
    # on a partial file and emit a confident 50-clip panel; `produces` pointing at it also meant
    # a crashed run would never relaunch. The marker is written only on a zero-exit run.
    # Relaunch is safe either way: inference.py auto-resumes, skipping already-scored filenames.
    dict(name="freedecode_fw2_s73", gpu=True,
         produces=f"{SH}/freedecode_fw2_s73.DONE", needs=[],
         cmd=f"{PY} -u src/inference.py "
             f"--config {RV}/configs/config.l7audioonly.s73.fw2RETRAIN.yaml "
             f"--checkpoint {SH}/checkpoints/full/l7audioonly_fw2_seed73/best.pt "
             f"--test_dir {SH}/data/processed_layer7/test --top_k 1 --start 0 --end 1500 "
             f"--out {SH}/freedecode_fw2_s73.json "
             f"&& touch {SH}/freedecode_fw2_s73.DONE"),

    # Scored against the INSTRUMENT CSV (the default), so the parser touches predictions only
    # and no value is laundered through the verbalizer into "truth".
    dict(name="score_freedecode_fw2", gpu=False,
         produces=f"{SH}/freedecode_fw2_s73.scored.txt",
         needs=[f"{SH}/freedecode_fw2_s73.DONE"],
         cmd=f"{PY} -u scripts/score_matched_test.py "
             f"--features_csv {SH}/data/features_corrected_merged/test.csv "
             f"{SH}/data/descriptions_corrected_fw2.json {SH}/freedecode_fw2_s73.json "
             f"> {SH}/freedecode_fw2_s73.scored.txt 2>&1"),

    dict(name="pitch_probe_seed42", gpu=True,
         produces=f"{SH}/pitch_intervention_s42.json", needs=[],
         cmd=f"{PY} -u scripts/pitch_intervention.py "
             f"--checkpoint {SH}/checkpoints/full/l7audioonly_attnconcat_seed42/best.pt "
             f"--test_dir {SH}/data/processed_layer7/test "
             f"--clean_dir {SH}/data/audio_corrected/test-s1clean "
             f"--out {SH}/pitch_intervention_s42.json --n 40"),
]


JOB_BY_NAME = {j['name']: j for j in JOBS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu_jobs", type=int, default=3,
                    help="max concurrent CPU-only jobs (not GPU-gated)")
    ap.add_argument("--per_node", type=int, default=2, help="max concurrent jobs per allocation")
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    state_path = f"{SH}/dispatcher_state.json"
    running: dict[str, str] = {}                        # job name -> jobid
    attempts: dict[str, int] = {}                       # job name -> launches so far
    # Persisted launch times (job name -> epoch seconds). In-memory state dies with a
    # dispatcher restart, which is exactly when duplicate launches happen.
    ledger_path = f"{SH}/logs/dispatch_ledger.json"
    try:
        ledger: dict[str, float] = json.load(open(ledger_path))
    except Exception:                                   # noqa: BLE001
        ledger = {}
    failed: dict[str, str] = {}                         # job name -> why it was given up on
    MAX_ATTEMPTS = 2

    def in_flight(job, stale_after: int = 2400) -> bool:
        """Is this job ALREADY running, possibly launched by a previous dispatcher instance?

        Restarting the dispatcher must not duplicate work: a second copy of a job wastes a node
        and, for jobs that write one JSON, two writers can race on the same path. State lives in
        memory, so it does not survive a restart — instead infer liveness from the job's log
        being RECENTLY WRITTEN. A live job prints progress; a finished one has its marker; a dead
        one goes stale and becomes eligible again.
        """
        # ⚠️ A MISSING LOG MUST NOT RESURRECT A LIVE JOB. Log mtime alone was the only liveness
        # signal, so removing two logs by hand spawned a SECOND `abstention_3arm` beside the
        # running one, both writing the same --out path. The launch ledger below is persisted
        # across dispatcher restarts and cannot be affected by log manipulation.
        #
        # NOT pgrep: `pgrep -f <pattern>` excludes only ITSELF, not the `sh -c` wrapper
        # subprocess spawns — whose argv contains the pattern. Every check would return >=1,
        # every job would look permanently in flight, and the dispatcher would deadlock and
        # never launch anything again. Verified before shipping; do not reintroduce it.
        started = ledger.get(job["name"])
        if started and (time.time() - started) < stale_after and not crashed(job):
            return True
        log = f"{SH}/logs/dispatch_{job['name']}.log"
        if not os.path.exists(log):
            return False
        # A CRASHED job is never in flight, however recently its log was touched. The
        # termination message itself refreshes the mtime, so mtime alone said "alive" for the
        # full staleness window while crashed() said "dead" — the job was popped and re-skipped
        # every pass, producing an infinite [retry] loop that never relaunched anything.
        if crashed(job):
            return False
        return (time.time() - os.path.getmtime(log)) < stale_after

    def crashed(job) -> str | None:
        """A job that exits WITHOUT writing its marker has crashed. Relaunching once is right
        (nodes die, files lock); relaunching forever is a loop on a broken script. Report the
        actual error so a human reads a cause, not a silence."""
        log = f"{SH}/logs/dispatch_{job['name']}.log"
        if not os.path.exists(log):
            return None
        try:
            tail = open(log, errors="ignore").read()[-4000:]
        except Exception:                                # noqa: BLE001
            return None
        # "forcing job termination" is included because a killed job WRITES that line, which
        # refreshes its log mtime and would otherwise make `in_flight()` treat a corpse as
        # running for the full staleness window.
        for marker in ("Traceback (most recent call last)", "[fatal]",
                       "CUDA out of memory", "srun: error",
                       "forcing job termination", "slurmstepd: error"):
            if marker in tail:
                line = next((ln.strip() for ln in reversed(tail.splitlines())
                             if marker.split("(")[0].strip() in ln or ln.startswith(marker)),
                            marker)
                return line[:200]
        return None

    while True:
        ids = live_jobids()
        if not ids:
            print("[dispatch] no RUNNING allocations; waiting", flush=True)
        # a job is finished when its marker exists; drop it from the running set
        for nm in list(running):
            j = next((x for x in JOBS if x["name"] == nm), None)
            if not j:
                running.pop(nm, None)
                continue
            if os.path.exists(j["produces"]):
                print(f"[done] {nm}", flush=True)
                running.pop(nm, None)
                continue
            why = crashed(j)
            if why:
                running.pop(nm, None)
                if attempts.get(nm, 0) >= MAX_ATTEMPTS:
                    failed[nm] = why
                    print(f"[FAILED] {nm} after {attempts[nm]} attempts: {why}", flush=True)
                else:
                    print(f"[retry] {nm} (attempt {attempts.get(nm, 0)}): {why}", flush=True)

        # Capacity from MEASURED occupancy, so hand-launched work counts too.
        slots = {i: max(0, a.per_node - gpu_procs(i)) for i in ids}
        print(f"[capacity] {slots}", flush=True)

        status = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "allocations": ids,
                  "running": dict(running), "done": [], "skipped": {}, "pending": [],
                  "failed": dict(failed), "attempts": dict(attempts)}

        for j in JOBS:
            nm = j["name"]
            if os.path.exists(j["produces"]):
                status["done"].append(nm)
                continue
            if nm in running or nm in failed:
                continue
            if in_flight(j):
                # adopted from a previous dispatcher instance (or this one, pre-restart)
                running.setdefault(nm, "adopted")
                status["running"][nm] = "adopted"
                continue
            if not all(os.path.exists(n) for n in j["needs"]):
                status["pending"].append(nm)
                continue
            g = j.get("gate")
            if g:
                res = g(j)
                if res is None:
                    status["pending"].append(nm)
                    continue
                ok, why = res
                if not ok:
                    status["skipped"][nm] = why
                    print(f"[skip] {nm}: {why}", flush=True)
                    continue
                print(f"[gate-pass] {nm}: {why}", flush=True)
            # CPU-ONLY JOBS MUST NOT QUEUE BEHIND GPU OCCUPANCY. Every job declared a `gpu` flag
            # but nothing ever read it, so the ridge probes, the sweep scorers and the 3-arm
            # abstention comparison — none of which touch the GPU — sat blocked whenever the two
            # GPU slots were busy. They still run via `srun --overlap` (compute-node cores, not
            # the login node), just against their own concurrency cap.
            if not j.get("gpu", True):
                if sum(1 for n in running if not JOB_BY_NAME[n].get("gpu", True)) >= a.cpu_jobs:
                    status["pending"].append(nm)
                    continue
                jid = ids[0]
            else:
                # LEAST-LOADED, not first-with-capacity. `free[0]` always picked the lowest job ID,
                # so one allocation was packed to two heavy jobs while another sat idle.
                free = sorted((n, i) for i, n in slots.items() if n > 0)
                if not free:
                    status["pending"].append(nm)
                    continue
                jid = free[-1][1]                   # most free slots wins
            log = f"{SH}/logs/dispatch_{nm}.log"
            # setsid puts the job in its OWN session and process group, so it survives the
            # dispatcher being restarted. Without it the srun children stayed in tmux's process
            # group and a `tmux kill-session` SIGTERM'd them mid-run — observed as
            # "srun: forcing job termination" truncating a completed analysis. `< /dev/null`
            # detaches stdin so nothing blocks on a closed terminal.
            full = (f"setsid nohup srun --jobid={jid} --overlap bash -c "
                    f"'{ENV}; cd {RV} && {j['cmd']}' > {log} 2>&1 < /dev/null &")
            # truncate the log on (re)launch so `crashed()` reads THIS attempt's output, not a
            # stale traceback from the previous one — otherwise a retry is instantly marked dead
            open(log, "w").close()
            subprocess.run(full, shell=True)
            running[nm] = jid
            attempts[nm] = attempts.get(nm, 0) + 1
            if j.get("gpu", True):
                slots[jid] -= 1
            ledger[nm] = time.time()
            try:
                json.dump(ledger, open(ledger_path, "w"))
            except Exception:                           # noqa: BLE001
                pass
            print(f"[launch] {nm} on {jid} (attempt {attempts[nm]}) -> {log}", flush=True)

        json.dump(status, open(state_path, "w"), indent=2)
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
