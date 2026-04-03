"""Subprocess-isolated GPU rendering for dataset generation.

Each render runs in a fresh subprocess to guarantee complete GPU memory
cleanup when the process exits. This eliminates OptiX/CUDA memory leaks
that accumulate over thousands of renders in SAPIEN's ray-tracing backend.
"""

import multiprocessing as mp
import time
from multiprocessing.connection import wait as _mp_wait
from typing import Any, Dict, Generator, List, Optional, Tuple

_mp_ctx = mp.get_context("spawn")

# Fixed offsets so train/test seeds never overlap and each split is
# independently reproducible regardless of whether the other was generated.
SPLIT_SEED_OFFSETS = {"train": 0, "test": 1_000_000}


def _worker(
    conn: mp.connection.Connection,
    env_id: str,
    env_module: str,
    attempts: List[Dict[str, Any]],
):
    """Subprocess render worker. Creates env, renders, sends result via pipe.

    Args:
        conn: Write end of a ``multiprocessing.Pipe``.
        env_id: Gymnasium environment ID (e.g. ``"KitchenRenderEnv-v1"``).
        env_module: Python module to import so ``@register_env`` fires.
        attempts: List of dicts, each with keys ``env_kwargs``, ``seed``,
            ``num_steps``, ``min_segments``.
    """
    import importlib

    importlib.import_module(env_module)

    import gymnasium as gym

    for idx, attempt in enumerate(attempts):
        gym_env = None
        try:
            gym_env = gym.make(env_id, **attempt["env_kwargs"])
            obs, _ = gym_env.reset(seed=attempt["seed"])
            for _ in range(attempt["num_steps"]):
                obs, _, _, _, _ = gym_env.step(None)

            render_env = gym_env.unwrapped
            if render_env.is_valid_render(obs, attempt["min_segments"]):
                data = render_env.build_render_data(obs)
                # Convert torch tensors to numpy for pickling across processes
                data = {
                    k: v.numpy() if hasattr(v, "numpy") else v for k, v in data.items()
                }
                gym_env.close()
                gym_env = None
                conn.send({"data": data, "attempt_idx": idx, "error": None})
                return

            print(f"Subprocess: Low quality render (attempt {idx + 1}/{len(attempts)})")
        except Exception as e:
            print(f"Subprocess: Render error (attempt {idx + 1}): {e}")
        finally:
            if gym_env is not None:
                try:
                    gym_env.close()
                except Exception:
                    pass

    conn.send({"data": None, "attempt_idx": None, "error": "All attempts failed"})


def render_in_subprocess(
    env_id: str,
    env_module: str,
    attempts: List[Dict[str, Any]],
    timeout: float = 300,
) -> Tuple[Optional[Dict], Optional[int]]:
    """Run render attempts in an isolated subprocess.

    The child process is created with the ``"spawn"`` start method so it
    gets a completely fresh GPU context.  When the process exits all GPU
    resources (OptiX acceleration structures, CUDA allocations, etc.) are
    released by the OS.

    Args:
        env_id: Gymnasium environment ID.
        env_module: Module path to import for env registration.
        attempts: Attempt configs forwarded to :func:`_worker`.
        timeout: Seconds to wait before killing the subprocess.

    Returns:
        ``(render_data, attempt_index)`` on success, ``(None, None)`` on
        failure.
    """
    parent_conn, child_conn = _mp_ctx.Pipe(duplex=False)
    proc = _mp_ctx.Process(
        target=_worker, args=(child_conn, env_id, env_module, attempts)
    )
    proc.start()
    child_conn.close()  # parent only reads

    try:
        if parent_conn.poll(timeout):
            result = parent_conn.recv()
        else:
            print(f"Warning: Render subprocess timed out after {timeout}s")
            proc.kill()
            proc.join(timeout=10)
            return None, None
    except EOFError:
        print(f"Warning: Render subprocess crashed (exit code: {proc.exitcode})")
        return None, None
    finally:
        parent_conn.close()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)

    if result.get("error"):
        print(f"Render failed: {result['error']}")
        return None, None

    return result["data"], result["attempt_idx"]


def _worker_batch(
    conn: mp.connection.Connection,
    env_id: str,
    env_module: str,
    render_jobs: List[Dict[str, Any]],
):
    """Subprocess worker that renders a batch of jobs.

    Pays the Python / SAPIEN / torch import cost only once for the whole
    batch.  A single gym env is created for the first job and then reused
    across all subsequent jobs: ``update_render_params`` swaps camera /
    lighting / asset params before each ``reset()``, avoiding repeated
    SAPIEN physics-engine initialisation overhead.

    If any attempt raises an exception the env is considered broken and is
    closed; it will be recreated from scratch for the next job.

    Args:
        conn: Write end of a ``multiprocessing.Pipe``.
        env_id: Gymnasium environment ID.
        env_module: Python module to import so ``@register_env`` fires.
        render_jobs: List of dicts, each with keys ``job_id`` and ``attempts``.
    """
    import importlib

    importlib.import_module(env_module)

    import gymnasium as gym

    gym_env = None
    render_env = None
    current_scene_key = object()  # sentinel: guaranteed != any real scene_key

    def _close_env():
        nonlocal gym_env, render_env
        if gym_env is not None:
            try:
                gym_env.close()
            except Exception:
                pass
            gym_env = None
            render_env = None

    results = []
    for job in render_jobs:
        # Rebuild the scene (gym.make) only when the scene group changes.
        # Jobs with the same scene_key share assets/kitchen-layout; only camera
        # and lighting differ, which update_render_params handles via
        # _update_sensor_pose without a full reconfigure.
        scene_key = job.get("scene_key")
        if scene_key != current_scene_key:
            _close_env()
            current_scene_key = scene_key

        job_result = {"job_id": job["job_id"], "data": None, "attempt_idx": None}
        for idx, attempt in enumerate(job["attempts"]):
            try:
                if gym_env is None:
                    gym_env = gym.make(env_id, **attempt["env_kwargs"])
                    render_env = gym_env.unwrapped
                else:
                    render_env.update_render_params(**attempt["env_kwargs"])

                obs, _ = gym_env.reset(seed=attempt["seed"])
                for _ in range(attempt["num_steps"]):
                    obs, _, _, _, _ = gym_env.step(None)

                if render_env.is_valid_render(obs, attempt["min_segments"]):
                    data = render_env.build_render_data(obs)
                    data = {
                        k: v.numpy() if hasattr(v, "numpy") else v
                        for k, v in data.items()
                    }
                    job_result = {
                        "job_id": job["job_id"],
                        "data": data,
                        "attempt_idx": idx,
                    }
                    break

                print(
                    f"Subprocess: Low quality render "
                    f"(job {job['job_id']}, attempt {idx + 1}/{len(job['attempts'])})"
                )
            except Exception as e:
                print(
                    f"Subprocess: Render error "
                    f"(job {job['job_id']}, attempt {idx + 1}): {e}"
                )
                _close_env()  # Env state is unknown; recreate for next attempt
                current_scene_key = object()  # force rebuild on next job too
        conn.send(job_result)  # stream each result immediately

    _close_env()
    conn.send(None)  # sentinel: batch complete


def render_batch_parallel(
    env_id: str,
    env_module: str,
    job_batches: List[List[Dict[str, Any]]],
    timeout_per_job: float = 30,
) -> Generator[Dict[str, Any], None, None]:
    """Run batches of render jobs across parallel subprocesses.

    Each batch runs in its own ``"spawn"`` subprocess so GPU memory is
    fully released when it exits.  Results are **yielded one at a time**
    as the worker sends them, so the caller's progress bar updates after
    every individual render instead of waiting for the whole batch.

    Args:
        env_id: Gymnasium environment ID.
        env_module: Module path to import for env registration.
        job_batches: ``[batch_0, batch_1, ...]`` where each batch is a list
            of job dicts with keys ``job_id`` and ``attempts``.
        timeout_per_job: Seconds before a silent subprocess is considered
            stalled and killed.

    Yields:
        Result dicts with keys ``job_id``, ``data``, ``attempt_idx`` —
        one per job, in completion order.
    """
    procs: List[mp.Process] = []
    # active maps parent_conn -> (proc, last_message_time)
    active: Dict[mp.connection.Connection, tuple] = {}

    for batch in job_batches:
        parent_conn, child_conn = _mp_ctx.Pipe(duplex=False)
        proc = _mp_ctx.Process(
            target=_worker_batch,
            args=(child_conn, env_id, env_module, batch),
        )
        proc.start()
        child_conn.close()
        procs.append(proc)
        active[parent_conn] = (proc, time.monotonic())

    try:
        while active:
            ready = _mp_wait(list(active.keys()), timeout=1.0)
            now = time.monotonic()

            for conn in ready:
                proc, _ = active[conn]
                try:
                    msg = conn.recv()
                except EOFError:
                    print(f"Warning: Subprocess crashed (exit code: {proc.exitcode})")
                    del active[conn]
                    conn.close()
                    continue

                if msg is None:  # sentinel — this subprocess finished cleanly
                    del active[conn]
                    conn.close()
                else:
                    active[conn] = (proc, now)  # reset stall timer
                    yield msg

            # Kill subprocesses that have gone silent for too long
            for conn, (proc, last_time) in list(active.items()):
                if now - last_time > timeout_per_job:
                    print(
                        f"Warning: Subprocess stalled for >{timeout_per_job:.0f}s, killing"
                    )
                    proc.kill()
                    del active[conn]
                    conn.close()
    finally:
        # Ensure all subprocesses are cleaned up even on early generator exit
        for conn in list(active.keys()):
            conn.close()
        for proc in procs:
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=10)
