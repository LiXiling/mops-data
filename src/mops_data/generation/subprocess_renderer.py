"""Subprocess-isolated GPU rendering for dataset generation.

Each render runs in a fresh subprocess to guarantee complete GPU memory
cleanup when the process exits. This eliminates OptiX/CUDA memory leaks
that accumulate over thousands of renders in SAPIEN's ray-tracing backend.
"""

import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

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
    batch, then creates and tears down a gym env per job.

    Args:
        conn: Write end of a ``multiprocessing.Pipe``.
        env_id: Gymnasium environment ID.
        env_module: Python module to import so ``@register_env`` fires.
        render_jobs: List of dicts, each with keys ``job_id`` and ``attempts``.
    """
    import importlib

    importlib.import_module(env_module)

    import gymnasium as gym

    results = []
    for job in render_jobs:
        job_result = {"job_id": job["job_id"], "data": None, "attempt_idx": None}
        for idx, attempt in enumerate(job["attempts"]):
            gym_env = None
            try:
                gym_env = gym.make(env_id, **attempt["env_kwargs"])
                obs, _ = gym_env.reset(seed=attempt["seed"])
                for _ in range(attempt["num_steps"]):
                    obs, _, _, _, _ = gym_env.step(None)

                render_env = gym_env.unwrapped
                if render_env.is_valid_render(obs, attempt["min_segments"]):
                    data = render_env.build_render_data(obs)
                    data = {
                        k: v.numpy() if hasattr(v, "numpy") else v
                        for k, v in data.items()
                    }
                    gym_env.close()
                    gym_env = None
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
            finally:
                if gym_env is not None:
                    try:
                        gym_env.close()
                    except Exception:
                        pass
        results.append(job_result)

    conn.send(results)


def render_batch_parallel(
    env_id: str,
    env_module: str,
    job_batches: List[List[Dict[str, Any]]],
    timeout_per_job: float = 30,
) -> List[Dict[str, Any]]:
    """Run batches of render jobs across parallel subprocesses.

    Each batch runs in its own ``"spawn"`` subprocess so GPU memory is
    fully released when it exits.  Multiple batches run concurrently.

    Args:
        env_id: Gymnasium environment ID.
        env_module: Module path to import for env registration.
        job_batches: ``[batch_0, batch_1, ...]`` where each batch is a list
            of job dicts with keys ``job_id`` and ``attempts``.
        timeout_per_job: Seconds budgeted per job; total timeout for a
            subprocess is ``timeout_per_job * len(batch)``.

    Returns:
        Flat list of result dicts (ordered by batch, then by job within
        each batch) with keys ``job_id``, ``data``, ``attempt_idx``.
    """
    pipes = []
    procs = []

    for batch in job_batches:
        parent_conn, child_conn = _mp_ctx.Pipe(duplex=False)
        proc = _mp_ctx.Process(
            target=_worker_batch,
            args=(child_conn, env_id, env_module, batch),
        )
        proc.start()
        child_conn.close()
        pipes.append(parent_conn)
        procs.append((proc, len(batch)))

    all_results = []
    for parent_conn, (proc, batch_len) in zip(pipes, procs):
        timeout = timeout_per_job * batch_len
        try:
            if parent_conn.poll(timeout):
                results = parent_conn.recv()
                all_results.extend(results)
            else:
                print(f"Warning: Batch subprocess timed out after {timeout:.0f}s")
                proc.kill()
                proc.join(timeout=10)
        except EOFError:
            print(
                f"Warning: Batch subprocess crashed (exit code: {proc.exitcode})"
            )
        finally:
            parent_conn.close()
            proc.join(timeout=30)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=10)

    return all_results
