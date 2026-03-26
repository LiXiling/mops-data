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
