from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

from .io_utils import append_text_line, ensure_dir


def run_with_auto_restart(
    target: Callable[[], None],
    restart_wait_seconds: float = 5.0,
    max_run_restarts: int | None = None,
    log_path: str | Path | None = None,
) -> None:
    attempt = 0
    log_file = Path(log_path) if log_path else None
    if log_file is not None:

        ensure_dir(log_file.parent)

    while True:
        try:
            attempt += 1
            start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"[{start_ts}] Starting attempt {attempt}"
            print(msg)
            if log_file is not None:
                append_text_line(log_file, msg)


            target()

            finish_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            done_msg = f"[{finish_ts}] Job completed successfully"
            print(done_msg)
            if log_file is not None:
                append_text_line(log_file, done_msg)
            return
        except KeyboardInterrupt:

            stop_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stop_msg = f"[{stop_ts}] Manual interrupt detected; stopping auto-restart."
            print(stop_msg)
            if log_file is not None:
                append_text_line(log_file, stop_msg)
            raise
        except Exception as exc:
            err_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tb = traceback.format_exc()
            err_msg = (
                f"[{err_ts}] Attempt {attempt} failed: {exc}\n"
                f"{tb.rstrip()}"
            )
            print(err_msg)
            if log_file is not None:
                append_text_line(log_file, err_msg)


            if max_run_restarts is not None and attempt > max_run_restarts:
                raise

            wait_msg = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Restarting after {restart_wait_seconds} seconds..."
            )
            print(wait_msg)
            if log_file is not None:
                append_text_line(log_file, wait_msg)


            time.sleep(max(0.0, float(restart_wait_seconds)))
