"""The one measurement the compile-time tools share: time a `mojo build`."""

import os
import shutil
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compile_once(body, work, name="probe", flags=(), cache=None, timeout=None,
                 mojo=("mojo",)):
    """Write `body` to `work/<name>.mojo` and time `mojo build` of it (from
    ROOT, output `work/<name>`). With `cache`, the build gets that private
    MODULAR_CACHE_DIR, cleared first: a solo-cold compile. Returns
    (seconds, CompletedProcess), or (None, None) past `timeout`."""
    src = os.path.join(work, name + ".mojo")
    with open(src, "w") as f:
        f.write(body)
    env = None
    if cache:
        shutil.rmtree(cache, ignore_errors=True)
        os.makedirs(cache)
        env = dict(os.environ, MODULAR_CACHE_DIR=cache)
    t0 = time.monotonic()
    try:
        r = subprocess.run(
            [*mojo, "build", *flags, src, "-o", os.path.join(work, name)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, None
    return time.monotonic() - t0, r
