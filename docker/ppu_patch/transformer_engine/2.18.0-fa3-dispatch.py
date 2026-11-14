#!/usr/bin/env python3
"""PPU post-install edit: make TE actually dispatch to its FlashAttention-3
kernels.

Based on:
  PPU source   https://github.com/flytiger-eco/TransformerEngine-for-sail
               (tag 2.18+v0.1.0, commit efa9bf4b -- the wheel installed here)
  Relax patch  docker/patch/transformer_engine/2.18.0-cp-padding-no-host-sync.patch
               -- applied on the CUDA line only; this wheel still carries TE's
               own `is_graph_capturing()` branch for that zero-fill, which is
               the sync-free path under capture, so it is not applied here and
               this script is the whole PPU delta.

TE 2.18-SAIL selects the backend in ``dot_product_attention/utils.py`` by assigning
``flash_attention_backend = FlashAttentionUtils.fa3_version`` whenever FA3 wins the
filter, but dispatches in ``backends.py`` on ``flash_attention_backend.major == 3``.
The PPU ``flash-attn-3`` wheel is versioned on the FA release line (2.8.2, major=2),
so that test is always false here and every forward silently lands on the FA2
extension -- while the log line still reads ``Selected backend = FlashAttention (2.8.2)``.

Also accept "the selected backend is exactly the installed FA3 package version":
the assignment above only ever produces that value when FA3 was chosen, and the FA2
version (2.7.4.post1) can never equal it, so the FA2 path keeps judging false.
``major == 3`` is kept for the upstream hopper version line (3.0.0bX).

Idempotent, and refuses to run if the anchor is missing or ambiguous.
"""

import glob
import hashlib
import importlib.util
import pathlib
import sys


REL = "pytorch/attention/dot_product_attention/backends.py"
MARKER = "/etc/relax-ppu-te-fa3.txt"

# `FlashAttentionUtils` is imported into backends.py as `fa_utils` (backends.py:72);
# that module has no bare `FlashAttentionUtils` name, so the added clause must use the
# alias or the forward would die with NameError at the first call.
OLD = (
    "        use_flash_attn_3 = (\n"
    "            flash_attention_backend is not None and flash_attention_backend.major == 3\n"
    "        )\n"
)
NEW = (
    "        use_flash_attn_3 = (\n"
    "            flash_attention_backend is not None\n"
    "            and (\n"
    "                flash_attention_backend.major == 3\n"
    "                # PPU: flash-attn-3 is packaged as 2.8.x, so match the selected\n"
    "                # backend against the installed FA3 version as well.\n"
    "                or (fa_utils.v3_is_installed\n"
    "                    and flash_attention_backend == fa_utils.fa3_version)\n"
    "            )\n"
    "        )\n"
)


def target() -> pathlib.Path:
    spec = importlib.util.find_spec("transformer_engine")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("te_fa3_dispatch: transformer_engine not importable")
    root = pathlib.Path(next(iter(spec.submodule_search_locations)))
    path = root / REL
    if not path.is_file():
        raise SystemExit(f"te_fa3_dispatch: no such file {path}")
    return path


def main() -> int:
    path = target()
    src = path.read_text()

    if NEW in src:
        state = "already applied"
    else:
        count = src.count(OLD)
        if count != 1:
            raise SystemExit(f"te_fa3_dispatch: anchor hit {count} times in {path}, expected exactly 1")
        patched = src.replace(OLD, NEW)
        compile(patched, str(path), "exec")  # never ship a broken edit
        path.write_text(patched)
        state = "patched"

    # Stale bytecode would keep the old dispatch alive until the next rebuild.
    for pyc in glob.glob(str(path.parent / "__pycache__" / "backends.*.pyc")):
        pathlib.Path(pyc).unlink()

    final = path.read_text()
    assert NEW in final and OLD not in final, "te_fa3_dispatch: post-write verification failed"
    digest = hashlib.sha256(final.encode()).hexdigest()
    try:
        pathlib.Path(MARKER).write_text(
            f"file={path}\nstate={state}\nsha256={digest}\n"
            "clause=fa_utils.v3_is_installed and flash_attention_backend == fa_utils.fa3_version\n"
        )
    except OSError as exc:  # read-only root: the edit itself still stands
        print(f"te_fa3_dispatch: warning: marker not written ({exc})")

    print(f"te_fa3_dispatch: {state} OK  {path}  sha256={digest[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
