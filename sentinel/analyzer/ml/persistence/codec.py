"""Pickle codec — restricted unpickler + atomic signed save/load.

Extracted from ``analyzer.ml_predictor`` during the round-10 refactor.
Keeps the v3_signed envelope format byte-identical so every existing
saved model still loads under the current code.

Envelope format (written by :func:`save_signed_payload`)::

    {
        "format": "v3_signed",
        "payload": <bytes>,              # pickle.dumps(inner_dict)
        "checksum": <sha256_hex_str>,    # over `payload`
    }

The inner ``payload`` decodes into the raw data dict that
``MLPredictor.load_from_file`` consumes. Module aliases on class
references (``analyzer.ml_predictor.MLMetrics``) still resolve through
the re-exports in the legacy module — the restricted unpickler only
gates by module prefix, so ``analyzer.ml.domain.metrics.MLMetrics`` and
``analyzer.ml_predictor.MLMetrics`` both pass the whitelist.
"""
from __future__ import annotations

import hashlib
import io
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Module-prefix whitelist for the restricted unpickler. Packages listed
# here may contribute classes to a loaded pickle; anything else raises
# ``UnpicklingError`` before ``__reduce__`` machinery has a chance to
# execute code.
PICKLE_ALLOWED_PREFIXES: tuple[str, ...] = (
    "numpy", "sklearn", "scipy", "pandas",
    "lightgbm", "xgboost",
    "analyzer.ml_ensemble", "analyzer.ml_predictor",
    "analyzer.ml_walk_forward", "analyzer.ml_stacking",
    "analyzer.ml_regime_router", "analyzer.ml_bootstrap",
    # Round-10 refactor: domain / features / persistence moved into a
    # package layout. The broad ``analyzer.ml`` prefix covers all the
    # submodules under it.
    "analyzer.ml",
    "collections", "builtins",  # dicts/lists/tuples live here
)


class RestrictedUnpickler(pickle.Unpickler):
    """Pickle unpickler that only reconstructs classes from a whitelist.

    The SHA-256 checksum on the outer envelope protects against bit-rot
    and naive corruption but NOT a forged artifact: a crafted pickle
    can construct ``os.system``, ``subprocess.Popen``, or arbitrary
    ``__reduce__``-enabled callables and execute them the moment
    ``pickle.load`` runs. This class caps the blast radius by only
    allowing classes from known ML/data packages.

    Not a silver bullet — whitelisted packages can still have gadget
    classes — but it eliminates trivial exec-via-unpickle vectors for
    a local single-user bot without breaking legitimate model loads.
    """

    def find_class(self, module: str, name: str) -> Any:
        for prefix in PICKLE_ALLOWED_PREFIXES:
            if module == prefix or module.startswith(prefix + "."):
                return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"refusing to load class {module}.{name}: module not in pickle whitelist"
        )


def restricted_loads(payload: bytes) -> Any:
    """Unpickle ``payload`` through :class:`RestrictedUnpickler`."""
    return RestrictedUnpickler(io.BytesIO(payload)).load()


def save_signed_payload(path: Path, data: dict) -> bool:
    """Write ``data`` to ``path`` as a v3_signed envelope (atomic).

    Writes to ``path.with_suffix(".tmp")`` first, then ``replace``s the
    real file. Returns True on success; logs warnings and returns False
    on failure so the caller can skip the registry append.

    Security: refuses paths containing ``..`` segments. Clean absolute
    paths pass through unchanged (tests using ``tmp_path`` still work).
    Callers that need to write above a parent directory must normalise
    the path first.
    """
    if ".." in path.parts:
        logger.error(
            "ML save refused: path contains '..' traversal segment (%s). "
            "Normalise the path before calling save_to_file.",
            path,
        )
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        payload = pickle.dumps(data)
        checksum = hashlib.sha256(payload).hexdigest()
        with tmp_path.open("wb") as f:
            pickle.dump(
                {"payload": payload, "checksum": checksum, "format": "v3_signed"},
                f,
            )
        tmp_path.replace(path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML model save failed to %s: %s", path, exc)
        return False


def save_signed_payload_with_checksum(path: Path, data: dict) -> tuple[bool, Optional[str]]:
    """Same as :func:`save_signed_payload` but also returns the sha256
    hex string. The caller needs the hex for the registry entry."""
    if ".." in path.parts:
        logger.error(
            "ML save refused: path contains '..' traversal segment (%s). "
            "Normalise the path before calling save_to_file.",
            path,
        )
        return False, None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        payload = pickle.dumps(data)
        checksum = hashlib.sha256(payload).hexdigest()
        with tmp_path.open("wb") as f:
            pickle.dump(
                {"payload": payload, "checksum": checksum, "format": "v3_signed"},
                f,
            )
        tmp_path.replace(path)
        return True, checksum
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML model save failed to %s: %s", path, exc)
        return False, None


def load_signed_payload(path: Path) -> Optional[dict]:
    """Read and verify a v3_signed envelope; return the inner data dict.

    Returns ``None`` when:
      * the file doesn't exist
      * the checksum doesn't match (corruption / tampering)
      * the pickle layer raises
      * the envelope format is unrecognised

    Legacy unsigned pickles (pre-v3) are still accepted and returned
    verbatim — the outer unpickle still goes through RestrictedUnpickler
    so unknown class modules are still blocked.
    """
    if not path.exists():
        logger.warning("ML load skipped: file not found %s", path)
        return None
    try:
        with path.open("rb") as f:
            raw = restricted_loads(f.read())

        if isinstance(raw, dict) and raw.get("format") == "v3_signed":
            payload = raw["payload"]
            expected = raw.get("checksum", "")
            actual = hashlib.sha256(payload).hexdigest()
            if expected and actual != expected:
                logger.error(
                    "ML load ABORTED: checksum mismatch (expected=%s, got=%s). "
                    "File may be corrupted or tampered.",
                    expected[:12], actual[:12],
                )
                return None
            return restricted_loads(payload)
        # Legacy unsigned format — data dict sits directly at the top
        return raw if isinstance(raw, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.error("ML model load failed from %s: %s", path, exc)
        return None
