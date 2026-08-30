#!/usr/bin/env python3
"""Install the TrajDeleter hooks into d3rlpy 1.0.0."""

from importlib.metadata import version
from pathlib import Path
import hashlib
import shutil


PATCH_FILES = {
    "base.py": "d3rlpy_base.py",
    "torch_utility.py": "d3rlpy_torch_utility.py",
    "algos/td3_plus_bc.py": "d3rlpy_td3_plus_bc.py",
    "algos/bcq.py": "d3rlpy_bcq.py",
    "algos/iql.py": "d3rlpy_iql.py",
    "algos/torch/ddpg_impl.py": "d3rlpy_ddpg_impl.py",
}

OFFICIAL_SHA256 = {
    "base.py": "324abb983b2b886a9ce21718ff85de543f76d237a1096c2997725c443f0b05ec",
    "torch_utility.py": "7d0f370958d043356401a0240c4aabe2a2c6ed4f13d8ad345fee9a33c39493dc",
    "algos/td3_plus_bc.py": "dbe7e641ec9be799f8585a8640bbfc59cfa40ba0794f406d18d885fcf009478b",
    "algos/bcq.py": "de56766698e81566bbfb174e71f5aa5cbeca0f8debca82161f0c8fc00593ff0a",
    "algos/iql.py": "9160b61899e7a523ee91cf8167b74e92c3454ffe94686f86f6a85c9fd5e63502",
    "algos/torch/ddpg_impl.py": "93f84f6866b19b1029f351027021ab9f21af1264a6758492ad46a2f953263401",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_patch(package_dir: Path) -> None:
    package_dir = package_dir.resolve()
    patch_dir = Path(__file__).resolve().parent
    pairs = [
        (target_name, patch_dir / source_name, package_dir / target_name)
        for target_name, source_name in PATCH_FILES.items()
    ]

    missing = [
        str(path)
        for _, source, target in pairs
        for path in (source, target)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing patch input: " + ", ".join(missing))

    states = []
    for target_name, source, target in pairs:
        source_hash = sha256(source)
        target_hash = sha256(target)
        backup = Path(str(target) + ".bak")
        if target_hash == source_hash:
            state = "patched"
        elif target_hash == OFFICIAL_SHA256[target_name]:
            state = "official"
        else:
            raise RuntimeError(f"Refusing to overwrite modified d3rlpy file: {target}")
        if backup.exists() and sha256(backup) != OFFICIAL_SHA256[target_name]:
            raise RuntimeError(f"Unexpected backup contents: {backup}")
        states.append((state, source, target, backup))

    for state, source, target, backup in states:
        if state == "patched":
            print(f"Already patched: {target}")
            continue
        if not backup.exists():
            shutil.copy2(target, backup)
        shutil.copy2(source, target)
        print(f"Patched {target} (backup: {backup})")


def main() -> None:
    if version("d3rlpy") != "1.0.0":
        raise RuntimeError("This patch requires d3rlpy 1.0.0.")

    import d3rlpy

    install_patch(Path(d3rlpy.__file__).resolve().parent)


if __name__ == "__main__":
    main()
