"""Install (and verify) everything the bioreactor GUI needs.

Run with the SAME interpreter you launch App.py with:

    py -3 install_dependencies.py

Why this is not just a loop of `pip install <pkg>`
--------------------------------------------------
It used to install each package in its own pip call. pip resolves each call
independently, so a later install is free to move a dependency an earlier one
was already compiled against. The usual casualty is numpy: pandas ships binary
extensions built against a specific numpy C ABI, and if numpy is swapped out
from under it you get, at import time,

    ValueError: numpy.dtype size changed, may indicate binary incompatibility.
    Expected 96 from C header, got 88 from PyObject

which looks like a code bug but is purely an install-order artefact. Installing
the whole core set in ONE pip call lets the resolver pick a mutually consistent
set. The verify step below then actually imports them, because a clean pip exit
code does not prove the binaries agree.
"""
import subprocess
import sys

# Resolved together in a single pip call — order here is irrelevant, which is
# the point. numpy is listed first only for readability.
CORE = [
    "numpy",
    "pandas",
    "matplotlib",
    "PyQt6",
    "pyserial",      # imported as `serial` — required for serial IP provisioning
]

# Microscope tab only ([Microscope] Enabled in config/config.ini). The GUI runs
# fine without these: the tab falls back to a synthetic demo image, so a failure
# here is reported but never fatal.
OPTIONAL = [
    "pycromanager",  # Micro-Manager bridge
    "tifffile",      # writing acquisition stacks
    "pydobot",       # Dobot Magician arm (Sampling tab). Absent -> Dobot.py runs simulated.
]

# Imported name -> pip name, for the post-install verification.
IMPORT_NAMES = {
    "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
    "PyQt6.QtWidgets": "PyQt6", "serial": "pyserial",
}


def pip(args, label):
    print(f"  $ pip {' '.join(args)}")
    r = subprocess.run([sys.executable, "-m", "pip", *args],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  [OK] {label}")
        return True
    print(f"  [FAILED] {label}")
    print((r.stderr or r.stdout).strip()[-2000:])
    return False


def verify():
    """Import each core package in a FRESH interpreter.

    A subprocess is essential: numpy/pandas C-extension mismatches only surface
    on first import, and this process may already have them loaded.
    """
    code = "import " + ", ".join(IMPORT_NAMES) + "; print('imports-ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode == 0 and "imports-ok" in r.stdout:
        return True, ""
    return False, (r.stderr or r.stdout).strip()


def main():
    print(f"Using Python: {sys.executable}")
    print(f"Version:      {sys.version.split()[0]}\n")

    print("Core dependencies (single resolver pass):")
    pip(["install", "--upgrade", *CORE], "core")

    print("\nOptional (microscope tab):")
    for pkg in OPTIONAL:
        pip(["install", pkg], pkg)

    print("\nVerifying imports in a fresh interpreter:")
    ok, err = verify()

    if not ok:
        print("  [MISMATCH] core imports failed:\n")
        print("    " + err.replace("\n", "\n    ")[:1500])
        # Almost always a stale binary built against a different numpy. Force a
        # clean rebuild of the pair, ignoring wheels cached from the bad combo.
        print("\n  Repairing: force-reinstalling numpy + pandas together...")
        pip(["install", "--force-reinstall", "--no-cache-dir", "numpy", "pandas"],
            "numpy + pandas")
        ok, err = verify()

    print()
    if ok:
        print("All core imports OK. Launch with:  py -3 App.py")
        return 0

    print("STILL FAILING after repair:\n")
    print("    " + err.replace("\n", "\n    ")[:1500])
    print("\nMost likely another Python has these packages and is shadowing this")
    print("one. Check that VS Code's selected interpreter is:")
    print(f"    {sys.executable}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
