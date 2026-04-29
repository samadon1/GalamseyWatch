"""Minimal Modal smoke test.

CPU-only, no external deps beyond debian_slim. Verifies that the Modal CLI
auth, container build, function submission, log streaming, and return value
marshaling all work end to end. Costs fractions of a cent and runs in under
a minute after the first container build.

Run with:
    uv run modal run scripts/modal_smoke_test.py
"""

import modal

app = modal.App("galamsey-smoke-test")

image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, timeout=120)
def verify() -> dict[str, str]:
    import platform
    import sys

    print("=" * 60)
    print("GalamseyWatch, Modal smoke test")
    print("=" * 60)
    print(f"Python:     {sys.version.split()[0]}")
    print(f"Platform:   {platform.platform()}")
    print(f"Machine:    {platform.machine()}")
    print("=" * 60)

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "status": "ok",
    }


@app.local_entrypoint()
def main() -> None:
    print("Submitting smoke test to Modal...")
    result = verify.remote()
    print(f"Remote returned: {result}")
    print("Smoke test PASSED.")
