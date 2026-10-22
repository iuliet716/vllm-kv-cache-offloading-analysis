import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment", choices=["cancel", "completion", "all"], default="all"
    )
    args = parser.parse_args()

    vllm_dir = args.vllm_dir.resolve()
    test_dir = vllm_dir / "tests/v1/kv_connector/unit/offloading_connector"
    if not (test_dir / "conftest.py").is_file():
        parser.error("--vllm-dir must point to the vLLM source checkout")

    experiments = Path(__file__).resolve().parents[0] / "experiments"
    files = {
        "cancel": "test_connector_cancel.py",
        "completion": "test_completion_delivery.py",
    }
    selected = files if args.experiment == "all" else [args.experiment]

    # Use upstream fixtures without overwriting the user's existing tests.
    with tempfile.TemporaryDirectory(prefix="analysis_", dir=test_dir) as temporary:
        for name in selected:
            shutil.copy(experiments / files[name], temporary)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "-s", "-rs", temporary],
            cwd=vllm_dir,
        )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
