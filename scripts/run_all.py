import os
import sys
import subprocess


def run_script(script_path):
    print(f"\n{'=' * 50}\nRunning {script_path}...\n{'=' * 50}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ret = subprocess.run([sys.executable, script_path], env=env)
    if ret.returncode != 0:
        print(f"Error running {script_path}. Exiting.")
        sys.exit(ret.returncode)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    scripts = ["scripts/download_data.py", "scripts/train.py", "scripts/evaluate.py"]

    for script in scripts:
        run_script(os.path.join(base_dir, script))

    print("\nAll scripts executed successfully!")
