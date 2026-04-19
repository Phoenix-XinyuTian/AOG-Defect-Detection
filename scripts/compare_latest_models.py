from pathlib import Path
import runpy
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "src" / "reporting" / "compare_models.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
