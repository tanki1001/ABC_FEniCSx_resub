from multiprocessing import Process
import runpy
from pathlib import Path

def run_script(script_path):
    runpy.run_path(script_path, run_name="__out__")

if __name__ == "__main__":
    base = Path(__file__).parent

    scripts = [
        base / "oldVSnew/geo_spherical/main.py",
        base / "oldVSnew/geo_roundedCubic/main.py",
        base / "oldVSnew/geo_new_broken_cubic/main.py",
    ]

    processes = []
    for script in scripts:
        p = Process(target=run_script, args=(script,))
        p.start()
        processes.append(p)

    # Attendre que tous les scripts soient terminés
    for p in processes:
        p.join()

    print("All scripts finished.")