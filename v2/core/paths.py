from pathlib import Path
from datetime import datetime
ROOT = Path(__file__).resolve().parents[1]

def create_run_dir(base="datasets"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    run_dir = ROOT / base / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir