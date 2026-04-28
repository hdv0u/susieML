from core.engine.train_engine import TrainEngine
from core.paths import create_run_dir
import json, shutil

def run_train(cfg_ctrl, dataset_path, log_fn, progress_fn, stop_ctrl):
    run_dir = create_run_dir()
    
    model_path = run_dir / "model.pt"
    dataset_out = run_dir / "dataset.json"
    config_out = run_dir / "config.json"
    
    shutil.copy(dataset_path, dataset_out)
    
    with open(config_out, "w") as f:
        json.dump(cfg_ctrl.raw(), f, indent=4)
        
    engine = TrainEngine(cfg_ctrl, log_fn=log_fn)
    
    engine.train(
        dataset_path=str(dataset_out),
        save_path=str(model_path),
        stop_ctrl=stop_ctrl,
        progress_fn=progress_fn
    )
    return model_path