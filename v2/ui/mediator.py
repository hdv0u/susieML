from core.engine.train_engine import TrainEngine
from core.engine.infer_engine import InferEngine
from core.paths import create_run_dir
from ui.mediator_bus import bus
from ui.workers.train_worker import TrainWorker
from ui.workers.infer_worker import InferWorker
from core.detection.screen_detector import ScreenDetector
from core.control.run_controller import RunController
from core.control.config_controller import ConfigController
class Mediator:
    def __init__(self, cfg, ui=None):
        self.ui = ui
        self.log = self.ui.log1.append
        self.stop_ctrl = RunController()
        self.cfg = cfg
        self.config = ConfigController(self.cfg)
        
        self.trainer = TrainEngine(cfg, log_fn=self.ui.log1.append)
        self.worker = None
        
        bus.train_requested.connect(self.handle_train)
        bus.run_requested.connect(self.handle_run)
        bus.stop_requested.connect(self.handle_stop)
        
    def handle_train(self, payload):
        dataset_path = payload["dataset_path"]
        config = payload["config"]
        
        self.config.update(config)
        
        run_dir = create_run_dir()
        model_path = run_dir / "model.pt"
        dataset_out_path = run_dir / "dataset.json"
        config_path = run_dir / "config.json"
        
        import shutil
        shutil.copy(dataset_path, dataset_out_path)
        
        import json
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        
        self.trainer = TrainEngine(self.config, log_fn=self.log)
          
        self.worker = TrainWorker(
            trainer=self.trainer,
            payload={
                "dataset_path": str(dataset_out_path),
                "save_path": str(model_path),
                "epochs": config["training"]["epochs"]
            },
            stop_ctrl=self.stop_ctrl
        )

        self.worker.log.connect(self.ui.log1.append)
        self.worker.progress.connect(self.ui.progress1.setValue)
        
        self.worker.start()
            
    def handle_run(self, payload):
        model_path = payload["model_path"]
        config = payload["config"]
        
        self.config.update(config)
        if self.worker:
            self.stop_ctrl.stop()
            self.worker.wait()
            self.worker = None
        
        self.stop_ctrl.reset()
        
        engine = InferEngine(self.config, model_path, log_fn=self.ui.log2.append)
        
        self.detector = ScreenDetector(
            engine=engine,
            cfg=self.config
        )
        
        if hasattr(self, "worker") and self.worker:
            self.worker.stop()
            
        self.worker = InferWorker(self.detector, self.stop_ctrl)
        
        self.worker.frame_ready.connect(self.ui.update_frame)
        self.worker.log.connect(self.ui.log2.append)
        self.worker.start()
        
        self.ui.log2.append(f"Model loaded {model_path}")
            
    def handle_stop(self):
        if self.stop_ctrl:
            self.stop_ctrl.stop()
        
        if self.worker:
            self.worker.wait()
        
        if self.ui:
            self.ui.log1.append("Stop requested")
            self.ui.log2.append("Stop requested")