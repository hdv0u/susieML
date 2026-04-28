from core.usecases.train import run_train
from core.usecases.infer import build_detector
from ui.mediator_bus import bus
from ui.workers.train_worker import TrainWorker
from ui.workers.infer_worker import InferWorker
from core.control.run_controller import RunController
from core.control.config_controller import ConfigController
class Mediator:
    def __init__(self, cfg, ui=None):
        self.ui = ui
        self.log = self.ui.log1.append
        self.stop_ctrl = RunController()
        self.config = ConfigController(cfg)
        
        self.worker = None
        
        bus.train_requested.connect(self.handle_train)
        bus.run_requested.connect(self.handle_run)
        bus.stop_requested.connect(self.handle_stop)
        
    def handle_train(self, payload):
        self.config.update(payload["config"])
        
        def logger(msg):
            self.ui.log1.append(str(msg))
        
        self.worker = TrainWorker(
            trainer=None,
            payload={
                "dataset_path": payload["dataset_path"],
                "runner": lambda progress_fn: run_train(
                    cfg_ctrl=self.config,
                    dataset_path=payload["dataset_path"],
                    log_fn=logger,
                    progress_fn=progress_fn,
                    stop_ctrl=self.stop_ctrl
                )
            },
            stop_ctrl=self.stop_ctrl
        )

        self.worker.log.connect(self.ui.log1.append)
        self.worker.progress.connect(self.ui.progress1.setValue)
        
        self.worker.start()
            
    def handle_run(self, payload):
        self.config.update(payload["config"])
        
        if self.worker:
            self.stop_ctrl.stop()
            self.worker.wait()
            self.worker = None
        
        self.stop_ctrl.reset()
        
        self.detector = build_detector(
            cfg_ctrl=self.config,
            model_path=payload["model_path"],
            log_fn=self.ui.log2.append
        )
        
        if hasattr(self, "worker") and self.worker:
            self.worker.stop()
            
        self.worker = InferWorker(self.detector, self.stop_ctrl)
        
        self.worker.frame_ready.connect(self.ui.update_frame)
        self.worker.log.connect(self.ui.log2.append)
        self.worker.start()
        
        self.ui.log2.append(f"Model loaded {payload['model_path']}")
            
    def handle_stop(self):
        self.stop_ctrl.stop()
        
        if self.worker:
            self.worker.wait()
        
        if self.ui:
            self.ui.log1.append("Stop requested")
            self.ui.log2.append("Stop requested")