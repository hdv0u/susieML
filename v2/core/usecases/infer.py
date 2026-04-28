from core.engine.infer_engine import InferEngine
from core.detection.screen_detector import ScreenDetector

def build_detector(cfg_ctrl, model_path, log_fn):
    engine = InferEngine(cfg_ctrl, model_path, log_fn=log_fn)
    detector = ScreenDetector(engine=engine, cfg=cfg_ctrl)
    return detector