import torch
from ui.window_test import TestWindow
# universal constants(except for denseNN with device)
win = TestWindow()
settings = win.get_settings()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
GENERATIONS = 50
LEARNING_RATE = 5e-4

MODEL_SAVE_PATH = "py/models/"

CNN = {
    "threshold": settings.get("threshold", 0.67),
    "side_len": settings.get("side_len", 128),
    "steps": settings.get("steps", 96),
    "window_history": 5,
    "epsilon": 0.05,
    "label_smoothing": 0.1,
}
DENSE = {
    "input_size": 128*128*3,
    "cooldown": 0.5,
    "threshold": settings.get("threshold", 0.67),
    "channels": 3,
    "side_len": settings.get("side_len", 128),
    "steps": settings.get("steps", 96),
    "window_history": 1
}
RESNET = {
    "threshold": settings.get("threshold", 0.67),
    "side_len": settings.get("side_len", 128),
    "steps": settings.get("steps", 96),
    "window_history": 5,
}