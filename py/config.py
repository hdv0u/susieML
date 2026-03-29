import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
GENERATIONS = 50
LEARNING_RATE = 5e-4

MODEL_SAVE_PATH = "py/models/"

DEFAULT_SETTINGS = {
    "threshold": 0.67,
    "side_len": 128,
    "steps": 96,
}

def update_settings(settings: dict):
    if not settings:
        return
    
    DEFAULT_SETTINGS.update(settings)
    
    if "threshold" in settings:
        CNN["threshold"] = settings["threshold"]
        DENSE["threshold"] = settings["threshold"]
        RESNET["threshold"] = settings["threshold"]
    
    if "side_len" in settings:
        CNN["side_len"] = settings["side_len"]
        DENSE["side_len"] = settings["side_len"]
        RESNET["side_len"] = settings["side_len"]

    if "steps" in settings:
        CNN["steps"] = settings["steps"]
        DENSE["steps"] = settings["steps"]
        RESNET["steps"] = settings["steps"]

CNN = {
    "threshold": DEFAULT_SETTINGS["threshold"],
    "side_len": DEFAULT_SETTINGS["side_len"],
    "steps": DEFAULT_SETTINGS["steps"],
    "window_history": 5,
    "epsilon": 0.05,
    "label_smoothing": 0.1,
}
DENSE = {
    "input_size": 128*128*3,
    "cooldown": 0.5,
    "threshold": DEFAULT_SETTINGS["threshold"],
    "channels": 3,
    "side_len": DEFAULT_SETTINGS["side_len"],
    "steps": DEFAULT_SETTINGS["steps"],
    "window_history": 1
}
RESNET = {
    "threshold": DEFAULT_SETTINGS["threshold"],
    "side_len": DEFAULT_SETTINGS["side_len"],
    "steps": DEFAULT_SETTINGS["steps"],
    "window_history": 5,
}