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
    "arch_depth": 5,
    "lr": 5e-4,
    "gen": 50,
    "augment_count": 5,
}

AUGMENT_SETTINGS = {
    "augment_count": 5,
    "flip_enabled": True,
    "brightness_enabled": True,
    "brightness_min": 0.8,
    "brightness_max": 1.2,
    "shift_enabled": True,
    "shift_max": 0.1,
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

    if "arch_depth" in settings:
        CNN["arch_depth"] = settings["arch_depth"]
        DENSE["arch_depth"] = settings["arch_depth"]
        RESNET["arch_depth"] = settings["arch_depth"]
    
    if "lr" in settings:
        CNN["lr"] = settings["lr"]
        DENSE["lr"] = settings["lr"]
        RESNET["lr"] = settings["lr"]
    
    if "gen" in settings:
        CNN["gen"] = settings["gen"]
        DENSE["gen"] = settings["gen"]
        RESNET["gen"] = settings["gen"]
    
    if "augment_count" in settings:
        CNN["augment_count"] = settings["augment_count"]
        DENSE["augment_count"] = settings["augment_count"]
        RESNET["augment_count"] = settings["augment_count"]
    
    # Update augmentation settings
    augment_keys = ["augment_count", "flip_enabled", "brightness_enabled", "brightness_min", "brightness_max", "shift_enabled", "shift_max"]
    for key in augment_keys:
        if key in settings:
            AUGMENT_SETTINGS[key] = settings[key]

CNN = {
    "threshold": DEFAULT_SETTINGS["threshold"],
    "side_len": DEFAULT_SETTINGS["side_len"],
    "steps": DEFAULT_SETTINGS["steps"],
    "arch_depth": DEFAULT_SETTINGS["arch_depth"],
    "lr": DEFAULT_SETTINGS["lr"],
    "gen": DEFAULT_SETTINGS["gen"],
    "augment_count": DEFAULT_SETTINGS["augment_count"],
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
    "arch_depth": DEFAULT_SETTINGS["arch_depth"],
    "lr": DEFAULT_SETTINGS["lr"],
    "gen": DEFAULT_SETTINGS["gen"],
    "augment_count": DEFAULT_SETTINGS["augment_count"],
    "window_history": 1
}
RESNET = {
    "threshold": DEFAULT_SETTINGS["threshold"],
    "side_len": DEFAULT_SETTINGS["side_len"],
    "steps": DEFAULT_SETTINGS["steps"],
    "arch_depth": DEFAULT_SETTINGS["arch_depth"],
    "lr": DEFAULT_SETTINGS["lr"],
    "gen": DEFAULT_SETTINGS["gen"],
    "augment_count": DEFAULT_SETTINGS["augment_count"],
    "window_history": 5,
}