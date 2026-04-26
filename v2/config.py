MODEL = {
    "name": "cnn",
    "arch_depth": 1,
    "input_size": (128,128),
    "out_channels": 1
}

TRAINING = {
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 5e-4,
    "optimizer": "adam",
    "loss": "bce_with_logits",
    "shuffle": True,
    "device": "auto",
}

AUGMENT = {
    "augment_count": 5,
    "flip_enabled": True,
    "brightness_enabled": True,
    "brightness_min": 0.8,
    "brightness_max": 1.2,
    "shift_enabled": True,
    "shift_max": 0.1,
}

INFERENCE = {
    "threshold": 0.67,
    "side_len": 128,
    "steps": 96,
    "batch_size": 4,
}

GLOBAL = {
    "seed": 42,
    "log_level": "info",
}

cfg = {
    "model": MODEL,
    "augment": AUGMENT,
    "training": TRAINING,
    "inference": INFERENCE,
    "global": GLOBAL
}