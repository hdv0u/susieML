import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
GENERATIONS = 50
LEARNING_RATE = 5e-4

MODEL_SAVE_PATH = "models/"

CNN = {
    "threshold": 0.67,
    "side_len": 128,
    "steps": 96,
    "window_history": 5
}
DENSE = {
    "input_size": 128*128*3,
    "cooldown": 0.5,
    "threshold": 0.5,
    "channels": 3
}
RESNET = {
    "threshold": 0.67,
    "side_len": 128,
    "steps": 64,
    "window_history": 5,
}