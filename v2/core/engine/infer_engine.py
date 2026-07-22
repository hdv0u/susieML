import torch
from models.convnn import SussyCNN
from models.convnn_fcn import SussyCNN_FCN
class InferEngine:
    def __init__(self, cfg, model_path, log_fn=print):
        self.cfg = cfg
        self.log = log_fn
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        mode = self.cfg.get_value("model", "mode")
        if mode == "patch":
            self.model = SussyCNN(
                out_channels=self.cfg.get_value("model", "out_channels")
            ).to(self.device)
        elif mode == "fcn":
            self.model = SussyCNN_FCN(
                out_channels=self.cfg.get_value("model", "out_channels")
            ).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        
    def forward(self, batch_tensor):
        with torch.no_grad():
            return self.model(batch_tensor)