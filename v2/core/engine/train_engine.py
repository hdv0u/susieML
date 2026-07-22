import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset_loader import load_dataset
from data.preproc import build_dataset
from models.convnn import SussyCNN
from models.convnn_fcn import SussyCNN_FCN

class TrainEngine:
    def __init__(self, cfg, log_fn=print):
        self.cfg = cfg
        self.log = log_fn
        
    def _prepare_data(self, json_path):
        path, labels = load_dataset(json_path)
        X, y = build_dataset(path, labels, self.cfg)
        
        X = X.float()
        y = y.float()
        
        dataset = list(zip(X, y))
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.get_value("training", "batch_size"),
            shuffle=True,
            pin_memory=True
        )
        return loader
    
    def _build_model(self):
        mode = self.cfg.get_value("model", "mode", "patch")
        print(f"Building model for mode: {mode}")
        if mode == "fcn":
            self.log("Building FCN model")
            model = SussyCNN_FCN(
                out_channels=self.cfg.get_value("model", "out_channels")
            )
        else:
            self.log("Building patch-based CNN model")
            model = SussyCNN(
                out_channels=self.cfg.get_value("model", "out_channels")
            )
        return model
    
    def train(self, dataset_path, save_path, stop_ctrl=None, progress_fn=None, override_epochs=None):
        mode = self.cfg.get_value("model", "mode", "patch")
        
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        model = self._build_model().to(device)
        loader = self._prepare_data(dataset_path)
        loss_fn = nn.BCEWithLogitsLoss()
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.get_value("training", "learning_rate")
        )
        epochs = override_epochs or self.cfg.get_value("training", "epochs")
        
        epsilon = self.cfg.get_value("training", "epsilon")
        
        for epoch in range(epochs):
            if stop_ctrl and not stop_ctrl.is_running():
                self.log("Training stopped")
                return
            model.train()
            total_loss = 0.0
            
            for X_batch, y_batch in loader:
                if stop_ctrl and not stop_ctrl.is_running():
                    self.log("Training stopped")
                    return
                
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                if outputs.dim() == 4:
                    y_batch = y_batch.view(y_batch.size(0), y_batch.size(1), 1, 1)
                    y_batch = y_batch.repeat(1,1,outputs.size(2), outputs.size(3))
                  
                y_batch = y_batch * (1 - epsilon) + (epsilon / 2) # label smoothing
                loss = loss_fn(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                avg_loss = total_loss / len(loader)
            
            self.log(f"Epoch {epoch+1}/{epochs} | loss: {avg_loss:.4f}")
            
            if progress_fn:
                progress_fn(int((epoch + 1) / epochs * 100))
                
        torch.save(model.state_dict(), save_path)
        self.log(f"Model saved to {save_path}")