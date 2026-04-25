import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset_loader import load_dataset
from data.preproc import build_dataset
from models.convnn import SussyCNN

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
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=True,
            pin_memory=True
        )
        return loader
    
    def _build_model(self):
        model = SussyCNN(
            out_channels=self.cfg["model"]["out_channels"]
        )
        return model
    
    def train(self, dataset_path, save_path, stop_flag=None, progress_fn=None, override_epochs=None):
        
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        model = self._build_model().to(device)
        loader = self._prepare_data(dataset_path)
        loss_fn = nn.BCEWithLogitsLoss()
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg["training"]["learning_rate"]
        )
        epochs = override_epochs if override_epochs is not None else self.cfg["training"]["epochs"]
        
        epsilon = self.cfg["training"].get("epsilon", 0.05)
        
        for epoch in range(epochs):
            if stop_flag and stop_flag():
                self.log("Training stopped")
                return
            model.train()
            total_loss = 0.0
            
            for X_batch, y_batch in loader:
                
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_batch = y_batch * (1 - epsilon) + (epsilon / 2) # label smoothing
                
                optimizer.zero_grad()
                
                outputs = model(X_batch)
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