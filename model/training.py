import torch
import pytorch_lightning as pl
from model.model import FasterRCNNLightning
import mlflow
import mlflow.pytorch

if __name__ == '__main__':
    mlflow.set_experiment("fasterrcnn_object_detection")

    with mlflow.start_run():
        model = FasterRCNNLightning()
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(model)
        torch.save(model.model, "model/fasterrcnn.pth")

        # Save PyTorch model
        mlflow.pytorch.log_model(model.model, "model")
        mlflow.log_param("learning_rate", 0.005)
        mlflow.log_param("epochs", 10)
