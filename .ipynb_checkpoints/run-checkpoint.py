from steps.trainer import train_func_per_worker
import ray.train.torch
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_Glucoma(num_workers=4, use_gpu=True):
    global_batch_size = 2048

    train_config = {
        "lr": 1e-3,
        "epochs": 1,
        "batch_size_per_worker": global_batch_size // num_workers,
    }

    # Configure computation resources
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu,trainer_resources={"CPU": 8},resources_per_worker={"CPU": 4,"GPU": 1,})

    # Initialize a Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
    )

    # [4] Start distributed training
    # Run `train_func_per_worker` on all workers
    # =============================================
    result = trainer.fit()
    print(f"Training result: {result}")

if __name__ == "__main__":
    train_Glucoma(num_workers=3, use_gpu=True)