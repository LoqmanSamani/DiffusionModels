import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from noise_predictor import NoisePredictor
from text_encoder import TextEncoder
from hyper_param import HyperParamsSDE
from train_sde import TrainSDE

torch.serialization.add_safe_globals([HyperParamsSDE])


class MockDataset(Dataset):

    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, 64, 64)
        self.labels = [f"label_{i}" for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def test_ddpm_training():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda")
    noise_model = NoisePredictor(
        in_channels=3,
        down_channels=[32, 64],
        mid_channels=[64, 64],
        up_channels=[64, 32],
        down_sampling=[True, True],
        time_embed_dim=64,
        y_embed_dim=64,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        dropout_rate=0.1,
        down_sampling_factor=2,
        where_y=True,
        y_to_all=False
    )
    conditional_model = TextEncoder(
        use_pretrained_model=True,
        model_name="bert-base-uncased",
        vocabulary_size=30522,
        num_layers=2,
        input_dimension=64,
        output_dimension=64,
        num_heads=1,
        context_length=77,
        dropout_rate=0.1,
        qkv_bias=False,
        scaling_value=4,
        epsilon=1e-5
    )

    hyper_params = HyperParamsSDE(num_steps=100, beta_start=1e-4, beta_end=0.02, beta_method="linear",
                                  sigma_start=1e-3, sigma_end=1.0, start=0.0, end=1.0)
    train_dataset = MockDataset(num_samples=10)
    val_dataset = MockDataset(num_samples=5)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    optimizer = torch.optim.Adam(
        list(noise_model.parameters()) + list(conditional_model.parameters()),
        lr=1e-3
    )
    objective = nn.MSELoss()

    trainer1 = TrainSDE(
        method="ode",  # ve, vp, sub-vp, ode
        noise_predictor=noise_model,
        hyper_params_model=hyper_params,
        data_loader=train_loader,
        optimizer=optimizer,
        objective=objective,
        val_loader=val_loader,
        max_epoch=3,
        device=device,
        conditional_model=conditional_model,
        store_path="test_ddpm.pth",
        patience=2,
        warmup_epochs=1
    )

    trainer2 = TrainSDE(
        method="ode",  # ve, vp, sub-vp, ode
        noise_predictor=noise_model,
        hyper_params_model=hyper_params,
        data_loader=train_loader,
        optimizer=optimizer,
        objective=objective,
        val_loader=val_loader,
        max_epoch=3,
        device=device,
        conditional_model=None,
        store_path="test_ddpm.pth",
        patience=2,
        warmup_epochs=1
    )

    print("Starting training...")
    try:
        train_losses1, best_val_loss1 = trainer1()
        print(f"Training completed. Final train losses: {train_losses1}")
        print(f"Best validation loss: {best_val_loss1:.4f}")

        assert os.path.exists("test_ddpm.pth"), "Checkpoint file was not saved"

        epoch1, loss1 = trainer1.load_checkpoint("test_ddpm.pth")
        print(f"Loaded checkpoint from epoch {epoch1} with loss {loss1:.4f}")
        assert epoch1 >= 0, "Invalid epoch loaded from checkpoint"
        assert loss1 < float('inf'), "Invalid loss loaded from checkpoint"

        assert len(train_losses1) > 0, "No training losses recorded"
        assert all(isinstance(l, float) for l in train_losses1), "Invalid training losses"
        assert isinstance(best_val_loss1, float), "Invalid best validation loss"

        print("All tests passed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

    finally:
        if os.path.exists("test_ddpm.pth"):
            os.remove("test_ddpm.pth")
        if os.path.exists("test_ddpm.pth_early_stop.pth"):
            os.remove("test_ddpm.pth_early_stop.pth")

    try:
        train_losses2, best_val_loss2 = trainer2()
        print(f"Training completed. Final train losses: {train_losses2}")
        print(f"Best validation loss: {best_val_loss2:.4f}")

        assert os.path.exists("test_ddpm.pth"), "Checkpoint file was not saved"

        epoch2, loss2 = trainer2.load_checkpoint("test_ddpm.pth")
        print(f"Loaded checkpoint from epoch {epoch2} with loss {loss2:.4f}")
        assert epoch2 >= 0, "Invalid epoch loaded from checkpoint"
        assert loss2 < float('inf'), "Invalid loss loaded from checkpoint"

        assert len(train_losses2) > 0, "No training losses recorded"
        assert all(isinstance(l, float) for l in train_losses2), "Invalid training losses"
        assert isinstance(best_val_loss2, float), "Invalid best validation loss"

        print("All tests passed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

    finally:
        if os.path.exists("test_ddpm.pth"):
            os.remove("test_ddpm.pth")
        if os.path.exists("test_ddpm.pth_early_stop.pth"):
            os.remove("test_ddpm.pth_early_stop.pth")


if __name__ == "__main__":
    test_ddpm_training()