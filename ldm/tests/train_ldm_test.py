import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from noise_predictor import NoisePredictor
from text_encoder import TextEncoder
from hyper_param import HyperParamsDDPM, HyperParamsSDE, HyperParamsDDIM
from forward_idm import ForwardDDPM, ForwardSDE, ForwardDDIM
from autoencoder import AutoencoderLDM
from train_ldm import TrainLDM




#torch.serialization.add_safe_globals([HyperParamsDDPM])
#torch.serialization.add_safe_globals([HyperParamsDDIM])
torch.serialization.add_safe_globals([HyperParamsSDE])


class MockDataset(Dataset):

    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, 128, 128)
        self.labels = [f"label_{i}" for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def test_ldm_training():

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda")
    auto_encod = AutoencoderLDM(
        in_channels=3,
        down_channels=[8, 16],
        up_channels=[8, 16],
        out_channels=3,
        latent_channels=3,
        dropout_rate=0.1,
        num_heads=1,
        num_groups=8,
        num_layers_per_block=1,
        total_down_sampling_factor=2,
        use_vq=False,
        num_embeddings=16,
        beta=1e-4
    )
    noise_model = NoisePredictor(
        in_channels=3,
        down_channels=[16, 32],
        mid_channels=[32, 32],
        up_channels=[32, 16],
        down_sampling=[True, True],
        time_embed_dim=32,
        y_embed_dim=32,
        num_down_blocks=1,
        num_mid_blocks=1,
        num_up_blocks=1,
        dropout_rate=0.1,
        down_sampling_factor=2,
        where_y=True,
        y_to_all=False
    )
    conditional_model = TextEncoder(
        use_pretrained_model=True,
        model_name="bert-base-uncased",
        vocabulary_size=30522,
        num_layers=1,
        input_dimension=32,
        output_dimension=32,
        num_heads=1,
        context_length=77,
        dropout_rate=0.1,
        qkv_bias=False,
        scaling_value=4,
        epsilon=1e-5
    )
    hyper_params = HyperParamsSDE(num_steps=100, beta_start=1e-4, beta_end=0.02, beta_method="linear")
    #hyper_params = HyperParamsDDPM(num_steps=100, beta_start=1e-4, beta_end=0.02, beta_method="linear")
    #hyper_params = HyperParamsDDIM(num_steps=100, beta_start=1e-4, beta_end=0.02, beta_method="linear")
    forward = ForwardSDE(hyper_params, "vp")
    #forward = ForwardDDPM(hyper_params)
    #forward = ForwardDDIM(hyper_params)
    train_dataset = MockDataset(num_samples=10)
    val_dataset = MockDataset(num_samples=5)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    optimizer = torch.optim.Adam(
        [p for p in noise_model.parameters() if p.requires_grad] +
        [p for p in conditional_model.parameters() if p.requires_grad],
        lr=1e-3
    )

    objective = nn.MSELoss()

    trainer1 = TrainLDM(
        forward_model=forward,
        hyper_params_model=hyper_params,
        noise_predictor=noise_model,
        compressor_model=auto_encod,
        data_loader=train_loader,
        optimizer=optimizer,
        objective=objective,
        val_loader=val_loader,
        max_epoch=3,
        device=device,
        conditional_model=conditional_model,
        store_path="test_ldm1.pth",
        patience=2
    )

    trainer2 = TrainLDM(
        forward_model=forward,
        hyper_params_model=hyper_params,
        noise_predictor=noise_model,
        compressor_model=auto_encod,
        data_loader=train_loader,
        optimizer=optimizer,
        objective=objective,
        val_loader=val_loader,
        max_epoch=3,
        device=device,
        conditional_model=None,
        store_path="test_ldm2.pth",
        patience=2
    )

    print("Starting training...")

    try:
        train_losses1, best_val_loss1 = trainer1()
        print(f"Training completed. Final train losses: {train_losses1}")
        print(f"Best validation loss: {best_val_loss1:.4f}")

        assert os.path.exists("test_ldm1.pth"), "Checkpoint file was not saved"

        epoch1, loss1 = trainer1.load_checkpoint("test_ldm1.pth")
        print(f"Loaded checkpoint from epoch {epoch1} with loss {loss1:.4f}")
        assert epoch1 >= 0, "Invalid epoch loaded from checkpoint"
        assert loss1 < float('inf'), "Invalid loss loaded from checkpoint"

        assert len(train_losses1) > 0, "No training losses recorded"
        assert all(isinstance(l, float) for l in train_losses1), "Invalid training losses"
        assert isinstance(best_val_loss1, float), "Invalid best validation loss"

        print("All tests (conditional) passed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

    finally:
        if os.path.exists("test_ldm1.pth"):
            os.remove("test_ldm1.pth")
        if os.path.exists("test_ldm1.pth_early_stop.pth"):
            os.remove("test_ldm1.pth_early_stop.pth")

    print("Starting training...")

    try:
        train_losses2, best_val_loss2 = trainer2()
        print(f"Training completed. Final train losses: {train_losses2}")
        print(f"Best validation loss: {best_val_loss2:.4f}")

        assert os.path.exists("test_ldm2.pth"), "Checkpoint file was not saved"

        epoch2, loss2 = trainer1.load_checkpoint("test_ldm2.pth")
        print(f"Loaded checkpoint from epoch {epoch2} with loss {loss2:.4f}")
        assert epoch2 >= 0, "Invalid epoch loaded from checkpoint"
        assert loss2 < float('inf'), "Invalid loss loaded from checkpoint"

        assert len(train_losses2) > 0, "No training losses recorded"
        assert all(isinstance(l, float) for l in train_losses2), "Invalid training losses"
        assert isinstance(best_val_loss2, float), "Invalid best validation loss"

        print("All tests (unconditional) passed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

    finally:
        if os.path.exists("test_ldm2.pth"):
            os.remove("test_ldm2.pth")
        if os.path.exists("test_ldm2.pth_early_stop.pth"):
            os.remove("test_ldm2.pth_early_stop.pth")


if __name__ == "__main__":
    test_ldm_training()