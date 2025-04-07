import torch
import numpy as np
from tqdm import tqdm
from forward_diffusion import ForwardDDPM  # Assuming this is the updated class


class DDPMTrain:
    """Trainer for Denoising Diffusion Probabilistic Models (DDPM)."""
    def __init__(self, config):
        self.config = config
        self.model = config.model
        self.save_path = config.model_path
        self.train_loader = config.train_loader
        self.num_epochs = config.num_epochs
        self.in_channels = config.in_channels
        self.learning_rate = config.learning_rate
        self.device = config.device
        self.num_steps = config.num_diffusion_steps  # Renamed for consistency

        # Initialize optimizer if not provided
        self.optimizer = config.optimizer if hasattr(config, 'optimizer') else torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        # Initialize loss function if not provided (default to MSE)
        self.loss = config.loss if hasattr(config, 'loss') else torch.nn.MSELoss()

        # Forward diffusion process
        self.forward_diffusion = ForwardDDPM(
            num_steps=self.num_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end
        ).to(self.device)

        # Validate timestep consistency
        if hasattr(config, 'num_time_steps') and config.num_time_steps != self.num_steps:
            raise ValueError(f"num_time_steps ({config.num_time_steps}) must equal num_diffusion_steps ({self.num_steps})")

    def fit(self):
        """Trains the DDPM model to predict noise added by the forward diffusion process."""
        self.model.to(self.device)
        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses = []

            # Progress bar over batches
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=True) as pbar:
                for batch, _ in pbar:
                    batch = batch.to(self.device)
                    self.optimizer.zero_grad()

                    # Generate noise and timesteps
                    noise = torch.randn_like(batch).to(self.device)
                    t = torch.randint(0, self.num_steps, (batch.shape[0],), device=self.device)

                    # Add noise using forward diffusion
                    noisy_batch = self.forward_diffusion(batch, noise, t)

                    # Predict noise
                    predicted_noise = self.model(batch=noisy_batch, t=t)

                    # Compute loss
                    loss = self.loss(predicted_noise, noise)
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss detected at epoch {epoch+1}")
                        continue

                    epoch_losses.append(loss.item())

                    # Backpropagation
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Prevent exploding gradients
                    self.optimizer.step()

                    # Update progress bar
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            mean_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            print(f"Epoch {epoch+1}/{self.num_epochs} | Mean Loss: {mean_epoch_loss:.4f}")

            # Save best model (using state_dict for portability)
            if mean_epoch_loss < best_loss:
                best_loss = mean_epoch_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  Saved best model with loss {best_loss:.4f} to {self.save_path}")

        print("Training process completed!")
