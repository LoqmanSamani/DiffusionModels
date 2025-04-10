import torch
import numpy as np
from tqdm import tqdm
from forward_ddpm import ForwardDDPM  # Assuming this is the updated class


class DDPMTrain:
    """Trainer for Denoising Diffusion Probabilistic Models (DDPM)."""
    def __init__(self, noise_predictor, hyper_params_model, train_loader, optimizer, objective, val_loader=None, in_channels=3, num_steps=1000, max_epochs=1000, device=None, conditional_model=None, store_path=None):

        self.noise_predictor = noise_predictor
        self.hyper_params_model = hyper_params_model
        self.conditional_model = conditional_model
        self.optimizer = optimizer
        self.objective = objective
        self.store_path = store_path or ""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.in_channels = in_channels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_steps = num_steps

        # Forward diffusion process
        self.forward_diffusion = ForwardDDPM(
            hyper_params=self.hyper_params_model
        ).to(self.device)


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
