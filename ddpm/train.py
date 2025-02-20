import torch
from tqdm import tqdm
import numpy as np
from forward_ddpm import ForwardDDPM




class Train:

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = config.model
        self.optimizer = config.optimizer
        self.loss = config.loss
        self.save_path = config.model_path
        self.train_loader = config.train_loader
        self.num_epochs = config.num_epochs
        self.in_channels = config.in_channels
        self.learning_rate = config.learning_rate
        self.device = config.device
        self.num_time_steps = config.num_time_steps

        # forward diffusion process
        self.forward_diffusion = ForwardDDPM(
            num_steps=config.num_diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end
        )

    def fit(self):

        self.model.to(self.device)
        best_loss = float('inf')

        for epoch in range(self.num_epochs):

            losses = []
            self.model.train()

            # loop over dataloader
            for batch, _ in tqdm(self.train_loader):
                #print(batch.shape)

                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                # generate noise and timestamps
                noise = torch.randn_like(batch).to(self.device)
                t = torch.randint(low=0, high=self.num_time_steps, size=(batch.shape[0], )).to(self.device)

                # add noise to the images using forward process
                noisy_batch = self.forward_diffusion.add_noise(batch=batch, noise=noise, time_steps=t)

                # predict noise
                predicted_noise = self.model(batch=noisy_batch, t=t)
                #predicted_noise = torch.where(torch.isnan(predicted_noise), torch.zeros_like(predicted_noise), predicted_noise)

                # calculate loss
                loss = self.loss(predicted_noise, noise)
                losses.append(loss.item())

                # backward pass
                loss.backward()
                self.optimizer.step()

            mean_epoch_loss = np.mean(losses)
            print()
            print(f"Epoch: {epoch+1} | Loss: {mean_epoch_loss: .4f}")

            if mean_epoch_loss < best_loss:
                best_loss = mean_epoch_loss
                torch.save(self.model, self.save_path)

        print(f'Training Process is Finished!')




