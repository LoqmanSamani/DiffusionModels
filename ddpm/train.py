import torch
from tqdm import tqdm
import numpy as np
from forward_ddpm import ForwardDDPM




class TrainConfig:
    """configuration class for training"""

    def __init__(
            self,
            model, # initialized u-net model
            train_loader, # initialized train loader
            optimizer, # initialized train optimizer
            loss, # initialized loss metric
            save_path, # path to store trained model/s
            num_epochs=100,
            in_channels=3, # for RGB images
            learning_rate=0.1e-4,
            num_diffusion_steps=1000,
            num_time_steps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            device=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss = loss
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.num_diffusion_steps = num_diffusion_steps
        self.num_time_steps = num_time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class Train:

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = config.model
        self.optimizer = config.optimizer
        self.loss = config.loss
        self.save_path = config.save_path
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
