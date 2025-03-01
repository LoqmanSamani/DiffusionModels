import torch
from tqdm import tqdm
import numpy as np
from forward_ddim import ForwardDDIM




class Train:
    def __init__(self, config, model=None, optimizer=None, loss_function=None):
        """
        handles the training process for a denoising model using the DDIM framework.
        initializes the model, optimizer, and loss function, and performs training
        using forward diffusion.

          :param config: configuration object containing model, optimizer, loss function, and training parameters
          :param model: optional, the neural network (u-net) model to be trained (default: uses config.model)
          :param optimizer: optional, optimizer for training the model (default: uses config.optimizer)
          :param loss_function: optional, loss function used for training (default: uses config.loss_function)
        """

        self.config = config
        if model:
            self.model = model
        else:
            self.model = config.model
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = config.optimizer
        if loss_function:
            self.loss_function = loss_function
        else:
            self.loss_function = config.loss_function
        self.model_path = config.model_path
        self.train_loader = config.train_loader
        self.num_epochs = config.train_epochs
        self.in_channels = config.in_channels
        self.learning_rate = config.learning_rate
        self.device = config.device
        self.num_steps = config.num_steps

        # forward diffusion process
        self.forward_diffusion = ForwardDDIM(self.config)

    def fit(self):
        """
        trains the model using the provided dataset and optimizes based on noise prediction.
        the training process involves:
            - applying forward diffusion to add noise to images
            - predicting the noise using the model
            - computing the loss and performing backpropagation
            - saving the best model based on validation loss

        :return: None, prints training progress and saves the best model
        """
        self.model.to(self.device)
        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            losses = []
            self.model.train()
            # loop over dataloader
            for x, _ in tqdm(self.train_loader):

                x = x.to(self.device)
                self.optimizer.zero_grad()
                # generate noise and timestamps
                noise = torch.randn_like(x).to(self.device)
                t = torch.randint(0, self.num_steps, (x.shape[0],)).to(self.device)
                # add noise to the images using forward process
                noisy_x = self.forward_diffusion(x, noise, t)
                print(noisy_x.shape)
                print(t.shape)
                # predict noise
                p_noise = self.model(noisy_x, t)
                # calculate loss
                loss = self.loss_function(p_noise, noise)
                losses.append(loss.item())
                # backward pass
                loss.backward()
                self.optimizer.step()

            mean_epoch_loss = np.mean(losses)
            print(f"\nEpoch: {epoch+1} | Loss: {mean_epoch_loss: .4f}")
            if mean_epoch_loss < best_loss:
                best_loss = mean_epoch_loss
                torch.save(self.model, self.model_path)

        print(f'Training Process is Finished!')