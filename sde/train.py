import torch
from forward_diffusion import ForwardSDE
from tqdm import tqdm







class Train:
    def __init__(self, config, model, data_loader, optimizer=None, objective=None, device=None):
        self.config = config
        self.model = model
        self.data_loader = data_loader
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = config.optimizer
        if objective:
            self.objective = objective
        else:
            self.objective = config.objective
        if device:
            self.device = device
        else:
            self.device = config.device

        self.forward_diff = ForwardSDE(config)

    def train(self):
        self.model.to(self.device)
        losses = []
        best_loss = float("inf")

        for epoch in range(self.config.max_epoch):
            losses_ = []
            self.model.train()

            for x, _ in tqdm(self.data_loader):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                # generate noise and timestamps
                noise = torch.randn_like(x).to(self.device)
                t = torch.randint(0, self.config.max_steps, (x.shape[0],)).to(self.device)
                # add noise to the images using forward process. the type of method (smld, vp or sub-vp) is already defined in config
                noisy_x = self.forward_diff.forward(x, t, self.device)

                # predict noise
                p_noise = self.model(noisy_x, t)
                # calculate loss
                loss = self.objective(p_noise, noise)
                losses_.append(loss.item())
                # backward pass
                loss.backward()
                self.optimizer.step()

            mean_loss = torch.mean(torch.tensor(losses_)).item()
            losses.append(mean_loss)
            print(f"\nEpoch: {epoch+1} | Loss: {mean_loss: .4f}")

            if epoch % self.config.checkpoint == 0:
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    torch.save(self.model, self.config.save_path)

        return losses, best_loss