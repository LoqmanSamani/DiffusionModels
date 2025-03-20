import torch
from reverse_diffusion import ReverseSDE





class Generate:
    def __init__(self, config, model, device=None):
        self.config = config
        self.model = model
        if device:
            self.device = device
        else:
            self.device = config.device
        self.reverse = ReverseSDE(self.config, self.model)


    @torch.no_grad()
    def forward(self):
        self.model.eval()
        frames = torch.zeros(self.config.max_steps, self.config.image_shape[0], self.config.image_shape[1], self.config.image_shape[-1]).to(self.device)
        noisy_sample = torch.randn(1, self.config.image_shape[0], self.config.image_shape[1], self.config.image_shape[-1]).to(self.device)

        f = 0
        for t in reversed(range(1, self.config.max_steps)):
            noisy_sample = self.reverse.forward(noisy_sample, torch.as_tensor((t,)).to(self.device), self.device)
            frames[f, :, :, :] = noisy_sample
            f += 1

        output = torch.clamp(frames, -1., 1.).detach().cpu()
        output = (output + 1) / 2

        return output