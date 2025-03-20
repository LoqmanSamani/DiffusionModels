import torch





class Config:
    def __init__(
            self,
            in_channels,  # number of input channels (e.g., 3 for RGB images)
            down_channels,  # list of channels used in the down-sampling path (e.g., [32, 64, 128, 256])
            mid_channels,  # list of channels in the middle block of the network (e.g., [256, 256, 128])
            up_channels,  # list of channels used in the up-sampling path (e.g., [256, 128, 64, 16])
            down_sampling,  # list of booleans specifying where down-sampling is applied (e.g., [True, True, False])
            num_groups,  # number of groups used in group normalization (nn.GroupNorm)
            embed_dim=None,  # integer specifying the embedding dimension for time encoding
            num_down_blocks=None,  # number of blocks in the down-sampling path
            num_mid_blocks=None,  # number of blocks in the middle part of the network
            num_up_blocks=None,  # number of blocks in the up-sampling path
            dropout_rate=None,  # dropout rate used to prevent overfitting
            num_attention_heads=None,  # number of attention heads in self-attention layers
            down_sampling_factor=None,  # factor that determines the down-sampling rate
            upsampling_factor=None,  # factor that determines the up-sampling rate
            apply_down_conv=None,  # boolean indicating if convolution layers should be applied in down-sampling
            apply_down_pool=None,  # boolean indicating if max pooling should be applied in down-sampling
            apply_up_conv=None,  # boolean indicating if convolution layers should be applied in up-sampling
            kernel_size=None,  # kernel size of convolution layers
            norm=None,  # boolean indicating if group normalization should be applied
            activation=None,  # boolean indicating if silu (swish) activation should be applied
            method=None,  # training method, options: "ve", "vp", or "sub-vp"
            start=None,  # start time for scheduling
            end=None,  # end time for scheduling
            max_steps=None,  # total number of steps in the diffusion process
            sigma_min=None,  # minimum noise level for variance-exploding SDE
            sigma_max=None,  # maximum noise level for variance-exploding SDE
            beta_range=None,  # range of beta values used in variance-preserving diffusion models
            beta_schedule_method=None,  # method for scheduling beta values ("linear", "sigmoid", etc.)
            max_epoch=None,  # maximum number of epochs for training
            device=None,  # computing device (e.g., "cuda" or "cpu")
            optimizer=None,  # optimizer used for training
            objective=None,  # loss function used for training
            save_path=None,  # file path to save the trained model
            checkpoint=None,  # frequency (in epochs) to save the model checkpoint
            image_shape=None # shape of image to be generated in generation phase
    ):
        self.in_channels = in_channels
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.up_channels = up_channels
        self.down_sampling = down_sampling
        self.num_groups = num_groups
        self.embed_dim = embed_dim or 256
        self.num_down_blocks = num_down_blocks or 2
        self.num_mid_blocks = num_mid_blocks or 2
        self.num_up_blocks = num_up_blocks or 2
        self.dropout_rate = dropout_rate or 0.2
        self.num_attention_heads = num_attention_heads or 4
        self.down_sampling_factor = down_sampling_factor or 2
        self.upsampling_factor = upsampling_factor or 2
        self.apply_down_conv = apply_down_conv or True
        self.apply_down_pool = apply_down_pool or True
        self.apply_up_conv = apply_up_conv or True
        self.kernel_size = kernel_size or 3
        self.norm = norm or True
        self.activation = activation or True
        self.method = method or "ve"  # default training method is "ve"
        self.start = start or 0
        self.end = end or 1
        self.max_steps = max_steps or 1000
        self.step_size = (self.end - self.start) / self.max_steps  # step size for time discretization
        self.sigma_min = sigma_min or 0.01
        self.sigma_max = sigma_max or 1.0
        self.beta_range = beta_range or (1e-4, 0.02)
        self.beta_schedule_method = beta_schedule_method or "linear"
        self.max_epoch = max_epoch or int(1e4)
        self.device = device or "cuda"
        self.optimizer = optimizer
        self.objective = objective or torch.nn.MSELoss()
        self.save_path = save_path or "default_model.pth"
        self.checkpoint = checkpoint or 100
        self.image_shape = image_shape or (3, 32, 32)


        # compute hyperparameters (sigmas for VE SDE, betas for VP/sub-VP SDEs, and time steps)
        self.sigmas, self.betas, self.cum_betas, self.t = self.compute_params()

    def compute_params(self):
        """
        computes time steps, sigmas for variance-exploding SDE,
        betas for variance-preserving SDE, and their cumulative sum.
        """
        t = torch.linspace(self.start, self.end, self.max_steps)  # evenly spaced time steps
        sigmas = self.sigma_min * (self.sigma_max / self.sigma_min) ** t  # geometric variance schedule
        betas = self.compute_beta_schedule(self.beta_range, self.max_steps, self.beta_schedule_method)
        cum_betas = torch.cumsum(betas, dim=0)  # cumulative sum of betas for VP SDE

        return sigmas, betas, cum_betas, t

    def compute_beta_schedule(self, beta_range, num_steps, method):
        """
        computes beta values based on different scheduling methods.
        beta controls the noise variance in VP diffusion models.
        """
        if method == "linear":
            betas = torch.linspace(beta_range[0], beta_range[1], num_steps)
        elif method == "sigmoid":
            betas = torch.linspace(-6, 6, num_steps)
            betas = torch.sigmoid(betas) * (beta_range[1] - beta_range[0]) + beta_range[0]
        elif method == "quadratic":
            betas = torch.linspace(beta_range[0] ** 0.5, beta_range[1] ** 0.5, num_steps) ** 2
        elif method == "constant":
            betas = torch.full((num_steps,), beta_range[1])
        else:  # default to inverse scheduling
            betas = 1.0 / torch.linspace(num_steps, 1, num_steps)

        return betas