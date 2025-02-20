import torch



class Config:
    """configuration class of a ddpm model"""

    def __init__(
            self,
            model_path=None, # path to store model during training and load trained model
            image_shape=None, # shape of the generated image during generation process
            model=None,  # initialized u-net model
            train_loader=None,  # initialized train loader
            optimizer=None,  # initialized train optimizer
            loss=None,  # initialized loss metric
            num_epochs=100,
            in_channels=3, # for RGB images
            learning_rate=0.1e-4,
            num_diffusion_steps=1000,
            num_time_steps=1000,
            num_steps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            device=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss = loss
        self.model_path = model_path
        self.num_epochs = num_epochs
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.num_diffusion_steps = num_diffusion_steps
        self.num_time_steps = num_time_steps
        self.num_step = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_shape = image_shape

