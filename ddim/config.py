import torch
import torch.nn as nn




class Config:
    def __init__(self, model_path=None, output_shape=None, model=None, train_loader=None, optimizer=None, learning_rate=None,
                 loss_func=None, train_epochs=None, in_channels=None, beta_range=None, beta_method="linear", num_steps=None,
                 num_tau_steps=None, device=None, eta=None, image_shape=None):
        """
        configuration class for a ddim (denoising diffusion implicit model)
        stores hyperparameters and precomputes diffusion-related values for both training and generation.
        params:
            :param model_path: path to save/load model
            :param output_shape: integer (height or width of image)shape of the output image. used in generation phase.
            :param model: neural network model
            :param train_loader: data loader for training
            :param optimizer: optimizer for training
            :param loss_func: loss function for training
            :param train_epochs: number of training epochs (default: 100)
            :param in_channels: number of input channels (it defines number of image channels not batch size!)
            :param beta_range: range of beta values (default: (1e-4, 0.02))
            :param beta_method: how the beta schedule will be calculated.
            :param num_steps: number of diffusion steps in training (default: 1000)
            :param num_tau_steps: number of diffusion steps in generation (default: 100, for efficiency)
            :param device: computation device (default: cuda if available, else cpu)
        """
        self.model_path = model_path
        self.output_shape = output_shape
        self.model = model # initialized u-net
        self.train_loader = train_loader
        self.learning_rate = learning_rate or 1e-4
        self.optimizer = optimizer
        self.loss_function = loss_func or nn.MSELoss()
        self.train_epochs = train_epochs or 100
        self.in_channels = in_channels
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.beta_range = beta_range or (1e-4, 0.02)
        self.beta_method = beta_method or "constant"
        self.num_steps = num_steps or 1000  # used for forward diffusion during training
        self.num_tau_steps = num_tau_steps or 100  # used for generation (fewer steps for efficiency)
        self.eta = eta or 0 # 1 for ddpm (stochastic), 0 for fully ddim
        self.image_shape = image_shape

        # precompute diffusion-related values for both training (num_steps) and generation (num_tau_steps)
        (self.alpha, self.alpha_tau, self.alpha_sqrt, self.alpha_tau_sqrt,
         self.alpha_bar, self.alpha_tau_bar, self.alpha_bar_sqrt,
         self.alpha_tau_bar_sqrt, self.alpha_bar_sqrt_, self.alpha_tau_bar_sqrt_) = \
            self.hyper_params(self.beta_range, self.num_steps, self.num_tau_steps, self.beta_method)

    def hyper_params(self, beta_range, num_steps, num_tau_steps, method):
        """
        computes diffusion-related parameters for both training and generation.

            :param beta_range: tuple containing min and max beta values
            :param num_steps: number of diffusion steps in training
            :param num_tau_steps: number of diffusion steps in generation
            :param method: how the beta schedule will be calculated.
            :return: tuple of computed diffusion parameters
        """
        beta = self.compute_beta_schedule(beta_range, num_steps, method)  # training betas
        beta_tau = self.compute_beta_schedule(beta_range, num_tau_steps, method) # generation betas

        # compute alpha values
        alpha = 1 - beta  # for training
        alpha_tau = 1 - beta_tau  # for generation

        # square root of alpha
        alpha_sqrt = torch.sqrt(alpha)
        alpha_tau_sqrt = torch.sqrt(alpha_tau)

        # cumulative product of alpha (used to track variance reduction over steps)
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_tau_bar = torch.cumprod(alpha_tau, dim=0)

        # square root of cumulative alpha
        alpha_bar_sqrt = torch.sqrt(alpha_bar)
        alpha_tau_bar_sqrt = torch.sqrt(alpha_tau_bar)

        # complementary values (1 - alpha_bar_sqrt), useful in noise computations
        alpha_bar_sqrt_ = 1 - alpha_bar_sqrt
        alpha_tau_bar_sqrt_ = 1 - alpha_tau_bar_sqrt

        return (alpha, alpha_tau, alpha_sqrt, alpha_tau_sqrt, alpha_bar,
                alpha_tau_bar, alpha_bar_sqrt, alpha_tau_bar_sqrt,
                alpha_bar_sqrt_, alpha_tau_bar_sqrt_)

    def compute_beta_schedule(self, beta_range, num_steps, method):
        """
        computes the beta schedule based on the selected method.

            :param beta_range: tuple containing min and max beta values
            :param num_steps: number of diffusion steps
            :param method: method used to compute beta schedule (default: "linear")
            :return: tensor of computed beta values
        """
        if method == "sigmoid":
            beta = torch.linspace(-6, 6, num_steps)
            beta = torch.sigmoid(beta) * (beta_range[1] - beta_range[0]) + beta_range[0]
        elif method == "quadratic":
            beta = torch.linspace(beta_range[0]**0.5, beta_range[1]**0.5, num_steps)**2
        elif method == "constant":
            beta = torch.full((num_steps,), beta_range[1])
        elif method == "inverse_time":
            beta = 1.0 / torch.linspace(num_steps, 1, num_steps)
        else:
            beta = torch.linspace(beta_range[0], beta_range[1], num_steps)

        return beta