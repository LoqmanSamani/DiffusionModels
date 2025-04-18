import torch
import torch.nn.functional as F
import lpips
from pytorch_fid import fid_score
import shutil
from torchvision.utils import save_image
import os


class Metrics:
    """Computes image quality metrics for evaluating diffusion models.

    Supports Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), Structural
    Similarity Index (SSIM), Fréchet Inception Distance (FID), and Learned Perceptual
    Image Patch Similarity (LPIPS) for comparing generated and ground truth images.

    Parameters
    ----------
    device : str, optional
        Device for computation (e.g., 'cuda', 'cpu') (default: 'cuda').
    fid : bool, optional
        If True, compute FID score (default: True).
    metrics : bool, optional
        If True, compute MSE, PSNR, and SSIM (default: False).
    lpips : bool, optional
        If True, compute LPIPS using VGG backbone (default: False).

    Attributes
    ----------
    device : str
        Computation device.
    fid : bool
        Flag for FID computation.
    metrics : bool
        Flag for MSE, PSNR, SSIM computation.
    lpips : bool
        Flag for LPIPS computation.
    lpips_model : lpips.LPIPS or None
        LPIPS model (VGG backbone) if `lpips=True`; otherwise, None.
    temp_dir_real : str
        Temporary directory for real images during FID computation.
    temp_dir_fake : str
        Temporary directory for fake (generated) images during FID computation.
    """

    def __init__(self, device="cuda", fid=True, metrics=False, lpips=False):
        self.device = device
        self.fid = fid
        self.metrics = metrics
        self.lpips = lpips
        self.lpips_model = lpips.LPIPS(net='vgg').to(device) if lpips else None
        self.temp_dir_real = "temp_real"
        self.temp_dir_fake = "temp_fake"

    def compute_fid(self, real_images, fake_images):
        """Computes the Fréchet Inception Distance (FID) between real and generated images.

        Saves images to temporary directories and uses Inception V3 to compute FID,
        cleaning up directories afterward.

        Parameters
        ----------
        real_images : torch.Tensor
            Real images, shape (batch_size, channels, height, width), in [-1, 1].
        fake_images : torch.Tensor
            Generated images, same shape, in [-1, 1].

        Returns
        -------
        float
            FID score, or `float('inf')` if computation fails.

        Notes
        -----
        - Images are normalized to [0, 1] and saved as PNG files for FID computation.
        - Uses Inception V3 with 2048-dimensional features (`dims=2048`).
        """
        if real_images.shape != fake_images.shape:
            raise ValueError(f"Shape mismatch: real_images {real_images.shape}, fake_images {fake_images.shape}")

        real_images = (real_images + 1) / 2
        fake_images = (fake_images + 1) / 2
        real_images = real_images.clamp(0, 1).cpu()
        fake_images = fake_images.clamp(0, 1).cpu()

        os.makedirs(self.temp_dir_real, exist_ok=True)
        os.makedirs(self.temp_dir_fake, exist_ok=True)

        try:
            for i, (real, fake) in enumerate(zip(real_images, fake_images)):
                save_image(real, f"{self.temp_dir_real}/{i}.png")
                save_image(fake, f"{self.temp_dir_fake}/{i}.png")

            fid = fid_score.calculate_fid_given_paths(
                paths=[self.temp_dir_real, self.temp_dir_fake],
                batch_size=50,
                device=self.device,
                dims=2048
            )
        except Exception as e:
            print(f"Error computing FID: {e}")
            fid = float('inf')
        finally:
            shutil.rmtree(self.temp_dir_real, ignore_errors=True)
            shutil.rmtree(self.temp_dir_fake, ignore_errors=True)

        return fid

    def compute_metrics(self, x, x_hat):
        """Computes MSE, PSNR, and SSIM for evaluating image quality.

        Parameters
        ----------
        x : torch.Tensor
            Ground truth images, shape (batch_size, channels, height, width).
        x_hat : torch.Tensor
            Generated images, same shape as `x`.

        Returns
        -------
        tuple
            Tuple of (mse, psnr, ssim) as floats, where:
            - mse: Mean squared error.
            - psnr: Peak signal-to-noise ratio.
            - ssim: Structural similarity index (mean over batch).
        """
        if x.shape != x_hat.shape:
            raise ValueError(f"Shape mismatch: x {x.shape}, x_hat {x_hat.shape}")

        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse)
        c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2  # Adjusted for [-1, 1] range
        eps = 1e-8
        mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(x_hat, kernel_size=3, stride=1, padding=1)
        mu_xy = mu_x * mu_y
        sigma_x_sq = F.avg_pool2d(x.pow(2), kernel_size=3, stride=1, padding=1) - mu_x.pow(2)
        sigma_y_sq = F.avg_pool2d(x_hat.pow(2), kernel_size=3, stride=1, padding=1) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(x * x_hat, kernel_size=3, stride=1, padding=1) - mu_xy
        ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x_sq + sigma_y_sq + c2) + eps
        )

        return mse.item(), psnr.item(), ssim.mean().item()

    def compute_lpips(self, x, x_hat):
        """Computes LPIPS using a pre-trained VGG network.

        Parameters
        ----------
        x : torch.Tensor
            Ground truth images, shape (batch_size, channels, height, width), in [-1, 1].
        x_hat : torch.Tensor
            Generated images, same shape as `x`.

        Returns
        -------
        float
            Mean LPIPS score over the batch.

        Raises
        ------
        RuntimeError
            If `lpips=True` but `lpips_model` is not initialized.
        """
        if self.lpips_model is None:
            raise RuntimeError("LPIPS model not initialized; set lpips=True in __init__")
        if x.shape != x_hat.shape:
            raise ValueError(f"Shape mismatch: x {x.shape}, x_hat {x_hat.shape}")

        x = x.to(self.device)
        x_hat = x_hat.to(self.device)
        return self.lpips_model(x, x_hat).mean().item()

    def forward(self, x, x_hat):
        """Computes specified metrics for ground truth and generated images.

        Parameters
        ----------
        x : torch.Tensor
            Ground truth images, shape (batch_size, channels, height, width), in [-1, 1].
        x_hat : torch.Tensor
            Generated images, same shape as `x`.

        Returns
        -------
        tuple
            A tuple containing:
            - fid: FID score (float, or `float('inf')` if `fid=False` or fails).
            - mse: Mean squared error (float, or None if `metrics=False`).
            - psnr: Peak signal-to-noise ratio (float, or None if `metrics=False`).
            - ssim: Structural similarity index (float, or None if `metrics=False`).
            - lpips: LPIPS score (float, or None if `lpips=False`).
        """
        fid = float('inf')
        mse, psnr, ssim = None, None, None
        lpips_score = None

        if self.metrics:
            mse, psnr, ssim = self.compute_metrics(x, x_hat)
        if self.fid:
            fid = self.compute_fid(x, x_hat)
        if self.lpips:
            lpips_score = self.compute_lpips(x, x_hat)

        return fid, mse, psnr, ssim, lpips_score