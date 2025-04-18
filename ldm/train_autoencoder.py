import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm





class TrainAE(nn.Module):
    """Trainer for the AutoencoderLDM variational autoencoder in Latent Diffusion Models.

    Optimizes the AutoencoderLDM model to compress images into latent space and reconstruct
    them, using reconstruction loss (MSE), regularization (KL or VQ), and optional
    perceptual loss (LPIPS). Supports mixed precision, KL warmup, early stopping, and
    learning rate scheduling, with evaluation metrics (MSE, PSNR, SSIM, FID, LPIPS).

    Parameters
    ----------
    model : AutoencoderLDM
        The variational autoencoder model (AutoencoderLDM) to train.
    optimizer : torch.optim.Optimizer
        Optimizer for training (e.g., Adam).
    data_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader, optional
        DataLoader for validation data (default: None).
    max_epoch : int, optional
        Maximum number of training epochs (default: 100).
    metrics_ : Metrics, optional
        Metrics object for computing MSE, PSNR, SSIM, FID, and LPIPS (default: None).
    device : str, optional
        Device for computation (e.g., 'cuda', 'cpu') (default: 'cuda').
    save_path : str, optional
        Path to save model checkpoints (default: 'vlc_model.pth').
    checkpoint : int, optional
        Frequency (in epochs) to save model checkpoints (default: 10).
    kl_warmup_epochs : int, optional
        Number of epochs for KL loss warmup (default: 10).
    patience : int, optional
        Number of epochs to wait for early stopping if validation loss doesn’t improve
        (default: 10).
    val_frequency : int, optional
        Frequency (in epochs) for validation and metric computation (default: 5).

    Attributes
    ----------
    device : torch.device
        Computation device.
    model : AutoencoderLDM
        Autoencoder model being trained.
    optimizer : torch.optim.Optimizer
        Training optimizer.
    data_loader : torch.utils.data.DataLoader
        Training DataLoader.
    val_loader : torch.utils.data.DataLoader or None
        Validation DataLoader.
    max_epoch : int
        Maximum training epochs.
    metrics_ : Metrics or None
        Metrics object for evaluation.
    save_path : str
        Checkpoint save path.
    checkpoint : int
        Checkpoint frequency.
    kl_warmup_epochs : int
        KL warmup epochs.
    patience : int
        Early stopping patience.
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler.
    val_frequency : int
        Validation frequency.
    """

    def __init__(self, model, optimizer, data_loader, val_loader=None, max_epoch=100, metrics_=None,
                 device="cuda", save_path="vlc_model.pth", checkpoint=10, kl_warmup_epochs=10,
                 patience=10, val_frequency=5):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.metrics_ = metrics_  # Metrics object, not moved to device
        self.save_path = save_path
        self.checkpoint = checkpoint
        self.kl_warmup_epochs = kl_warmup_epochs
        self.patience = patience
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.val_frequency = val_frequency

    def train(self):
        """Trains the AutoencoderLDM model with mixed precision and evaluation metrics.

        Performs training with reconstruction and regularization losses, KL warmup, gradient
        clipping, and learning rate scheduling. Saves checkpoints for the best validation
        loss and supports early stopping.

        Returns
        -------
        tuple
            A tuple containing:
            - train_losses: List of mean training losses per epoch.
            - best_val_loss: Best validation loss achieved (or best training loss if no validation).
        """
        scaler = GradScaler()
        self.model.train()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(self.max_epoch):
            if self.model.use_vq:
                beta = 1.0  # No warmup for VQ
            else:
                beta = min(1.0, epoch / self.kl_warmup_epochs) * self.model.beta
                self.model.current_beta = beta

            train_losses_ = []
            for x, _ in tqdm(self.data_loader):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    x_hat, loss, reg_loss, z = self.model(x)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                train_losses_.append(loss.item())

            mean_train_loss = torch.mean(torch.tensor(train_losses_)).item()
            train_losses.append(mean_train_loss)
            print(f"Epoch: {epoch + 1} | Train Loss: {mean_train_loss:.4f}", end="")

            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_loss, fid, mse, psnr, ssim, lpips_score = self.validate()
                print(f" | Val Loss: {val_loss:.4f}", end="")
                if self.metrics_ and self.metrics_.fid:
                    print(f" | FID: {fid:.4f}", end="")
                if self.metrics_ and self.metrics_.metrics:
                    print(f" | MSE: {mse:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}", end="")
                if self.metrics_ and self.metrics_.lpips:
                    print(f" | LPIPS: {lpips_score:.4f}", end="")
                print()  # Newline after metrics

                current_best = val_loss
                self.scheduler.step(val_loss)
            else:
                current_best = mean_train_loss

            if current_best < best_val_loss:
                best_val_loss = current_best
                wait = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                }, self.save_path)
                print(f" | Model saved at epoch {epoch + 1}")
            else:
                wait += 1
                if wait >= self.patience:
                    print("Early stopping triggered")
                    break

        return train_losses, best_val_loss

    def validate(self):
        """Validates the AutoencoderLDM model and computes evaluation metrics.

        Computes validation loss and optional metrics (MSE, PSNR, SSIM, FID, LPIPS) using
        the provided Metrics object.

        Returns
        -------
        tuple
            A tuple containing:
            - val_loss: Mean validation loss (float).
            - fid: Mean FID score (float, or `float('inf')` if not computed).
            - mse: Mean MSE (float, or None if not computed).
            - psnr: Mean PSNR (float, or None if not computed).
            - ssim: Mean SSIM (float, or None if not computed).
            - lpips_score: Mean LPIPS score (float, or None if not computed).
        """
        self.model.eval()
        val_losses = []
        fid_, mse_, psnr_, ssim_, lpips_score_ = [], [], [], [], []

        with torch.no_grad():
            for x, _ in self.val_loader:
                x = x.to(self.device)
                x_hat, loss, reg_loss, z = self.model(x)
                val_losses.append(loss.item())
                if self.metrics_ is not None:
                    fid, mse, psnr, ssim, lpips_score = self.metrics_.forward(x, x_hat)
                    if self.metrics_.fid:
                        fid_.append(fid)
                    if self.metrics_.metrics:
                        mse_.append(mse)
                        psnr_.append(psnr)
                        ssim_.append(ssim)
                    if self.metrics_.lpips:
                        lpips_score_.append(lpips_score)

        val_loss = torch.mean(torch.tensor(val_losses)).item()
        fid_ = torch.mean(torch.tensor(fid_)).item() if fid_ else float('inf')
        mse_ = torch.mean(torch.tensor(mse_)).item() if mse_ else None
        psnr_ = torch.mean(torch.tensor(psnr_)).item() if psnr_ else None
        ssim_ = torch.mean(torch.tensor(ssim_)).item() if ssim_ else None
        lpips_score_ = torch.mean(torch.tensor(lpips_score_)).item() if lpips_score_ else None

        self.model.train()
        return val_loss, fid_, mse_, psnr_, ssim_, lpips_score_