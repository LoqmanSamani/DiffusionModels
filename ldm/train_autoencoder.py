import torch
from tqdm import tqdm
import torch.nn.functional as F
import lpips
from pytorch_fid import fid_score
import os
import shutil
from torchvision.utils import save_image





class AETrain:
    def __init__(self, model, optimizer, data_loader, val_loader=None, max_epoch=100,
                 device="cuda", save_path="vlc_model.pth", checkpoint=10, kl_warmup_epochs=10,
                 patience=10, per_loss=True, metrics=True, fid=False, perceptual_weight=0.1):
        self.model = model # variational latent compressor model to be trained
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch
        self.device = device
        self.save_path = save_path # file path to save the trained model checkpoint
        self.checkpoint = checkpoint # frequency (in epochs) to save model checkpoints
        self.kl_warmup_epochs = kl_warmup_epochs # number of epochs for kl loss warmup
        self.patience = patience # number of epochs to wait for early stopping if validation loss doesn't improve
        self.per_loss = per_loss # if perceptual loss should be calculated
        self.metrics = metrics # if metrics should be calculated
        self.fid = fid
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(device) # lpips model for perceptual loss, using vgg backbone
        self.perceptual_weight = perceptual_weight # weight for the perceptual loss term in the total loss
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.temp_dir_real = "temp_real" # temporary directory to store real images for fid computation
        self.temp_dir_fake = "temp_fake" # temporary directory to store fake (reconstructed) images for fid computation

    def compute_metrics(self, x, x_hat):
        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse)
        c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
        mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(x_hat, kernel_size=3, stride=1, padding=1)
        mu_xy = mu_x * mu_y
        sigma_x_sq = F.avg_pool2d(x.pow(2), kernel_size=3, stride=1, padding=1) - mu_x.pow(2)
        sigma_y_sq = F.avg_pool2d(x_hat.pow(2), kernel_size=3, stride=1, padding=1) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(x * x_hat, kernel_size=3, stride=1, padding=1) - mu_xy
        ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x_sq + sigma_y_sq + c2))
        return {"mse": mse.item(), "psnr": psnr.item(), "ssim": ssim.mean().item()}


    def compute_fid(self, real_images, fake_images):
        """
        Compute FID between real and fake images.
        Args:
            real_images (torch.Tensor): Batch of real images in [-1, 1] range.
            fake_images (torch.Tensor): Batch of fake images in [-1, 1] range.
        Returns:
            float: FID score.
        """
        # ensure images are on CPU and in [0, 1] range for saving
        real_images = (real_images + 1) / 2  # [-1, 1] -> [0, 1]
        fake_images = (fake_images + 1) / 2  # [-1, 1] -> [0, 1]
        real_images = real_images.clamp(0, 1).cpu()
        fake_images = fake_images.clamp(0, 1).cpu()

        # create temporary directories
        os.makedirs(self.temp_dir_real, exist_ok=True)
        os.makedirs(self.temp_dir_fake, exist_ok=True)

        try:
            # save images to disk
            for i, (real, fake) in enumerate(zip(real_images, fake_images)):
                save_image(real, f"{self.temp_dir_real}/{i}.png")
                save_image(fake, f"{self.temp_dir_fake}/{i}.png")

            # compute FID with explicit dims=2048
            fid = fid_score.calculate_fid_given_paths(
                paths=[self.temp_dir_real, self.temp_dir_fake],
                batch_size=50,
                device=self.device,
                dims=2048  # inception V3 feature dimension
            )
        except Exception as e:
            print(f"Error computing FID: {e}")
            fid = float('inf')  # return a sentinel value on failure
        finally:
            # clean up directories
            shutil.rmtree(self.temp_dir_real, ignore_errors=True)
            shutil.rmtree(self.temp_dir_fake, ignore_errors=True)

        return fid

    def train(self):
        self.model.train()
        self.model.to(self.device)
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(self.max_epoch):
            if self.model.use_vq:
                beta = 1.0  # no warmup for vq
            else:
                beta = min(1.0, epoch / self.kl_warmup_epochs) * self.model.beta
                self.model.current_beta = beta

            train_losses_ = []
            metrics_epoch = {"mse": [], "psnr": [], "ssim": []}
            all_real, all_fake = [], []

            for x, _ in tqdm(self.data_loader):
                x = x.to(self.device)
                x_hat, total_loss, reg_loss, z = self.model(x)
                if self.per_loss:
                    percep_loss = self.perceptual_loss(x_hat, x).mean()
                    loss = total_loss + self.perceptual_weight * percep_loss
                else:
                    loss = total_loss
                train_losses_.append(loss.item())

                # only collect images and compute metrics if needed
                if self.metrics or self.fid:
                    with torch.no_grad():
                        if self.metrics:
                            batch_metrics = self.compute_metrics(x, x_hat)
                            for k, v in batch_metrics.items():
                                metrics_epoch[k].append(v)
                        if self.fid:
                            all_real.append(x)
                            all_fake.append(x_hat)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            mean_train_loss = torch.mean(torch.tensor(train_losses_)).item()
            train_losses.append(mean_train_loss)

            # compute metrics and fid if enabled
            metrics_summary = {k: sum(v) / len(v) for k, v in metrics_epoch.items()} if self.metrics else {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
            fid = self.compute_fid(torch.cat(all_real), torch.cat(all_fake)) if self.fid and all_real else float('inf')

            # print metrics conditionally
            print(f"\nEpoch: {epoch + 1} | Loss: {mean_train_loss:.4f} | Reg Weight: {beta:.4f}", end="")
            if self.metrics:
                print(f" | PSNR: {metrics_summary['psnr']:.2f} | SSIM: {metrics_summary['ssim']:.4f}", end="")
            if self.fid:
                print(f" | FID: {fid:.2f}", end="")
            print()  # newline

            if self.val_loader is not None:
                val_loss, val_metrics, val_fid = self.validate()
                # print validation metrics conditionally
                print(f"Val Loss: {val_loss:.4f}", end="")
                if self.metrics:
                    print(f" | Val PSNR: {val_metrics['psnr']:.2f} | Val SSIM: {val_metrics['ssim']:.4f}", end="")
                if self.fid:
                    print(f" | Val FID: {val_fid:.2f}", end="")
                print()
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
                print(f"Model saved at epoch {epoch + 1}")
            else:
                wait += 1
                if wait >= self.patience:
                    print("Early stopping triggered")
                    break

        return train_losses, best_val_loss

    def validate(self):
        self.model.eval()
        val_losses = []
        metrics_val = {"mse": [], "psnr": [], "ssim": []}
        all_real, all_fake = [], []

        with torch.no_grad():
            for x, _ in self.val_loader:
                x = x.to(self.device)
                x_hat, total_loss, reg_loss, z = self.model(x)
                # compute perceptual loss only if enabled
                if self.per_loss:
                    percep_loss = self.perceptual_loss(x_hat, x).mean()
                    loss = total_loss + self.perceptual_weight * percep_loss
                else:
                    loss = total_loss
                val_losses.append(loss.item())

                # only collect images and compute metrics if needed
                if self.metrics or self.fid:
                    if self.metrics:
                        batch_metrics = self.compute_metrics(x, x_hat)
                        for k, v in batch_metrics.items():
                            metrics_val[k].append(v)
                    if self.fid:
                        all_real.append(x)
                        all_fake.append(x_hat)

        # compute metrics and fid if enabled
        mean_val_loss = torch.mean(torch.tensor(val_losses)).item()
        metrics_summary = {k: sum(v) / len(v) for k, v in metrics_val.items()} if self.metrics else {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
        fid = self.compute_fid(torch.cat(all_real), torch.cat(all_fake)) if self.fid and all_real else float('inf')

        self.model.train()
        return mean_val_loss, metrics_summary, fid