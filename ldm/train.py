import torch
from tqdm import tqdm
import torch.nn.functional as F



class VLCTrain:
    def __init__(self, model, optimizer, objective, data_loader, val_loader=None, max_epoch=None,
                 device=None, save_path=None, checkpoint=None, kl_warmup_epochs=None):

        self.model = model # initialized variational latent compressor model.
        self.optimizer = optimizer # optimizer used for training.
        self.objective = objective # loss function used for training.
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.max_epoch = max_epoch or 1e4 # maximum number of epochs for training
        self.device = device or "cuda" # loss function used for training.
        self.save_path = save_path or "vlc_model.pth" # file path to save the trained model.
        self.checkpoint = checkpoint or 100 # frequency (in epochs) to save the model checkpoint.
        self.kl_warmup_epochs = kl_warmup_epochs or 10


    def compute_metrics(self, x, x_hat):

        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse)
        c1 = (0.01 * 2) ** 2
        c2 = (0.03 * 2) ** 2
        mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(x_hat, kernel_size=3, stride=1, padding=1)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        sigma_x_sq = F.avg_pool2d(x.pow(2), kernel_size=3, stride=1, padding=1) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(x_hat.pow(2), kernel_size=3, stride=1, padding=1) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * x_hat, kernel_size=3, stride=1, padding=1) - mu_xy
        ssim_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim = ssim_n / ssim_d
        ssim_mean = ssim.mean()

        return {"mse": mse.item(), "psnr": psnr.item(), "ssim": ssim_mean.item()}


    def train(self):

        self.model.train()
        self.model.to(self.device)
        train_losses = []
        best_val_loss = float("inf")

        for epoch in range(self.max_epoch):

            if self.kl_warmup_epochs > 0:
                beta = min(1.0, epoch / self.kl_warmup_epochs) * self.model.beta
            else:
                beta = self.model.beta
            self.model.current_beta = beta

            train_losses_ = []
            metrics_epoch = {"mse": [], "psnr": [], "ssim": []}

            for x, _ in tqdm(self.data_loader):

                x = x.to(self.device)
                x_hat, kl_loss = self.model(x)
                recon_loss = self.objective(x_hat, x)
                loss = recon_loss + kl_loss
                train_losses_.append(loss.item())

                with torch.no_grad():
                    batch_metrics = self.compute_metrics(x, x_hat)
                    for k, v in batch_metrics.items():
                        metrics_epoch[k].append(v)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            mean_train_loss = torch.mean(torch.tensor(train_losses_)).item()
            train_losses.append(mean_train_loss)

            metrics_summary = {k: sum(v) / len(v) for k, v in metrics_epoch.items()}
            print(f"\nEpoch: {epoch + 1} | Loss: {mean_train_loss:.4f} | "
                  f"KL Weight: {beta:.4f} | "
                  f"PSNR: {metrics_summary['psnr']:.2f} | "
                  f"SSIM: {metrics_summary['ssim']:.4f}")

            if self.val_loader is not None:
                val_loss, val_metrics = self.validate()
                print(f"Validation Loss: {val_loss:.4f} | "
                      f"Val PSNR: {val_metrics['psnr']:.2f} | "
                      f"Val SSIM: {val_metrics['ssim']:.4f}")
                current_best = val_loss
            else:
                current_best = mean_train_loss

            if epoch % self.checkpoint == 0:
                if current_best < best_val_loss:
                    best_val_loss = current_best
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': current_best,
                    }, self.save_path)
                    print(f"Model saved at epoch {epoch + 1}")

        return train_losses, best_val_loss

    def validate(self):

        self.model.eval()
        val_losses = []
        metrics_val = {"mse": [], "psnr": [], "ssim": []}

        with torch.no_grad():
            for x, _ in self.val_loader:

                x = x.to(self.device)
                x_hat, kl_loss, _ = self.model(x)
                recon_loss = self.objective(x_hat, x)
                loss = recon_loss + kl_loss
                val_losses.append(loss.item())
                batch_metrics = self.compute_metrics(x, x_hat)

                for k, v in batch_metrics.items():
                    metrics_val[k].append(v)

        self.model.train()
        mean_val_loss = torch.mean(torch.tensor(val_losses)).item()
        metrics_summary = {k: sum(v) / len(v) for k, v in metrics_val.items()}

        return mean_val_loss, metrics_summary