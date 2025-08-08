import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Callable, Any
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
import os
from project_decoder import ProjectDecoder
from ddim_model import ReverseDDIM, ForwardDDIM
from transformers import BertTokenizer
import warnings



class TrainUnClipDecoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            time_embed_dim: int,
            noise_predictor: nn.Module,
            clip_model: nn.Module,
            hyper_params: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            objective: Callable,
            text_projection: Optional[nn.Module] = None,
            image_projection: Optional[nn.Module] = None,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            conditional_model: torch.nn.Module = None,  # GLIDE text encoder
            metrics_: Optional[Any] = None,
            tokenizer: Optional[BertTokenizer] = None,
            max_epoch: int = 1000,
            device: Optional[Union[str, torch.device]] = None,
            store_path: str = "unclip_decoder",
            patience: int = 100,
            warmup_epochs: int = 100,
            val_frequency: int = 10,
            use_ddp: bool = False,
            num_grad_accumulation: int = 1,
            progress_frequency: int = 1,
            compilation: bool = False,
            output_range: Tuple[float, float] = (-1.0, 1.0),
            reduce_dim: bool = True,
            output_dim: int = 319,
            normalize: bool = True,
            classifier_free: float = 0.1,  # paper specifies 10%
            drop_caption: float = 0.5,  # paper specifies 50%
            max_length: int = 77  # add max_length for tokenization
    ):
        super().__init__()
        # training configuration
        self.use_ddp = use_ddp
        self.num_grad_accumulation = num_grad_accumulation
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # setup distributed training
        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_single_gpu()

        # core models
        self.noise_predictor = noise_predictor.to(self.device)
        self.clip_model = clip_model.to(self.device)
        self.hyper_params = hyper_params.to(self.device)
        self.forward_diffusion = ForwardDDIM(hyper_params=self.hyper_params).to(self.device)
        self.reverse_diffusion = ReverseDDIM(hyper_params=self.hyper_params).to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None

        # projection models (PCA equivalent in the paper)
        self.reduce_dim = reduce_dim
        if self.reduce_dim and text_projection is not None and image_projection is not None:
            self.text_projection = text_projection.to(self.device)
            self.image_projection = image_projection.to(self.device)
        else:
            self.text_projection = None
            self.image_projection = None

        #self.embedding_dim = embedding_dim
        self.embedding_dim = output_dim if self.reduce_dim else embedding_dim
        self.time_embed_dim = time_embed_dim

        # paper: "projecting CLIP embeddings into four extra tokens of context"
        self.decoder_projection = ProjectDecoder(input_dim=self.embedding_dim, num_tokens=4).to(self.device)

        # training components
        self.metrics_ = metrics_
        self.optimizer = optimizer
        self.objective = objective
        self.train_loader = train_loader
        self.val_loader = val_loader

        # training parameters
        self.max_epoch = max_epoch
        self.patience = patience
        self.val_frequency = val_frequency
        self.progress_frequency = progress_frequency
        self.compilation = compilation
        self.output_range = output_range
        self.reduce_dim = reduce_dim
        self.normalize = normalize
        self.output_dim = output_dim
        self.classifier_free = classifier_free
        self.drop_caption = drop_caption
        self.max_length = max_length

        self.clip_time_proj = nn.Linear(self.embedding_dim, self.time_embed_dim).to(self.device)

        # checkpoint management
        self.store_path = store_path

        # learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.patience,
            factor=0.5
        )
        self.warmup_lr_scheduler = self.warmup_scheduler(self.optimizer, warmup_epochs)

        # initialize tokenizer
        if tokenizer is None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")

    def _setup_ddp(self) -> None:
        required_env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"DDP enabled but {var} environment variable not set")

        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA but CUDA is not available")

        if not torch.distributed.is_initialized():
            init_process_group(backend="nccl")

        self.ddp_rank = int(os.environ["RANK"])
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])

        self.device = torch.device(f"cuda:{self.ddp_local_rank}")
        torch.cuda.set_device(self.device)

        self.master_process = self.ddp_rank == 0

        if self.master_process:
            print(f"DDP initialized with world_size={self.ddp_world_size}")

    def _setup_single_gpu(self) -> None:
        """setup single GPU or CPU training configuration."""
        self.ddp_rank = 0
        self.ddp_local_rank = 0
        self.ddp_world_size = 1
        self.master_process = True

    @staticmethod
    def warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
        def lr_lambda(epoch):
            return min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0

        return LambdaLR(optimizer, lr_lambda)

    def _wrap_models_for_ddp(self) -> None:
        """wrap models with DistributedDataParallel for multi-GPU training."""
        if self.use_ddp:
            self.noise_predictor = DDP(
                self.noise_predictor,
                device_ids=[self.ddp_local_rank],
                find_unused_parameters=True
            )
            if self.reduce_dim and self.text_projection is not None and self.image_projection is not None:
                self.text_projection = DDP(self.text_projection, device_ids=[self.ddp_local_rank])
                self.image_projection = DDP(self.image_projection, device_ids=[self.ddp_local_rank])
            if self.conditional_model is not None:
                self.conditional_model = DDP(self.conditional_model, device_ids=[self.ddp_local_rank])

    def _compile_models(self) -> None:
        """compile models for optimization if supported."""
        if self.compilation:
            try:
                self.noise_predictor = torch.compile(self.noise_predictor)
                if self.reduce_dim and self.text_projection is not None and self.image_projection is not None:
                    self.text_projection = torch.compile(self.text_projection)
                    self.image_projection = torch.compile(self.image_projection)
                if self.conditional_model is not None:
                    self.conditional_model = torch.compile(self.conditional_model)
                if self.master_process:
                    print("Models compiled successfully")
            except Exception as e:
                if self.master_process:
                    print(f"Model compilation failed: {e}. Continuing without compilation.")

    def _get_clip_embeddings(
            self,
            images: torch.Tensor,
            texts: Union[List, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """encode text y with CLIP text encoder and image x with CLIP image encoder"""
        with torch.no_grad():
            # encode text y with CLIP text encoder: z_t ← CLIP_text(y)
            text_embeddings = self.clip_model(data=texts, data_type="text", normalize=self.normalize)
            # encode image x with CLIP image encoder: z_i ← CLIP_image(x)
            image_embeddings = self.clip_model(data=images, data_type="img", normalize=self.normalize)
        return text_embeddings, image_embeddings

    def _apply_dimensionality_reduction(
            self,
            text_embeddings: torch.Tensor,
            image_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reduce dimensionality: z_i ← P · z_i
        note: paper uses PCA algorithm, we use learned projections
        """
        with torch.no_grad(): # these models should be trained with the prior model
            if self.reduce_dim and self.text_projection is not None and self.image_projection is not None:
                text_embeddings = self.text_projection(text_embeddings)
                image_embeddings = self.image_projection(image_embeddings)
        return text_embeddings, image_embeddings

    def _apply_classifier_free_guidance(self, image_embeddings: torch.Tensor, p_value: float) -> torch.Tensor:
        """
        classifier-free guidance
        sample p ~ Uniform(0,1)
        if p < 0.1 then Set z_i ← 0 {classifier-free guidance}
        """
        if p_value < self.classifier_free:
            # set z_i ← 0 {classifier-free guidance}
            image_embeddings = torch.zeros_like(image_embeddings)

        return image_embeddings

    def _apply_text_dropout(self, text_embeddings: torch.Tensor, p_value: float) -> Optional[torch.Tensor]:
        """
        text caption dropout
        if p < 0.5 then Set y ← ∅ {drop text caption}
        """
        if p_value < self.drop_caption:
            # set y ← ∅ {drop text caption}
            return None

        return text_embeddings

    def _project_to_tokens(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        project z_i to 4 tokens: c ← Project(z_i)
        paper: "projecting CLIP embeddings into four extra tokens of context"
        """
        return self.decoder_projection(image_embeddings)

    def _encode_text_with_glide(self, texts: Union[List, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        encode text y: y_enc ← GLIDE_text(y)
        """
        if texts is None:
            return None

        if self.conditional_model is None:
            return None

        # convert to string list if needed
        if isinstance(texts, torch.Tensor):
            texts = texts.cpu().numpy().tolist()
        texts = [str(item) for item in texts]

        # tokenize
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # get embeddings from GLIDE text encoder
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        y_encoded = self.conditional_model(input_ids, attention_mask)

        return y_encoded

    def _concatenate_embeddings(self, y_encoded: Optional[torch.Tensor], c: torch.Tensor) -> torch.Tensor:
        """
        concatenate: s ← [y_enc, c]
        paper: "concatenated to the sequence of outputs from the GLIDE text encoder"
        """
        if y_encoded is not None:
            # ensure y_encoded has sequence dimension
            if len(y_encoded.shape) == 2:  # [batch_size, embed_dim]
                y_encoded = y_encoded.unsqueeze(1)  # [batch_size, 1, embed_dim]

            # concatenate along the sequence dimension
            s = torch.cat([y_encoded, c], dim=1)  # [batch_size, seq_len + 4, embed_dim]
        else:
            s = c  # [batch_size, 4, embed_dim]

        return s

    def _sample_timestep_and_noise(self, batch_size: int, image_shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample timestep t ~ Uniform(1, T)
        sample noise ε ~ N(0, I)
        """
        # sample timestep t ~ Uniform(1, T)
        t = torch.randint(0, self.hyper_params.num_steps, (batch_size,), device=self.device)
        # sample noise ε ~ N(0, I)
        noise = torch.randn(image_shape, device=self.device)
        return t, noise

    def _compute_noisy_image(self, images: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """compute noised image: x_t ← √(α_t)x + √(1-α_t)ε"""
        return self.forward_diffusion(images, noise, t)

    def _project_clip_image_embedding(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """projecting CLIP image embeddings"""
        return self.clip_time_proj(image_embeddings)

    def _predict_noise(
            self,
            noisy_images: torch.Tensor,
            t: torch.Tensor,
            clip_image_embedding: torch.Tensor,
            context: torch.Tensor
    ) -> torch.Tensor:
        """predict noise"""
        return self.noise_predictor(noisy_images, t, clip_image_embedding, context)

    def forward(self) -> Tuple[List[float], float]:

        # set models to training mode
        self.noise_predictor.train()
        #if self.reduce_dim and self.text_projection is not None and self.image_projection is not None:
        #    self.text_projection.train()
        #    self.image_projection.train()
        if self.conditional_model is not None:
            self.conditional_model.train()

        # compile and wrap models
        self._compile_models()
        self._wrap_models_for_ddp()

        # initialize training components
        scaler = torch.GradScaler()
        train_losses = []
        best_val_loss = float("inf")
        wait = 0

        # main training loop
        for epoch in range(self.max_epoch):
            # set epoch for distributed sampler if using DDP
            if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_losses_epoch = []

            # training step loop with gradient accumulation
            for step, (images, texts) in enumerate(tqdm(self.train_loader, disable=not self.master_process)):
                images = images.to(self.device, non_blocking=True)
                #print("image batch shape: ", images.size())
                #print("text batch shape: ", len(images))

                # forward pass with mixed precision
                with torch.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):

                    # encode text and image with CLIP
                    text_embeddings, image_embeddings = self._get_clip_embeddings(images, texts)
                    #print("image embedding batch shape: ", image_embeddings.size())
                    #print("text embedding batch shape: ", text_embeddings.size())

                    # reduce dimensionality (PCA equivalent)
                    text_embeddings, image_embeddings = self._apply_dimensionality_reduction(
                        text_embeddings, image_embeddings
                    )
                    #print("image embedding reduced batch shape: ", image_embeddings.size())
                    #print("text embedding reduced batch shape: ", text_embeddings.size())

                    # classifier-free guidance
                    p_classifier_free = torch.rand(1).item()
                    image_embeddings = self._apply_classifier_free_guidance(image_embeddings, p_classifier_free)

                    # text dropout
                    p_text_drop = torch.rand(1).item()
                    text_embeddings = self._apply_text_dropout(text_embeddings, p_text_drop)

                    #print("we are here")
                    # project z_i to 4 tokens
                    c = self._project_to_tokens(image_embeddings)
                    #print("z i to 4 tokens: ", c.size())


                    # encode text with GLIDE
                    y_encoded = self._encode_text_with_glide(texts if text_embeddings is not None else None)
                    #if y_encoded is not None:
                        #print("y_encodded : ", y_encoded.size())


                    # concatenate embeddings
                    s = self._concatenate_embeddings(y_encoded, c)
                    #print("y_encodded and c concat : ", s.size())

                    # sample timestep and noise
                    t, noise = self._sample_timestep_and_noise(images.shape[0], images.shape)
                    #print("t : ", t.size())
                    #print("noise : ", noise.size())

                    # compute noisy image
                    noisy_images = self._compute_noisy_image(images, noise, t)
                    #print("noisy images : ", noisy_images.size())

                    clip_image_embedding = self._project_clip_image_embedding(image_embeddings)
                    #print("clip image embedded : ", clip_image_embedding.size())

                    # predict noise
                    predicted_noise = self._predict_noise(noisy_images, t, clip_image_embedding, s)
                    #print("predicted noise : ", predicted_noise.size())

                    # compute loss
                    loss = self.objective(predicted_noise, noise) / self.num_grad_accumulation

                scaler.scale(loss).backward()

                if (step + 1) % self.num_grad_accumulation == 0:
                    # clip gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.noise_predictor.parameters(), max_norm=1.0)
                    if self.conditional_model is not None:
                        torch.nn.utils.clip_grad_norm_(self.conditional_model.parameters(), max_norm=1.0)
                    if self.text_projection is not None:
                        torch.nn.utils.clip_grad_norm_(self.text_projection.parameters(), max_norm=1.0)
                    if self.image_projection is not None:
                        torch.nn.utils.clip_grad_norm_(self.image_projection.parameters(), max_norm=1.0)

                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    self.warmup_lr_scheduler.step()
                    torch.cuda.empty_cache()  # clear memory after optimizer step

                train_losses_epoch.append(loss.item() * self.num_grad_accumulation)

            mean_train_loss = self._compute_mean_loss(train_losses_epoch)
            train_losses.append(mean_train_loss)

            if self.master_process and (epoch + 1) % self.progress_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.max_epoch} | LR: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f}", end="")

            current_loss = mean_train_loss


            if self.val_loader is not None and (epoch + 1) % self.val_frequency == 0:
                val_metrics = self.validate()
                val_loss, fid, mse, psnr, ssim, lpips_score = val_metrics

                if self.master_process:
                    print(f" | Val Loss: {val_loss:.4f}", end="")
                    if self.metrics_ and hasattr(self.metrics_, 'fid') and self.metrics_.fid:
                        print(f" | FID: {fid:.4f}", end="")
                    if self.metrics_ and hasattr(self.metrics_, 'metrics') and self.metrics_.metrics:
                        print(f" | MSE: {mse:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}", end="")
                    if self.metrics_ and hasattr(self.metrics_, 'lpips') and self.metrics_.lpips:
                        print(f" | LPIPS: {lpips_score:.4f}", end="")
                    print()

            self.scheduler.step(current_loss)

            if self.master_process:
                if current_loss < best_val_loss and (epoch + 1) % self.val_frequency == 0:
                    best_val_loss = current_loss
                    wait = 0
                    self._save_checkpoint(epoch + 1, best_val_loss, is_best=True)
                else:
                    wait += 1
                    if wait >= self.patience:
                        print("Early stopping triggered")
                        self._save_checkpoint(epoch + 1, current_loss, suffix="_early_stop")
                        break

        if self.use_ddp:
            destroy_process_group()

        return train_losses, best_val_loss

    def _compute_mean_loss(self, losses: List[float]) -> float:
        """compute mean loss with DDP synchronization if needed."""
        if not losses:
            return 0.0
        mean_loss = sum(losses) / len(losses)
        if self.use_ddp:
            # synchronize loss across all processes
            loss_tensor = torch.tensor(mean_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            mean_loss = (loss_tensor / self.ddp_world_size).item()

        return mean_loss

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False, suffix: str = ""):
        """Save model checkpoint."""
        if not self.master_process:
            return
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            # core models
            'noise_predictor_state_dict': self.noise_predictor.module.state_dict() if self.use_ddp else self.noise_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),

            # training configuration
            'embedding_dim': self.embedding_dim,
            'time_embed_dim': self.time_embed_dim,
            'output_dim': self.output_dim,
            'reduce_dim': self.reduce_dim,
            'normalize': self.normalize,
            'classifier_free': self.classifier_free,
            'drop_caption': self.drop_caption,
            'max_length': self.max_length,
        }

        # trainable components of the trainer itself
        # save CLIP time projection layer
        checkpoint['clip_time_proj_state_dict'] = self.clip_time_proj.state_dict()

        # save decoder projection layer
        checkpoint['decoder_projection_state_dict'] = self.decoder_projection.state_dict()

        # save hyperparams model if it has trainable parameters
        if hasattr(self.hyper_params, 'state_dict'):
            checkpoint['hyper_params_state_dict'] = self.hyper_params.state_dict()

        # save projection models (PCA equivalent)
        if self.reduce_dim and self.text_projection and self.image_projection:
            checkpoint[
                'text_projection_state_dict'] = self.text_projection.module.state_dict() if self.use_ddp else self.text_projection.state_dict()
            checkpoint[
                'image_projection_state_dict'] = self.image_projection.module.state_dict() if self.use_ddp else self.image_projection.state_dict()

        # save conditional model (GLIDE text encoder)
        if self.conditional_model:
            checkpoint[
                'conditional_model_state_dict'] = self.conditional_model.module.state_dict() if self.use_ddp else self.conditional_model.state_dict()

        # save schedulers state
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        checkpoint['warmup_scheduler_state_dict'] = self.warmup_lr_scheduler.state_dict()

        filename = f"unclip_decoder_epoch_{epoch}{suffix}.pth"
        if is_best:
            filename = f"unclip_decoder_best{suffix}.pth"

        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)

        if is_best:
            print(f"Best model saved: {filepath}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        def _load_model_state_dict(model: nn.Module, state_dict: dict, model_name: str) -> None:
            """Helper function to load state dict with DDP compatibility."""
            try:
                # handle DDP state dict compatibility
                if self.use_ddp and not any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                elif not self.use_ddp and any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                model.load_state_dict(state_dict)
                if self.master_process:
                    print(f"✓ Loaded {model_name}")
            except Exception as e:
                warnings.warn(f"Failed to load {model_name}: {e}")

        # load core noise predictor model
        if 'noise_predictor_state_dict' in checkpoint:
            _load_model_state_dict(self.noise_predictor, checkpoint['noise_predictor_state_dict'], 'noise_predictor')

        # load trainer's own trainable components
        if 'clip_time_proj_state_dict' in checkpoint:
            try:
                self.clip_time_proj.load_state_dict(checkpoint['clip_time_proj_state_dict'])
                if self.master_process:
                    print("✓ Loaded CLIP time projection layer")
            except Exception as e:
                warnings.warn(f"Failed to load CLIP time projection: {e}")

        if 'decoder_projection_state_dict' in checkpoint:
            try:
                self.decoder_projection.load_state_dict(checkpoint['decoder_projection_state_dict'])
                if self.master_process:
                    print("✓ Loaded decoder projection layer")
            except Exception as e:
                warnings.warn(f"Failed to load decoder projection: {e}")

        # load hyperparams model (if it has trainable parameters)
        if 'hyper_params_state_dict' in checkpoint:
            try:
                if hasattr(self.hyper_params, 'load_state_dict'):
                    self.hyper_params.load_state_dict(checkpoint['hyper_params_state_dict'])
                    if self.master_process:
                        print("✓ Loaded hyperparams model")
                else:
                    warnings.warn("Hyperparams model doesn't support state_dict loading")
            except Exception as e:
                warnings.warn(f"Failed to load hyperparams model: {e}")

        # load projection models (PCA equivalent)
        if self.reduce_dim and self.text_projection and self.image_projection:
            if 'text_projection_state_dict' in checkpoint:
                _load_model_state_dict(self.text_projection, checkpoint['text_projection_state_dict'],
                                       'text_projection')

            if 'image_projection_state_dict' in checkpoint:
                _load_model_state_dict(self.image_projection, checkpoint['image_projection_state_dict'],
                                       'image_projection')

        # load conditional model (GLIDE text encoder)
        if self.conditional_model and 'conditional_model_state_dict' in checkpoint:
            _load_model_state_dict(self.conditional_model, checkpoint['conditional_model_state_dict'],
                                   'conditional_model')

        # load optimizer
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.master_process:
                    print("✓ Loaded optimizer")
            except Exception as e:
                warnings.warn(f"Failed to load optimizer state: {e}")

        # load schedulers
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if self.master_process:
                    print("✓ Loaded main scheduler")
            except Exception as e:
                warnings.warn(f"Failed to load scheduler state: {e}")

        if 'warmup_scheduler_state_dict' in checkpoint:
            try:
                self.warmup_lr_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
                if self.master_process:
                    print("✓ Loaded warmup scheduler")
            except Exception as e:
                warnings.warn(f"Failed to load warmup scheduler state: {e}")

        # verify configuration compatibility
        if 'embedding_dim' in checkpoint:
            if checkpoint['embedding_dim'] != self.embedding_dim:
                warnings.warn(
                    f"Embedding dimension mismatch: checkpoint={checkpoint['embedding_dim']}, current={self.embedding_dim}")

        if 'reduce_dim' in checkpoint:
            if checkpoint['reduce_dim'] != self.reduce_dim:
                warnings.warn(
                    f"Reduce dimension setting mismatch: checkpoint={checkpoint['reduce_dim']}, current={self.reduce_dim}")

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

        if self.master_process:
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        return epoch, loss


    def validate(self) -> Tuple[float, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:

        self.noise_predictor.eval()
        if self.reduce_dim and self.text_projection is not None:
            self.text_projection.eval()
        if self.reduce_dim and self.image_projection is not None:
            self.image_projection.eval()
        if self.conditional_model is not None:
            self.conditional_model.eval()

        val_losses = []
        fid_scores, mse_scores, psnr_scores, ssim_scores, lpips_scores = [], [], [], [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                if isinstance(y, torch.Tensor):
                    y = y.to(self.device)
                x_orig = x.clone()
                text_embeddings, image_embeddings = self._get_clip_embeddings(x, y)
                text_embeddings, image_embeddings = self._apply_dimensionality_reduction(
                    text_embeddings, image_embeddings
                )
                p_classifier_free = torch.rand(1).item()
                image_embeddings = self._apply_classifier_free_guidance(image_embeddings, p_classifier_free)
                p_text_drop = torch.rand(1).item()
                text_embeddings = self._apply_text_dropout(text_embeddings, p_text_drop)
                c = self._project_to_tokens(image_embeddings)
                y_encoded = self._encode_text_with_glide(y if text_embeddings is not None else None)
                s = self._concatenate_embeddings(y_encoded, c)
                t, noise = self._sample_timestep_and_noise(x.shape[0], x.shape)
                noisy_images = self._compute_noisy_image(x, noise, t)
                clip_image_embedding = self._project_clip_image_embedding(image_embeddings)
                predicted_noise = self._predict_noise(noisy_images, t, clip_image_embedding, s)
                loss = self.objective(predicted_noise, noise)
                val_losses.append(loss.item())

                if self.metrics_ is not None and self.reverse_diffusion is not None:
                    xt = torch.randn_like(x).to(self.device)

                    for t in reversed(range(self.hyper_params.tau_num_steps)):
                        time_steps = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
                        prev_time_steps = torch.full((xt.shape[0],), max(t - 1, 0), device=self.device,
                                                     dtype=torch.long)
                        predicted_noise = self.noise_predictor(xt, time_steps, clip_image_embedding, s)
                        xt, _ = self.reverse_diffusion(xt, predicted_noise, time_steps, prev_time_steps)

                    x_hat = torch.clamp(xt, min=self.output_range[0], max=self.output_range[1])

                    if self.normalize:
                        x_hat = (x_hat - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
                        x_orig = (x_orig - self.output_range[0]) / (self.output_range[1] - self.output_range[0])

                    metrics_result = self.metrics_.forward(x_orig, x_hat)
                    fid = metrics_result[0] if getattr(self.metrics_, 'fid', False) else float('inf')
                    mse = metrics_result[1] if getattr(self.metrics_, 'metrics', False) else None
                    psnr = metrics_result[2] if getattr(self.metrics_, 'metrics', False) else None
                    ssim = metrics_result[3] if getattr(self.metrics_, 'metrics', False) else None
                    lpips_score = metrics_result[4] if getattr(self.metrics_, 'lpips', False) else None

                    if fid:
                        fid_scores.append(fid)
                    if mse:
                        mse_scores.append(mse)
                    if psnr:
                        psnr_scores.append(psnr)
                    if ssim:
                        ssim_scores.append(ssim)
                    if lpips_score:
                        lpips_scores.append(lpips_score)

        val_loss = torch.tensor(val_losses).mean().item()

        if self.use_ddp:
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()


        fid_avg = torch.tensor(fid_scores).mean().item() if fid_scores else float('inf')
        mse_avg = torch.tensor(mse_scores).mean().item() if mse_scores else None
        psnr_avg = torch.tensor(psnr_scores).mean().item() if psnr_scores else None
        ssim_avg = torch.tensor(ssim_scores).mean().item() if ssim_scores else None
        lpips_avg = torch.tensor(lpips_scores).mean().item() if lpips_scores else None

        # return to training mode
        self.noise_predictor.train()
        if self.reduce_dim and self.text_projection is not None:
            self.text_projection.train()
        if self.reduce_dim and self.image_projection is not None:
            self.image_projection.train()
        if self.conditional_model is not None:
            self.conditional_model.train()

        return val_loss, fid_avg, mse_avg, psnr_avg, ssim_avg, lpips_avg

from utils import NoisePredictor, TextEncoder, Metrics
from clip_model import CLIPEncoder
from project_decoder import ProjectDecoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from project_prior import Projection
import torch
from ddim_model import HyperParamsDDIM


class CIFAR10WithCaptions(Dataset):
    def __init__(self, cifar_dataset):
        self.dataset = cifar_dataset
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        # More descriptive templates
        self.templates = [
            "A photo of a {}",
            "An image of a {}",
            "A picture of a {}",
            "This is a {}",
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        class_name = self.class_names[label]
        # Use different templates for variety
        template = self.templates[idx % len(self.templates)]
        caption = template.format(class_name)
        return image, caption


# Updated transforms for CLIP
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 with captions
cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_dataset = CIFAR10WithCaptions(cifar_train)
test_dataset = CIFAR10WithCaptions(cifar_test)

# Small subset for testing
train_subset_indices = torch.randperm(len(train_dataset))[:10]
test_subset_indices = torch.randperm(len(test_dataset))[:5]
train_subset = Subset(train_dataset, train_subset_indices)
test_subset = Subset(test_dataset, test_subset_indices)

# DataLoaders
t_loader = DataLoader(train_subset, batch_size=5, shuffle=True, pin_memory=True)
v_loader = DataLoader(test_subset, batch_size=2, shuffle=False, pin_memory=True)

d = torch.device("cuda")

n_model = NoisePredictor(
        in_channels=3,
        down_channels=[16, 32],
        mid_channels=[32, 32],
        up_channels=[32, 16],
        down_sampling=[True, True],
        time_embed_dim=320,
        y_embed_dim=320,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        down_sampling_factor=2
).to(d)

c_model = CLIPEncoder(model_name="openai/clip-vit-base-patch32").to(d)

t_proj = Projection(
    input_dim=512,
    output_dim=320,
    hidden_dim=468,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
).to(d)
i_proj = Projection(
    input_dim=512,
    output_dim=320,
    hidden_dim=468,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
).to(d)

h_model = HyperParamsDDIM(
    num_steps=500,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=False,
    beta_method="linear"
).to(d)

cond = TextEncoder(
    use_pretrained_model=True,
    model_name="bert-base-uncased",
    vocabulary_size=30522,
    num_layers=2,
    input_dimension=320,
    output_dimension=320,
    num_heads=2,
    context_length=77
).to(d)

opt = torch.optim.AdamW(
    [p for p in h_model.parameters() if p.requires_grad] +
    [p for p in n_model.parameters() if p.requires_grad] +
    [p for p in cond.parameters() if p.requires_grad], lr=1e-3)

obj = nn.MSELoss()

mets = Metrics(
    device="cpu",
    fid=True,
    metrics=True,
    lpips_=True
)




model = TrainUnClipDecoder(
    embedding_dim=512,
    time_embed_dim=320,
    noise_predictor=n_model,
    clip_model=c_model,
    hyper_params=h_model,
    train_loader=t_loader,
    optimizer=opt,
    objective=obj,
    text_projection=t_proj,
    image_projection=i_proj,
    val_loader=v_loader,
    conditional_model=cond,
    metrics_=mets,
    tokenizer=None,
    max_epoch=5,
    device="cuda",
    store_path="unclip_decoder",
    patience=5,
    warmup_epochs=2,
    val_frequency=3,
    use_ddp=False,
    num_grad_accumulation=2,
    progress_frequency=1,
    compilation=False,
    output_range=(-1.0, 1.0),
    reduce_dim=True,
    output_dim=320,
    normalize=True,
    classifier_free=0.1,
    drop_caption=0.5,
    max_length=77
)

one, two = model()


models = [h_model, n_model, cond, t_proj, i_proj, h_model]

total_params = 0
for model in models:
    total_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_params)