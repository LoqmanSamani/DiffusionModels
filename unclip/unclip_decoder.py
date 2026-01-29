import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
from project_decoder import CLIPContextProjection
from transformers import BertTokenizer
import torch.nn.functional as F





class UnClipDecoder(nn.Module):
    """Decoder for UnCLIP diffusion models.

    Combines CLIP image embeddings and text embeddings to guide the denoising process,
    using a noise predictor and diffusion processes. Incorporates classifier-free guidance,
    text caption dropout, and projection of CLIP embeddings into context tokens.

    Parameters
    ----------
    `clip_embed_dim` : int
        Dimensionality of the input embeddings.
    `diff_net` : nn.Module
        Model to predict noise/x0 during the denoising process.
    `fwd_unclip` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise.
    `rwd_unclip` : nn.Module
        Reverse diffusion module (e.g., ReverseUnCLIP) for denoising.
    `glide_text_encoder` : nn.Module, optional
        GLIDE text encoder for processing text prompts, default None.
    `tokenizer` : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    `device` : str, optional
        Device for computation (default: CUDA).
    `norm_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `norm_clip_embed` : bool, optional
        Whether to normalize outputs (default: True).
    `classifier_free_prop` : float, optional
        Probability for classifier-free guidance (default: 0.1, per paper).
    `drop_caption` : float, optional
        Probability for text caption dropout (default: 0.5, per paper).
    `max_token_length` : int, optional
        Maximum length for tokenized prompts (default: 77).
    """
    def __init__(
            self,
            clip_embed_dim: int,
            diff_net: nn.Module,
            fwd_unclip: nn.Module,
            rwd_unclip: nn.Module,
            glide_text_encoder: torch.nn.Module = None,  # GLIDE text encoder
            tokenizer: Optional[BertTokenizer] = None,
            device: str = 'cuda',
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            norm_clip_embed: bool = True,
            classifier_free_prop: float = 0.1,  # paper specifies 10%
            drop_caption: float = 0.5,  # paper specifies 50%
            max_token_length: int = 77  # max_token_length for tokenization
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.clip_embed_dim = clip_embed_dim
        # core models
        self.diff_net = diff_net.to(self.device)
        self.fwd_unclip = fwd_unclip.to(self.device)
        self.rwd_unclip = rwd_unclip.to(self.device)
        self.glide_text_encoder = glide_text_encoder.to(self.device) if glide_text_encoder else None

        # projecting CLIP embeddings into four extra tokens of context
        self.clip_decoder_proj = CLIPContextProjection(clip_embed_dim=self.clip_embed_dim, num_tokens=4).to(self.device)
        self.clip_time_proj = nn.Linear(self.clip_embed_dim, self.clip_embed_dim).to(self.device)

        # training parameters
        self.norm_range = norm_range
        self.norm_clip_embed = norm_clip_embed
        self.classifier_free_prop = classifier_free_prop
        self.drop_caption = drop_caption
        self.max_token_length = max_token_length

        # initialize tokenizer
        if tokenizer is None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")

    def forward(
            self,
            img_embed: torch.Tensor,
            text_embed: torch.Tensor,
            imgs: torch.Tensor,
            texts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes embeddings and images to predict noise for training.

        Applies classifier-free guidance and text dropout, projects CLIP image embeddings
        into context tokens, encodes text with GLIDE, and predicts noise for the diffusion process.

        Parameters
        ----------
        `img_embed` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embed_dim).
        `text_embed` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embed_dim).
        `imgs` : torch.Tensor
            Input images, shape (batch_size, channels, height, width).
        `texts` : torch.Tensor
            Text prompts for conditional generation.

        Returns
        -------
        pred : torch.Tensor
            Predicted noise/x0 tensor, shape (batch_size, channels, height, width).
        target : torch.Tensor
            Ground truth noise/x0 tensor, shape (batch_size, channels, height, width).
        """
        if self.norm_clip_embed:
            img_embed = F.normalize(img_embed, dim=-1)
        img_embed = self._classifier_free_guidance(img_embed)
        text_embed = self._text_dropout(text_embed)
        # project z_i to 4 tokens
        c = self.clip_decoder_proj(img_embed)
        # encode text with GLIDE
        y = self._encode_text_with_glide(texts if text_embed is not None else None)
        # concatenate embeddings
        context = self._conc_embed(y, c)
        # sample timestep and noise
        t, noise = self._sample_time_and_noise(imgs.shape[0], imgs.shape)
        # compute noisy image
        noisy_imgs, target = self.fwd_unclip(imgs, noise, t)
        clip_img_embed = self.clip_time_proj(img_embed)
        pred = self.diff_net(noisy_imgs, t, context, clip_img_embed)
        return pred, target

    def inference_forward(self, img_embed, prompt_embed):
        pass

    def _classifier_free_guidance(self, img_embed: torch.Tensor) -> torch.Tensor:
        """Applies classifier-free guidance to image embeddings.

        Sets image embeddings to zero with a specified probability to implement
        classifier-free guidance, as described in the UnCLIP paper.

        Parameters
        ----------
        `img_embed` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embed_dim).

        Returns
        -------
        img_embed : torch.Tensor
            Modified image embeddings, shape (batch_size, embed_dim).
        """
        batch_size = img_embed.shape[0]
        mask = torch.rand(batch_size, 1, device=self.device) < self.classifier_free_prop
        return img_embed * (~mask).float()

    def _text_dropout(self, text_embed: torch.Tensor) -> torch.Tensor:
        """Applies text caption dropout to text embeddings.

        Drops text embeddings with a specified probability to implement text dropout,
        as described in the UnCLIP paper.

        Parameters
        ----------
        `text_embed` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embed_dim).

        Returns
        -------
        text_embed : torch.Tensor or None
            Modified text embeddings or None if dropped, shape (batch_size, embed_dim).
        """
        if text_embed is None:
            return None
        batch_size = text_embed.shape[0]
        mask = torch.rand(batch_size, 1, device=self.device) < self.drop_caption
        return text_embed * (~mask).float()


    def _encode_text_with_glide(self, texts: Union[List, torch.Tensor]) -> Optional[torch.Tensor]:
        """Encodes text prompts using the GLIDE text encoder.

        Tokenizes and encodes text prompts into embeddings using the GLIDE text encoder,
        returning None if no text or conditional model is provided.

        Parameters
        ----------
        `texts` : Union[List, torch.Tensor]
            Text prompts or tensor of text data.

        Returns
        -------
        y_encoded : torch.Tensor or None
            Encoded text embeddings, shape (batch_size, seq_len, embedding_dim), or None.
        """
        if texts is None:
            return None

        if self.glide_text_encoder is None:
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
            max_length=self.max_token_length,
            return_tensors="pt"
        ).to(self.device)

        # get embeddings from GLIDE text encoder
        input_ids = tokenized["input_ids"]
        att_mask = tokenized["attention_mask"]
        y_encoded = self.glide_text_encoder(input_ids, att_mask)
        return y_encoded

    def _conc_embed(self, y: Optional[torch.Tensor], c: torch.Tensor) -> torch.Tensor:
        """Concatenates GLIDE text embeddings and context tokens.

        Combines encoded text embeddings (if available) with projected context tokens
        along the sequence dimension, as specified in the UnCLIP paper.

        Parameters
        ----------
        `y` : torch.Tensor or None
            Encoded text embeddings from GLIDE, shape (batch_size, seq_len, embed_dim).
        `c` : torch.Tensor
            Projected context tokens, shape (batch_size, num_tokens, embed_dim).

        Returns
        -------
        s : torch.Tensor
            Concatenated embeddings, shape (batch_size, seq_len + num_tokens, embed_dim).
        """
        if y is not None:
            if len(y.shape) == 2:  # [batch_size, embed_dim]
                y = y.unsqueeze(1)  # [batch_size, 1, embed_dim]
            # concatenate along the sequence dimension
            s = torch.cat([y, c], dim=1)  # [batch_size, seq_len + 4, embed_dim]
        else:
            s = c  # [batch_size, 4, embed_dim]
        return s

    def _sample_time_and_noise(self, batch_size: int, img_shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples timesteps and noise for the diffusion process.

        Generates random timesteps and Gaussian noise for use in the forward diffusion process.

        Parameters
        ----------
        `batch_size` : int
            Number of samples in the batch.
        `img_size` : torch.Size
            Shape of the images, typically (batch_size, channels, height, width).

        Returns
        -------
        t : torch.Tensor
            Sampled timestep indices, shape (batch_size,).
        noise : torch.Tensor
            Sampled Gaussian noise, shape (batch_size, channels, height, width).
        """
        # sample timestep t ~ Uniform(1, T)
        t = torch.randint(0, self.fwd_unclip.vs.train_steps, (batch_size,), device=self.device)
        # sample noise ε ~ N(0, I)
        noise = torch.randn(img_shape, device=self.device)
        return t, noise