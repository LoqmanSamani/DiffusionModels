import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
from project_decoder import CLIPContextProjection
from transformers import BertTokenizer


class UnClipDecoder(nn.Module):
    """Decoder for UnCLIP diffusion models.

    Combines CLIP image embeddings and text embeddings to guide the denoising process,
    using a noise predictor and diffusion processes. Incorporates classifier-free guidance,
    text caption dropout, and projection of CLIP embeddings into context tokens.

    Parameters
    ----------
    `clip_embedding_dim` : int
        Dimensionality of the input embeddings.
    `noise_predictor` : nn.Module
        Model to predict noise during the denoising process.
    `forward_diffusion` : nn.Module
        Forward diffusion module (e.g., ForwardUnCLIP) for adding noise.
    `reverse_diffusion` : nn.Module
        Reverse diffusion module (e.g., ReverseUnCLIP) for denoising.
    `glide_text_encoder` : nn.Module, optional
        GLIDE text encoder for processing text prompts, default None.
    `bert_tokenizer` : BertTokenizer, optional
        Tokenizer for processing text prompts, default None (loads "bert-base-uncased").
    `device` : Union[str, torch.device], optional
        Device for computation (default: CUDA if available, else CPU).
    `image_output_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `normalize_clip_embeddings` : bool, optional
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
            clip_embedding_dim: int,
            noise_predictor: nn.Module,
            forward_diffusion: nn.Module,
            reverse_diffusion: nn.Module,
            glide_text_encoder: torch.nn.Module = None,  # GLIDE text encoder
            bert_tokenizer: Optional[BertTokenizer] = None,
            device: Optional[Union[str, torch.device]] = None,
            image_output_range: Tuple[float, float] = (-1.0, 1.0),
            normalize_clip_embeddings: bool = True,
            classifier_free_prop: float = 0.1,  # paper specifies 10%
            drop_caption: float = 0.5,  # paper specifies 50%
            max_token_length: int = 77  # max_token_length for tokenization
    ) -> None:
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.clip_embedding_dim = clip_embedding_dim

        # core models
        self.noise_predictor = noise_predictor.to(self.device)
        self.forward_diffusion = forward_diffusion.to(self.device)
        self.reverse_diffusion = reverse_diffusion.to(self.device)
        self.glide_text_encoder = glide_text_encoder.to(self.device) if glide_text_encoder else None

        # paper: "projecting CLIP embeddings into four extra tokens of context"
        self.clip_decoder_projection = CLIPContextProjection(
            clip_embedding_dim=self.clip_embedding_dim,
            num_tokens=4).to(self.device
        )
        self.clip_time_projection = nn.Linear(self.clip_embedding_dim, self.clip_embedding_dim).to(self.device)

        # training parameters
        self.image_output_range = image_output_range
        self.normalize_clip_embeddings = normalize_clip_embeddings
        self.classifier_free_prop = classifier_free_prop
        self.drop_caption = drop_caption
        self.max_token_length = max_token_length

        # initialize tokenizer
        if bert_tokenizer is None:
            try:
                self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                raise ValueError(f"Failed to load default tokenizer: {e}. Please provide a tokenizer.")


    def forward(
            self,
            image_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor,
            images: torch.Tensor,
            texts: torch.Tensor,
            p_classifier_free: float,
            p_text_drop: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes embeddings and images to predict noise for training.

        Applies classifier-free guidance and text dropout, projects CLIP image embeddings
        into context tokens, encodes text with GLIDE, and predicts noise for the diffusion process.

        Parameters
        ----------
        `image_embeddings` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embedding_dim).
        `text_embeddings` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embedding_dim).
        `images` : torch.Tensor
            Input images, shape (batch_size, channels, height, width).
        `texts` : torch.Tensor
            Text prompts for conditional generation.
        `p_classifier_free` : float
            Probability for applying classifier-free guidance.
        `p_text_drop` : float
            Probability for applying text caption dropout.

        Returns
        -------
        predicted_noise : torch.Tensor
            Predicted noise tensor, shape (batch_size, channels, height, width).
        noise : torch.Tensor
            Ground truth noise tensor, shape (batch_size, channels, height, width).
        """

        image_embeddings = self._apply_classifier_free_guidance(image_embeddings, p_classifier_free)
        text_embeddings = self._apply_text_dropout(text_embeddings, p_text_drop)

        # project z_i to 4 tokens
        c = self.clip_decoder_projection(image_embeddings)
        # print("z i to 4 tokens: ", c.size())

        # encode text with GLIDE
        y_encoded = self._encode_text_with_glide(texts if text_embeddings is not None else None)
        # if y_encoded is not None:
        #print("y_encodded : ", y_encoded.size())

        # concatenate embeddings
        context = self._concatenate_embeddings(y_encoded, c)
        # print("y_encodded and c concat : ", s.size())

        # sample timestep and noise
        t, noise = self._sample_timestep_and_noise(images.shape[0], images.shape)
        # print("t : ", t.size())
        # print("noise : ", noise.size())

        # compute noisy image
        noisy_images = self.forward_diffusion(images, noise, t)
        # print("noisy images : ", noisy_images.size())

        clip_image_embedding = self.clip_time_projection(image_embeddings)
        # print("clip image embedded : ", clip_image_embedding.size())

        predicted_noise = self.noise_predictor(noisy_images, t, context, clip_image_embedding)
        # print("predicted noise : ", predicted_noise.size())

        return predicted_noise, noise
    def inference_forward(self, image_embeddings, prompt_embeddings):
        pass

    def _apply_classifier_free_guidance(self, image_embeddings: torch.Tensor, p_value: float) -> torch.Tensor:
        """Applies classifier-free guidance to image embeddings.

        Sets image embeddings to zero with a specified probability to implement
        classifier-free guidance, as described in the UnCLIP paper.

        Parameters
        ----------
        `image_embeddings` : torch.Tensor
            CLIP image embeddings, shape (batch_size, embedding_dim).
        `p_value` : float
            Probability for applying classifier-free guidance.

        Returns
        -------
        image_embeddings : torch.Tensor
            Modified image embeddings, shape (batch_size, embedding_dim).
        """
        if p_value < self.classifier_free_prop:
            # set z_i ← 0 {classifier-free guidance}
            image_embeddings = torch.zeros_like(image_embeddings)

        return image_embeddings

    def _apply_text_dropout(self, text_embeddings: torch.Tensor, p_value: float) -> Optional[torch.Tensor]:
        """Applies text caption dropout to text embeddings.

        Drops text embeddings with a specified probability to implement text dropout,
        as described in the UnCLIP paper.

        Parameters
        ----------
        `text_embeddings` : torch.Tensor
            CLIP text embeddings, shape (batch_size, embedding_dim).
        `p_value` : float
            Probability for applying text caption dropout.

        Returns
        -------
        text_embeddings : torch.Tensor or None
            Modified text embeddings or None if dropped, shape (batch_size, embedding_dim).
        """
        if p_value < self.drop_caption:
            # set y ← ∅ {drop text caption}
            return None

        return text_embeddings


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
        tokenized = self.bert_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt"
        ).to(self.device)

        # get embeddings from GLIDE text encoder
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        y_encoded = self.glide_text_encoder(input_ids, attention_mask)
        print("y shape: ", y_encoded.size())

        return y_encoded

    def _concatenate_embeddings(self, y_encoded: Optional[torch.Tensor], c: torch.Tensor) -> torch.Tensor:
        """Concatenates GLIDE text embeddings and context tokens.

        Combines encoded text embeddings (if available) with projected context tokens
        along the sequence dimension, as specified in the UnCLIP paper.

        Parameters
        ----------
        `y_encoded` : torch.Tensor or None
            Encoded text embeddings from GLIDE, shape (batch_size, seq_len, embedding_dim).
        `c` : torch.Tensor
            Projected context tokens, shape (batch_size, num_tokens, embedding_dim).

        Returns
        -------
        s : torch.Tensor
            Concatenated embeddings, shape (batch_size, seq_len + num_tokens, embedding_dim).
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
        """Samples timesteps and noise for the diffusion process.

        Generates random timesteps and Gaussian noise for use in the forward diffusion process.

        Parameters
        ----------
        `batch_size` : int
            Number of samples in the batch.
        `image_shape` : torch.Size
            Shape of the images, typically (batch_size, channels, height, width).

        Returns
        -------
        t : torch.Tensor
            Sampled timestep indices, shape (batch_size,).
        noise : torch.Tensor
            Sampled Gaussian noise, shape (batch_size, channels, height, width).
        """
        # sample timestep t ~ Uniform(1, T)
        t = torch.randint(0, self.forward_diffusion.variance_scheduler.num_steps, (batch_size,), device=self.device)
        # sample noise ε ~ N(0, I)
        noise = torch.randn(image_shape, device=self.device)
        return t, noise