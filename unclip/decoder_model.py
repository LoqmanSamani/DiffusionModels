import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
from project_decoder import ProjectDecoder
from transformers import BertTokenizer


class UnClipDecoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            noise_predictor: nn.Module,
            forward_diffusion: nn.Module,
            reverse_diffusion: nn.Module,
            conditional_model: torch.nn.Module = None,  # GLIDE text encoder
            tokenizer: Optional[BertTokenizer] = None,
            device: Optional[Union[str, torch.device]] = None,
            output_range: Tuple[float, float] = (-1.0, 1.0),
            normalize: bool = True,
            classifier_free: float = 0.1,  # paper specifies 10%
            drop_caption: float = 0.5,  # paper specifies 50%
            max_length: int = 77  # max_length for tokenization
    ) -> None:
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.embedding_dim = embedding_dim

        # core models
        self.noise_predictor = noise_predictor.to(self.device)
        self.forward_diffusion = forward_diffusion.to(self.device)
        self.reverse_diffusion = reverse_diffusion.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None

        # paper: "projecting CLIP embeddings into four extra tokens of context"
        self.decoder_projection = ProjectDecoder(input_dim=self.embedding_dim, num_tokens=4).to(self.device)
        self.clip_time_proj = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)

        # training parameters
        self.output_range = output_range
        self.normalize = normalize
        self.classifier_free = classifier_free
        self.drop_caption = drop_caption
        self.max_length = max_length

        # initialize tokenizer
        if tokenizer is None:
            try:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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

        image_embeddings = self._apply_classifier_free_guidance(image_embeddings, p_classifier_free)
        text_embeddings = self._apply_text_dropout(text_embeddings, p_text_drop)

        # project z_i to 4 tokens
        c = self.decoder_projection(image_embeddings)
        # print("z i to 4 tokens: ", c.size())

        # encode text with GLIDE
        y_encoded = self._encode_text_with_glide(texts if text_embeddings is not None else None)
        # if y_encoded is not None:
        # print("y_encodded : ", y_encoded.size())

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

        clip_image_embedding = self.clip_time_proj(image_embeddings)
        # print("clip image embedded : ", clip_image_embedding.size())

        predicted_noise = self.noise_predictor(noisy_images, t, context, clip_image_embedding)
        # print("predicted noise : ", predicted_noise.size())

        return predicted_noise, noise

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
        t = torch.randint(0, self.forward_diffusion.variance_scheduler.num_steps, (batch_size,), device=self.device)
        # sample noise ε ~ N(0, I)
        noise = torch.randn(image_shape, device=self.device)
        return t, noise