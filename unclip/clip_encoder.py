import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from PIL import Image
from transformers import CLIPProcessor, CLIPModel



class CLIPEncoder(nn.Module):
    """Encodes images or text using a pre-trained CLIP model.

    Loads a CLIP model and processor from the transformers library, providing methods to
    encode images or text into embeddings and compute similarity scores between them.

    Parameters
    ----------
    `model_name` : str, optional
        Name of the CLIP model to load (default: 'openai/clip-vit-base-patch32').
    `device` : str, optional
        Device to run the model on (default: 'cuda' if available, else 'cpu').
    `use_fast` : bool, optional
        Whether to use the fast image processor (torchvision-based) (default: False).
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = 'cuda',
        use_fast: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        try:
            # load clip model and processor
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=use_fast)
            self.model = self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model or processor for {self.model_name}: {e}")
        # set model to evaluation mode by default
        self.model.eval()

    def forward(
            self,
            data: Union[torch.Tensor, List[str], str, Image.Image, List[Image.Image]],
            data_type: str,
            normalize: bool = True
    ) -> torch.Tensor:
        """Encodes input data (image or text) using the CLIP model.

        Processes input data (images or text) to produce embeddings, with optional L2
        normalization.

        Parameters
        ----------
        `data` : Union[torch.Tensor, List[str], str, Image.Image, List[Image.Image]]
            Input data to encode:
                - torch.Tensor: Preprocessed image tensor (batch_size, channels, height, width).
                - List[str] or str: Text or list of texts.
                - PIL.Image.Image or List[PIL.Image.Image]: Single or list of PIL images.
        `data_type` : str
            Type of input data ('img' or 'text').
        `normalize` : bool, optional
            Whether to L2-normalize the output embeddings (default: True).

        Returns
        -------
        outputs : torch.Tensor
            Encoded embeddings, shape (batch_size, embedding_dim).
        """
        if data_type not in ["img", "text"]:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'img' or 'text'.")

        with torch.no_grad():
            if data_type == "img":
                outputs = self._encode_images(data)
            else:
                outputs = self._encode_texts(data)

            # normalize embeddings if requested
            if normalize:
                outputs = F.normalize(outputs, p=2, dim=-1)

            return outputs

    def _encode_images(self, data: Union[torch.Tensor, Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Encodes images into embeddings using the CLIP model.

        Processes image inputs (tensors or PIL images) to produce image embeddings.

        Parameters
        ----------
        `data` : Union[torch.Tensor, Image.Image, List[Image.Image]]
            Input images as a tensor or PIL image(s).

        Returns
        -------
        image_features : torch.Tensor
            Image embeddings, shape (batch_size, embedding_dim).
        """
        if isinstance(data, torch.Tensor):
            if data.dim() == 3:
                data = data.unsqueeze(0)
            inputs = {"pixel_values": data.to(self.device)}
        elif isinstance(data, (Image.Image, list)):
            if isinstance(data, Image.Image):
                data = [data]
            inputs = self.processor(images=data, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            raise ValueError(f"Invalid image data type: {type(data)}. Expected torch.Tensor, PIL.Image.Image, or List[PIL.Image.Image].")
        return self.model.get_image_features(**inputs)

    def _encode_texts(self, data: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        """Encodes texts into embeddings using the CLIP model.

        Processes text inputs (strings or tokenized tensors) to produce text embeddings.

        Parameters
        ----------
        `data` : Union[str, List[str], torch.Tensor]
            Input texts as strings or tokenized tensor.

        Returns
        -------
        text_features : torch.Tensor
            Text embeddings, shape (batch_size, embedding_dim).
        """
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
            if data.dim() == 2:
                return data
            if data.dim() == 1:
                data = data.unsqueeze(0)
            attention_mask = torch.ones_like(data)
            return self.model.get_text_features(input_ids=data, attention_mask=attention_mask)

        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list) and all(isinstance(t, str) for t in data):
            pass
        else:
            raise ValueError(
                f"Invalid text data type: {type(data)}. Expected str, List[str], or torch.Tensor."
            )

        inputs = self.processor(text=data, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.get_text_features(**inputs)

    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity between image and text embeddings.

        Calculates the cosine similarity matrix between batches of image and text embeddings.

        Parameters
        ----------
        `image_features` : torch.Tensor
            Image embeddings, shape (batch_size, embedding_dim).
        `text_features` : torch.Tensor
            Text embeddings, shape (batch_size, embedding_dim).

        Returns
        -------
        similarity : torch.Tensor
            Cosine similarity scores, shape (batch_size, batch_size).
        """
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return torch.matmul(image_features, text_features.T)