from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
from typing import List, Union
from PIL import Image


class CLIP(nn.Module):
    """A PyTorch module for encoding images or text using a CLIP model.

    Attributes:
        model_name (str): Name of the CLIP model to load (e.g., 'openai/clip-vit-base-patch32').
        model (CLIPModel): The loaded CLIP model from transformers.
        processor (CLIPProcessor): The CLIP processor for preprocessing inputs.
        device (str): The device to run the model on (e.g., 'cuda' or 'cpu').
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None) -> None:
        """Initialize the CLIP model and processor.

        Args:
            model_name (str): Name of the CLIP model to load. Defaults to 'openai/clip-vit-base-patch32'.
            device (str, optional): Device to run the model on. If None, auto-selects 'cuda' if available, else 'cpu'.
        """
        super().__init__()

        # Set model name
        self.model_name = model_name

        # Set device
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Load CLIP model and processor
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model or processor for {self.model_name}: {e}")

        # Set model to evaluation mode by default
        self.model.eval()

    def forward(self, data: Union[torch.Tensor, List[str], str, Image.Image, List[Image.Image]], data_type: str) -> torch.Tensor:
        """Encodes input data (image or text) using the CLIP model.

        Args:
            data: Input data to encode. Can be:
                - torch.Tensor: Preprocessed image tensor (batch_size, channels, height, width).
                - List[str] or str: Text or list of texts to encode.
                - PIL.Image.Image or List[PIL.Image.Image]: Single or list of PIL images.
            data_type (str): Type of input data ('img' or 'text').

        Returns:
            torch.Tensor: Encoded features (image or text embeddings).

        Raises:
            ValueError: If data_type is invalid or data format is incorrect.
        """
        if data_type not in ["img", "text"]:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'img' or 'text'.")

        # Move model to the correct device and ensure no gradients are computed
        self.model.to(self.device)
        with torch.no_grad():
            if data_type == "img":
                # Handle image input
                if isinstance(data, torch.Tensor):
                    # Assume tensor is already preprocessed (batch_size, channels, height, width)
                    inputs = {"pixel_values": data.to(self.device)}
                elif isinstance(data, (Image.Image, list)):
                    # Convert single PIL image to list for consistent processing
                    if isinstance(data, Image.Image):
                        data = [data]
                    # Process PIL images using the CLIP processor
                    inputs = self.processor(images=data, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    raise ValueError(
                        f"Invalid image data type: {type(data)}. Expected torch.Tensor, PIL.Image.Image, or List[PIL.Image.Image]."
                    )

                # Get image embeddings
                outputs = self.model(**inputs)
                return outputs.image_embeds

            else:  # data_type == "text"
                # Handle text input
                if isinstance(data, str):
                    # Convert single string to list for consistent processing
                    data = [data]
                elif not isinstance(data, list) or not all(isinstance(t, str) for t in data):
                    raise ValueError(
                        f"Invalid text data type: {type(data)}. Expected str or List[str]."
                    )

                # Process text using the CLIP processor
                inputs = self.processor(text=data, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get text embeddings
                outputs = self.model(**inputs)
                return outputs.text_embeds
