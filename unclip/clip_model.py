import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel




class CLIPEncoder(nn.Module):
    """A PyTorch module for encoding images or text using a CLIP model.

    Attributes:
        model_name (str): Name of the CLIP model (e.g., 'openai/clip-vit-base-patch32').
        model (CLIPModel): The loaded CLIP model from transformers.
        processor (CLIPProcessor): The CLIP processor for preprocessing inputs.
        device (str): The device to run the model on (e.g., 'cuda' or 'cpu').
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        use_fast: bool = False,
    ) -> None:
        """Initialize the CLIP model, processor, and optional projection layer.

        Args:
            model_name (str): Name of the CLIP model to load. Defaults to 'openai/clip-vit-base-patch32'.
            device (str, optional): Device to run the model on. If None, auto-selects 'cuda' if available, else 'cpu'.
            use_fast (bool): Whether to use the fast image processor (torchvision-based). Defaults to False.
        """
        super().__init__()

        # Set model name and device
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Load CLIP model and processor
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

        Args:
            data: Input data to encode. Can be:
                - torch.Tensor: Preprocessed image tensor (batch_size, channels, height, width).
                - List[str] or str: Text or list of texts to encode.
                - PIL.Image.Image or List[PIL.Image.Image]: Single or list of PIL images.
            data_type (str): Type of input data ('img' or 'text').
            normalize (bool): Whether to L2-normalize the output embeddings. Defaults to True.

        Returns:
            torch.Tensor: Encoded features (image or text embeddings).
                Shape: (batch_size, embedding_dim).
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
        """Helper method to encode images."""
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
            raise ValueError(
                f"Invalid image data type: {type(data)}. Expected torch.Tensor, PIL.Image.Image, or List[PIL.Image.Image]."
            )
        return self.model.get_image_features(**inputs)

    def _encode_texts(self, data: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        """Helper method to encode texts."""
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
        """Compute cosine similarity between image and text features.

        Args:
            image_features: Image embeddings (batch_size, output_dim or embedding_dim)
            text_features: Text embeddings (batch_size, output_dim or embedding_dim)

        Returns:
            torch.Tensor: Similarity scores (batch_size, batch_size)
        """
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return torch.matmul(image_features, text_features.T)



"""
# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    "Demonstrate how to use the CLIP class."

    print("=== CLIP Usage Example ===\n")

    # Initialize CLIP model
    clip_model = CLIPEncoder(model_name="openai/clip-vit-base-patch32")
    print(f"Model loaded on device: {clip_model.device}")
    print(f"Model name: {clip_model.model_name}\n")

    # ========================================================================
    # TEXT ENCODING EXAMPLES
    # ========================================================================
    print("1. TEXT ENCODING:")
    print("-" * 50)

    # Single text
    single_text = "a photo of a cat"
    text_features_single = clip_model(single_text, data_type="text")
    print(f"Single text: '{single_text}'")
    print(f"Output shape: {text_features_single.shape}")
    print(f"Output dtype: {text_features_single.dtype}")
    print(f"Output range: [{text_features_single.min():.4f}, {text_features_single.max():.4f}]\n")

    # Multiple texts
    multiple_texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a beautiful sunset over mountains",
        "a red sports car"
    ]
    text_features_multiple = clip_model(multiple_texts, data_type="text")
    print(f"Multiple texts ({len(multiple_texts)} items):")
    for i, text in enumerate(multiple_texts):
        print(f"  {i + 1}. '{text}'")
    print(f"Output shape: {text_features_multiple.shape}")
    print(f"Output dtype: {text_features_multiple.dtype}")
    print(f"Output range: [{text_features_multiple.min():.4f}, {text_features_multiple.max():.4f}]\n")

    # ========================================================================
    # IMAGE ENCODING EXAMPLES (using synthetic data for demo)
    # ========================================================================
    print("2. IMAGE ENCODING:")
    print("-" * 50)

    # Create synthetic PIL images for demonstration
    synthetic_images = []
    for i in range(3):
        # Create random RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img_array)
        synthetic_images.append(pil_img)

    # Single image
    single_image = synthetic_images[0]
    image_features_single = clip_model(single_image, data_type="img")
    print(f"Single PIL image (size: {single_image.size})")
    print(f"Output shape: {image_features_single.shape}")
    print(f"Output dtype: {image_features_single.dtype}")
    print(f"Output range: [{image_features_single.min():.4f}, {image_features_single.max():.4f}]\n")

    # Multiple images
    image_features_multiple = clip_model(synthetic_images, data_type="img")
    print(f"Multiple PIL images ({len(synthetic_images)} images)")
    print(f"Output shape: {image_features_multiple.shape}")
    print(f"Output dtype: {image_features_multiple.dtype}")
    print(f"Output range: [{image_features_multiple.min():.4f}, {image_features_multiple.max():.4f}]\n")

    # Tensor input (pre-processed)
    tensor_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    image_features_tensor = clip_model(tensor_input, data_type="img")
    print(f"Tensor input shape: {tensor_input.shape}")
    print(f"Output shape: {image_features_tensor.shape}")
    print(f"Output dtype: {image_features_tensor.dtype}\n")

    # ========================================================================
    # SIMILARITY COMPUTATION EXAMPLE
    # ========================================================================
    print("3. SIMILARITY COMPUTATION:")
    print("-" * 50)

    # Compute similarity between images and texts
    similarity_matrix = clip_model.compute_similarity(image_features_multiple, text_features_multiple)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity matrix (images vs texts):")
    print(similarity_matrix.detach().cpu().numpy())
    print()

    # Find best matches
    best_matches = similarity_matrix.argmax(dim=1)
    print("Best text matches for each image:")
    for i, match_idx in enumerate(best_matches):
        print(f"  Image {i + 1} -> Text {match_idx + 1}: '{multiple_texts[match_idx]}'")
        print(f"    Similarity score: {similarity_matrix[i, match_idx]:.4f}")
    print()

    # ========================================================================
    # EXPECTED INPUT/OUTPUT SUMMARY
    # ========================================================================
    print("4. INPUT/OUTPUT SUMMARY:")
    print("-" * 50)
    print("INPUT TYPES:")
    print("  Text:")
    print("    - str: Single text string")
    print("    - List[str]: List of text strings")
    print("  Images:")
    print("    - PIL.Image.Image: Single PIL image")
    print("    - List[PIL.Image.Image]: List of PIL images")
    print("    - torch.Tensor: Pre-processed tensor (C, H, W) or (B, C, H, W)")
    print()
    print("OUTPUT:")
    print("    - torch.Tensor: Shape (batch_size, 512) for ViT-Base models")
    print("    - dtype: torch.float32")
    print("    - Range: [-1, 1] if normalized (default), varies if not normalized")
    print("    - Device: Same as model device")


if __name__ == "__main__":
    main()
"""