import unittest
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from text_conditional import TextEncoder






class DummyDataset(Dataset):
    def __init__(self, num_samples=32, max_length=77):
        self.num_samples = num_samples
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Dummy text prompts (in a real dataset, these would come from your data)
        self.text_prompts = [
            "a cat sitting on a chair",
            "a dog running in the park",
            "a sunny beach with palm trees",
            "a snowy mountain landscape"
        ] * (num_samples // 4)  # Repeat to match num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.text_prompts[idx]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)  # Shape: [max_length]
        attention_mask = encoded["attention_mask"].squeeze(0)  # Shape: [max_length]
        return input_ids, attention_mask

class TestTextEncoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.context_length = 77
        self.input_dimension = 768
        self.output_dimension = 768
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create dataset and dataloader
        self.dataset = DummyDataset(num_samples=self.batch_size, max_length=self.context_length)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize TextEncoder in both modes
        self.encoder_pretrained = TextEncoder(
            use_pretrained_model=True,
            model_name="bert-base-uncased",
            vocabulary_size=30522,
            num_layers=6,
            input_dimension=self.input_dimension,
            output_dimension=self.output_dimension,
            num_heads=8,
            context_length=self.context_length,
            dropout_rate=0.1,
            qkv_bias=False,
            scaling_value=4,
            epsilon=1e-5
        ).to(self.device)

        self.encoder_custom = TextEncoder(
            use_pretrained_model=False,
            model_name="bert-base-uncased",
            vocabulary_size=30522,
            num_layers=6,
            input_dimension=self.input_dimension,
            output_dimension=self.output_dimension,
            num_heads=8,
            context_length=self.context_length,
            dropout_rate=0.1,
            qkv_bias=False,
            scaling_value=4,
            epsilon=1e-5
        ).to(self.device)

    def test_output_shape_pretrained(self):
        """Test the output shape of the TextEncoder in pre-trained BERT mode."""
        self.encoder_pretrained.eval()
        with torch.no_grad():
            for input_ids, attention_masks in self.dataloader:
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                key_padding_mask = (attention_masks == 0)  # Invert for nn.MultiheadAttention
                output = self.encoder_pretrained(input_ids, attention_mask=key_padding_mask)
                expected_shape = (self.batch_size, self.context_length, self.output_dimension)
                self.assertEqual(output.shape, expected_shape, f"Expected output shape {expected_shape}, got {output.shape}")
                break  # Test one batch

    def test_output_shape_custom(self):
        """Test the output shape of the TextEncoder in custom transformer mode."""
        self.encoder_custom.eval()
        with torch.no_grad():
            for input_ids, attention_masks in self.dataloader:
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                key_padding_mask = (attention_masks == 0)
                output = self.encoder_custom(input_ids, attention_mask=key_padding_mask)
                expected_shape = (self.batch_size, self.context_length, self.output_dimension)
                self.assertEqual(output.shape, expected_shape, f"Expected output shape {expected_shape}, got {output.shape}")
                break

    def test_attention_mask_effect_pretrained(self):
        """Test that the attention_mask correctly masks padding tokens in pre-trained mode."""
        self.encoder_pretrained.eval()
        with torch.no_grad():
            for input_ids, attention_masks in self.dataloader:
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                print("Input IDs (pre-trained):", input_ids[0])
                print("Attention Mask (pre-trained):", attention_masks[0])

                output_with_mask = self.encoder_pretrained(input_ids, attention_mask=attention_masks)
                modified_input_ids = input_ids.clone()
                padding_mask = (attention_masks == 0)
                modified_input_ids[padding_mask] = 100
                output_with_modified = self.encoder_pretrained(modified_input_ids, attention_mask=attention_masks)

                self.assertTrue(torch.allclose(output_with_mask, output_with_modified, atol=1e-3, rtol=1e-3),
                                "Attention mask is not correctly masking padding tokens in pre-trained mode")
                break

    def test_attention_mask_effect_custom(self):
        """Test that the attention_mask correctly masks padding tokens in custom mode."""
        self.encoder_custom.eval()
        with torch.no_grad():
            for input_ids, attention_masks in self.dataloader:
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                key_padding_mask = (attention_masks == 0)

                output_with_mask = self.encoder_custom(input_ids, attention_mask=key_padding_mask)
                modified_input_ids = input_ids.clone()
                padding_mask = (attention_masks == 0)
                modified_input_ids[padding_mask] = 100
                output_with_modified = self.encoder_custom(modified_input_ids, attention_mask=key_padding_mask)

                self.assertTrue(torch.allclose(output_with_mask, output_with_modified, atol=1e-3, rtol=1e-3),
                                "Attention mask is not correctly masking padding tokens in custom mode")
                break

    def test_numerical_stability_pretrained(self):
        """Test for numerical stability in pre-trained mode (no NaNs or Infs)."""
        self.encoder_pretrained.eval()
        with torch.no_grad():
            for input_ids, attention_masks in self.dataloader:
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                key_padding_mask = (attention_masks == 0)
                output = self.encoder_pretrained(input_ids, attention_mask=key_padding_mask)
                self.assertFalse(torch.isnan(output).any(), "NaNs detected in pre-trained mode output")
                self.assertFalse(torch.isinf(output).any(), "Infs detected in pre-trained mode output")
                break

    def test_numerical_stability_custom(self):
        """Test for numerical stability in custom mode (no NaNs or Infs)."""
        self.encoder_custom.eval()
        with torch.no_grad():
            for input_ids, attention_masks in self.dataloader:
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                key_padding_mask = (attention_masks == 0)
                output = self.encoder_custom(input_ids, attention_mask=key_padding_mask)
                self.assertFalse(torch.isnan(output).any(), "NaNs detected in custom mode output")
                self.assertFalse(torch.isinf(output).any(), "Infs detected in custom mode output")
                break

    def test_gradient_flow_custom(self):
        """Test that gradients flow correctly in custom mode (trainable parameters)."""
        self.encoder_custom.train()
        for input_ids, attention_masks in self.dataloader:
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            key_padding_mask = (attention_masks == 0)

            # Forward pass
            output = self.encoder_custom(input_ids, attention_mask=key_padding_mask)
            # Dummy loss (sum of outputs)
            loss = output.sum()
            loss.backward()

            # Check that trainable parameters have gradients
            for name, param in self.encoder_custom.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, f"No gradient for parameter {name} in custom mode")
                    self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for parameter {name} in custom mode")
            break

    def test_gradient_flow_pretrained(self):
        """Test that gradients flow correctly in pre-trained mode (only projection layer)."""
        self.encoder_pretrained.train()
        for input_ids, attention_masks in self.dataloader:
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            key_padding_mask = (attention_masks == 0)

            # Forward pass
            output = self.encoder_pretrained(input_ids, attention_mask=key_padding_mask)
            # Dummy loss
            loss = output.sum()
            loss.backward()

            # Check that BERT parameters are frozen (no gradients)
            for name, param in self.encoder_pretrained.bert.named_parameters():
                self.assertIsNone(param.grad, f"Gradient found for frozen BERT parameter {name} in pre-trained mode")

            # Check that projection layer has gradients
            self.assertIsNotNone(self.encoder_pretrained.projection.weight.grad,
                                 "No gradient for projection.weight in pre-trained mode")
            self.assertIsNotNone(self.encoder_pretrained.projection.bias.grad,
                                 "No gradient for projection.bias in pre-trained mode")
            break

if __name__ == "__main__":
    unittest.main()