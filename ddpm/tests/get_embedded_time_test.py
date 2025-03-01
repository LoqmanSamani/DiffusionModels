import torch
from network import GetEmbeddedTime



def test_time_embedding():
    batch_size = 4
    embedd_dim = 16

    time_emb = GetEmbeddedTime(embedd_dim)

    time_steps = torch.randint(0, 1000, (batch_size,))
    embeddings = time_emb(time_steps)

    assert embeddings.shape == (
    batch_size, embedd_dim), f"Expected shape {(batch_size, embedd_dim)}, got {embeddings.shape}"
    assert torch.all(torch.isfinite(embeddings)), "Embeddings contain NaN or Inf values"

    time_steps_2 = torch.randint(0, 1000, (batch_size,))
    embeddings_2 = time_emb(time_steps_2)
    assert not torch.allclose(embeddings, embeddings_2), "Embeddings should vary for different time steps"

    print("All tests passed successfully!")


test_time_embedding()