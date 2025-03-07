from config import Config
import numpy as np


def test_config():
    """test the config class to ensure correct initialization and parameter computation."""
    config = Config(method="smld", max_steps=10, sigma_min=0.1, sigma_max=1.0, beta_range=(0.01, 0.1))

    assert config.method == "smld"
    assert config.max_steps == 10
    assert config.sigma_min == 0.1
    assert config.sigma_max == 1.0
    assert config.beta_range == (0.01, 0.1)

    # check if sigmas and betas are correctly computed
    assert len(config.sigmas) == config.max_steps
    assert len(config.betas) == config.max_steps
    assert len(config.t) == config.max_steps

    # ensure sigmas follow a geometric progression
    assert np.all(np.diff(config.sigmas.numpy()) > 0), "Sigmas should be increasing"

    print("Config test passed.")

test_config()
