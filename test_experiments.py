"""
Validation script to verify that utils.py modifications don't break any experiment.
Tests imports, model instantiation, and forward passes for all 7 experiments.
"""
import torch
import sys
import traceback

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_ddpm_experiment():
    """Test DDPM experiment: imports, model creation, forward pass, sampling setup."""
    from torchdiff.ddpm import ForwardDDPM, ReverseDDPM, SchedulerDDPM, TrainDDPM, SampleDDPM
    from torchdiff.utils import DiffusionNetwork, Metrics, mse_loss

    diff_net = DiffusionNetwork(
        in_channels=1, down_channels=[32, 64, 128], mid_channels=[128, 128],
        up_channels=[128, 64, 32], down_sampling=[True, True],
        time_embed_dim=128, y_embed_dim=128,
        num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2,
        down_sampling_factor=2, cont_time=False, use_flash_attention=True
    ).to(device)

    vs = SchedulerDDPM(schedule_type="cosine")
    fwd = ForwardDDPM(vs, pred_type='noise')
    rwd = ReverseDDPM(vs, pred_type='noise')

    # Forward pass
    x = torch.randn(2, 1, 28, 28, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    out = diff_net(x, t)
    assert out.shape == x.shape, f"DDPM output shape mismatch: {out.shape} vs {x.shape}"
    assert torch.isfinite(out).all(), "DDPM output has non-finite values"

    # Forward diffusion (scheduler buffers are on CPU)
    noise = torch.randn_like(x)
    fwd_result = fwd(x.cpu(), t.cpu(), noise.cpu())
    x_t = fwd_result[0] if isinstance(fwd_result, tuple) else fwd_result
    assert x_t.shape == x.shape

    # Loss
    loss = mse_loss(out, noise.to(device))
    assert loss.dim() == 0

    # Sampler setup
    sampler = SampleDDPM(
        rwd_ddpm=rwd, diff_net=diff_net, img_size=(28, 28),
        batch_size=2, in_channels=1, device=device, norm_range=(-1, 1)
    )
    print("  DDPM: OK")


def test_ddim_experiment():
    """Test DDIM experiment: imports, model creation, forward pass."""
    from torchdiff.ddim import SchedulerDDIM, ReverseDDIM, ForwardDDIM, TrainDDIM, SampleDDIM
    from torchdiff.utils import DiffusionNetwork, Metrics, mse_loss

    diff_net = DiffusionNetwork(
        in_channels=3, down_channels=[32, 64, 128, 256], mid_channels=[256, 256, 256],
        up_channels=[256, 128, 64, 32], down_sampling=[True, True, True],
        time_embed_dim=256, y_embed_dim=256,
        num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2,
        down_sampling_factor=2, cont_time=False, use_flash_attention=True
    ).to(device)

    vs = SchedulerDDIM(schedule_type="cosine", train_steps=1000, sample_steps=500)
    fwd = ForwardDDIM(vs, pred_type='noise')
    rwd = ReverseDDIM(vs, pred_type='noise', eta=0.0)

    x = torch.randn(2, 3, 32, 32, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    out = diff_net(x, t)
    assert out.shape == x.shape, f"DDIM output shape mismatch: {out.shape} vs {x.shape}"
    assert torch.isfinite(out).all(), "DDIM output has non-finite values"

    noise = torch.randn_like(x)
    fwd_result = fwd(x.cpu(), t.cpu(), noise.cpu())
    x_t = fwd_result[0] if isinstance(fwd_result, tuple) else fwd_result
    assert x_t.shape == x.shape

    loss = mse_loss(out, noise)
    assert loss.dim() == 0

    sampler = SampleDDIM(
        rwd_ddim=rwd, diff_net=diff_net, img_size=(32, 32),
        batch_size=2, in_channels=3, device=device
    )
    print("  DDIM: OK")


def test_ldm_experiment():
    """Test LDM experiment: imports, model creation, VAE + diffusion forward passes."""
    from torchdiff.ldm import AutoencoderLDM, TrainAE, TrainLDM, SampleLDM
    from torchdiff.ddpm import SchedulerDDPM, ReverseDDPM, ForwardDDPM
    from torchdiff.utils import DiffusionNetwork, Metrics, mse_loss

    # VAE
    comp_net = AutoencoderLDM(
        in_channels=3, down_channels=[64, 128], up_channels=[128, 64],
        out_channels=3, dropout_rate=0.2, latent_channels=2,
        num_heads=4, num_groups=8, num_layers_per_block=6,
        total_down_sampling_factor=2, use_flash=True, num_embeddings=256
    ).to(device)

    x = torch.randn(2, 3, 32, 32, device=device)
    vae_out = comp_net(x)
    recon = vae_out[0] if isinstance(vae_out, (tuple, list)) else vae_out
    assert recon.shape == x.shape, f"LDM VAE output shape mismatch: {recon.shape} vs {x.shape}"

    # Latent diffusion net
    diff_net = DiffusionNetwork(
        in_channels=2, down_channels=[64, 128, 256], mid_channels=[256, 256],
        up_channels=[256, 128, 64], down_sampling=[True, True],
        time_embed_dim=256, y_embed_dim=256,
        num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2,
        down_sampling_factor=4, cont_time=False, use_flash_attention=True
    ).to(device)

    vs = SchedulerDDPM(schedule_type="cosine")
    fwd = ForwardDDPM(vs, pred_type='noise')
    rwd = ReverseDDPM(vs, pred_type='noise')

    # Get latent and run diffusion
    with torch.no_grad():
        z_result = comp_net.encode(x)
        z = z_result[0] if isinstance(z_result, tuple) else z_result
    t = torch.randint(0, 1000, (2,), device=device)
    out = diff_net(z, t)
    assert out.shape == z.shape, f"LDM diff output shape mismatch: {out.shape} vs {z.shape}"
    assert torch.isfinite(out).all(), "LDM diff output has non-finite values"
    print("  LDM: OK")


def test_vp_sde_experiment():
    """Test VP-SDE experiment."""
    from torchdiff.sde import SchedulerSDE, ForwardSDE, ReverseSDE, TrainSDE, SampleSDE
    from torchdiff.utils import TextEncoder, Metrics, snr_capped_loss, mse_loss, DiffusionNetwork

    diff_net = DiffusionNetwork(
        in_channels=1, down_channels=[32, 64, 128, 256], mid_channels=[256, 256, 256],
        up_channels=[256, 128, 64, 32], down_sampling=[True, True, True],
        time_embed_dim=256, y_embed_dim=256,
        num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2,
        down_sampling_factor=2, cont_time=True, use_flash_attention=True
    ).to(device)

    vs = SchedulerSDE("linear")
    fwd = ForwardSDE(vs, 'vp', 'noise')
    rev = ReverseSDE(vs, 'vp', 'noise')

    x = torch.randn(2, 1, 28, 28, device=device)
    t = torch.rand(2, device=device)  # Continuous time
    out = diff_net(x, t)
    assert out.shape == x.shape, f"VP-SDE output shape mismatch: {out.shape} vs {x.shape}"
    assert torch.isfinite(out).all(), "VP-SDE output has non-finite values"

    # Forward SDE
    noise = torch.randn_like(x)
    fwd_result = fwd(x, t, noise)
    x_t = fwd_result[0] if isinstance(fwd_result, tuple) else fwd_result
    assert x_t.shape == x.shape

    loss = mse_loss(out, noise)
    assert loss.dim() == 0

    sampler = SampleSDE(
        rwd_sde=rev, score_net=diff_net, img_size=(28, 28),
        batch_size=2, in_channels=1, device=device, norm_range=(-1, 1)
    )
    print("  VP-SDE: OK")


def test_ve_sde_experiment():
    """Test VE-SDE experiment."""
    from torchdiff.sde import SchedulerSDE, ForwardSDE, ReverseSDE, TrainSDE, SampleSDE
    from torchdiff.utils import mse_loss, DiffusionNetwork

    diff_net = DiffusionNetwork(
        in_channels=1, down_channels=[32, 64, 128], mid_channels=[128, 128],
        up_channels=[128, 64, 32], down_sampling=[True, True],
        time_embed_dim=128, y_embed_dim=128,
        num_down_blocks=4, num_mid_blocks=4, num_up_blocks=4,
        down_sampling_factor=4, cont_time=True, use_flash_attention=True
    ).to(device)

    vs = SchedulerSDE("cosine")
    fwd = ForwardSDE(vs, 've', 'noise')
    rev = ReverseSDE(vs, 've', 'noise')

    x = torch.randn(2, 1, 28, 28, device=device)
    t = torch.rand(2, device=device)
    out = diff_net(x, t)
    assert out.shape == x.shape, f"VE-SDE output shape mismatch: {out.shape} vs {x.shape}"
    assert torch.isfinite(out).all(), "VE-SDE output has non-finite values"

    noise = torch.randn_like(x)
    fwd_result = fwd(x, t, noise)
    x_t = fwd_result[0] if isinstance(fwd_result, tuple) else fwd_result
    assert x_t.shape == x.shape

    sampler = SampleSDE(
        rwd_sde=rev, score_net=diff_net, img_size=(28, 28),
        batch_size=2, in_channels=1, device=device, norm_range=(-1, 1)
    )
    print("  VE-SDE: OK")


def test_subvp_sde_experiment():
    """Test Sub-VP-SDE experiment."""
    from torchdiff.sde import SchedulerSDE, ForwardSDE, ReverseSDE, TrainSDE, SampleSDE
    from torchdiff.utils import snr_capped_loss, DiffusionNetwork, mse_loss

    diff_net = DiffusionNetwork(
        in_channels=1, down_channels=[32, 64, 128], mid_channels=[128, 128],
        up_channels=[128, 64, 32], down_sampling=[True, True],
        time_embed_dim=128, y_embed_dim=128,
        num_down_blocks=8, num_mid_blocks=8, num_up_blocks=8,
        down_sampling_factor=8, cont_time=True, use_flash_attention=True
    ).to(device)

    vs = SchedulerSDE("cosine")
    fwd = ForwardSDE(vs, 'sub-vp', 'noise')
    rev = ReverseSDE(vs, 'sub-vp', 'noise')

    x = torch.randn(2, 1, 28, 28, device=device)
    t = torch.rand(2, device=device)
    out = diff_net(x, t)
    assert out.shape == x.shape, f"Sub-VP-SDE output shape mismatch: {out.shape} vs {x.shape}"
    assert torch.isfinite(out).all(), "Sub-VP-SDE output has non-finite values"

    # Test snr_capped_loss specifically (we modified it)
    variance = t.clamp(min=0.01)
    loss = snr_capped_loss(out, torch.randn_like(out), variance, gamma=5.0)
    assert loss.dim() == 0 and torch.isfinite(loss), "snr_capped_loss broken"

    sampler = SampleSDE(
        rwd_sde=rev, score_net=diff_net, img_size=(28, 28),
        batch_size=2, in_channels=1, device=device, norm_range=(-1, 1)
    )
    print("  Sub-VP-SDE: OK")


def test_unclip_experiment():
    """Test UnCLIP experiment: imports, model creation, forward passes."""
    from torchdiff.unclip import SchedulerUnCLIP, ForwardUnCLIP, ReverseUnCLIP
    from torchdiff.unclip import CLIPEmbeddingProjection
    from torchdiff.unclip import UnCLIPTransformerPrior
    from torchdiff.unclip import UnClipDecoder
    from torchdiff.unclip import UpsamplerUnCLIP
    from torchdiff.utils import DiffusionNetwork, TextEncoder, Metrics

    # Prior
    prior_scheduler = SchedulerUnCLIP(
        schedule_type="cosine", train_steps=1000, sample_steps=1000,
        beta_min=1e-4, beta_max=0.02
    )
    prior_fwd = ForwardUnCLIP(prior_scheduler, pred_type="x0")
    prior_rwd = ReverseUnCLIP(prior_scheduler, pred_type="x0")

    text_projection = CLIPEmbeddingProjection(
        clip_embed_dim=512, trans_embed_dim=320, hidden_dim=480,
        num_layers=2, dropout=0.1, use_layer_norm=True
    )
    image_projection = CLIPEmbeddingProjection(
        clip_embed_dim=512, trans_embed_dim=320, hidden_dim=480,
        num_layers=2, dropout=0.1, use_layer_norm=True
    )

    prior_model = UnCLIPTransformerPrior(
        fwd_unclip=prior_fwd, rwd_unclip=prior_rwd,
        clip_text_proj=text_projection, clip_img_proj=image_projection,
        trans_embed_dim=320, num_layers=12, num_att_heads=8, ff_dim=512,
        max_sequence_length=2, dropout=0.3, use_flash=True, grad_check=False
    ).to(device)

    # Decoder with DiffusionNetwork (our modified class)
    noise_predictor = DiffusionNetwork(
        in_channels=3, down_channels=[16, 32, 64], mid_channels=[64, 64],
        up_channels=[64, 32, 16], down_sampling=[True, True],
        time_embed_dim=512, y_embed_dim=512,
        num_down_blocks=1, num_mid_blocks=1, num_up_blocks=1,
        down_sampling_factor=2
    ).to(device)

    # Forward pass through DiffusionNetwork
    x = torch.randn(2, 3, 16, 16, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    clip_emb = torch.randn(2, 512, device=device)
    out = noise_predictor(x, t, clip_embeddings=clip_emb)
    assert out.shape == x.shape, f"UnCLIP decoder output shape mismatch: {out.shape} vs {x.shape}"
    assert torch.isfinite(out).all(), "UnCLIP decoder output has non-finite values"

    # Upsampler
    up1_scheduler = SchedulerUnCLIP(
        schedule_type="cosine", train_steps=400, sample_steps=50,
        beta_min=1e-4, beta_max=0.02
    )
    up1_fwd = ForwardUnCLIP(up1_scheduler, pred_type="noise")
    up1_rwd = ReverseUnCLIP(up1_scheduler, pred_type="noise")

    upsampler_one = UpsamplerUnCLIP(
        fwd_unclip=up1_fwd, rwd_unclip=up1_rwd,
        in_channels=3, out_channels=3, model_channels=32, num_res_blocks=2,
        channel_mult=(1, 2, 4, 8), dropout=0.1, time_embed_dim=32,
        low_res_size=64, high_res_size=256
    ).to(device)

    print("  UnCLIP: OK")


def test_model_loading():
    """Test loading pre-trained model weights (verifies architecture compatibility)."""
    from torchdiff.utils import DiffusionNetwork
    import os

    base = "/home/loqman/Downloads/projs/TorchDiff/examples"

    # DDPM - load saved weights into matching architecture
    ddpm_path = os.path.join(base, "ddpm/best_model_epoch_50.pth")
    if os.path.exists(ddpm_path):
        diff_net = DiffusionNetwork(
            in_channels=1, down_channels=[32, 64, 128], mid_channels=[128, 128],
            up_channels=[128, 64, 32], down_sampling=[True, True],
            time_embed_dim=128, y_embed_dim=128,
            num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2,
            down_sampling_factor=2, cont_time=False, use_flash_attention=True
        ).to(device)
        state = torch.load(ddpm_path, map_location=device, weights_only=False)
        diff_net.load_state_dict(state['model_state_dict_diff_net'])
        x = torch.randn(1, 1, 28, 28, device=device)
        t = torch.randint(0, 1000, (1,), device=device)
        with torch.no_grad():
            out = diff_net(x, t)
        assert out.shape == x.shape
        print("  DDPM model load: OK")

    # DDIM
    ddim_path = os.path.join(base, "ddim/best_model_epoch_50.pth")
    if os.path.exists(ddim_path):
        diff_net = DiffusionNetwork(
            in_channels=3, down_channels=[32, 64, 128, 256], mid_channels=[256, 256, 256],
            up_channels=[256, 128, 64, 32], down_sampling=[True, True, True],
            time_embed_dim=256, y_embed_dim=256,
            num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2,
            down_sampling_factor=2, cont_time=False, use_flash_attention=True
        ).to(device)
        state = torch.load(ddim_path, map_location=device, weights_only=False)
        diff_net.load_state_dict(state['model_state_dict_diff_net'])
        x = torch.randn(1, 3, 32, 32, device=device)
        t = torch.randint(0, 1000, (1,), device=device)
        with torch.no_grad():
            out = diff_net(x, t)
        assert out.shape == x.shape
        print("  DDIM model load: OK")

    # LDM diffusion model
    ldm_diff_path = os.path.join(base, "ldm/diff_model.pth")
    if os.path.exists(ldm_diff_path):
        diff_net = DiffusionNetwork(
            in_channels=2, down_channels=[64, 128, 256], mid_channels=[256, 256],
            up_channels=[256, 128, 64], down_sampling=[True, True],
            time_embed_dim=256, y_embed_dim=256,
            num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2,
            down_sampling_factor=4, cont_time=False, use_flash_attention=True
        ).to(device)
        state = torch.load(ldm_diff_path, map_location=device, weights_only=False)
        diff_net.load_state_dict(state['model_state_dict_diff_net'])
        x = torch.randn(1, 2, 8, 8, device=device)
        t = torch.randint(0, 1000, (1,), device=device)
        with torch.no_grad():
            out = diff_net(x, t)
        assert out.shape == x.shape
        print("  LDM diff model load: OK")

    # VP-SDE
    vp_path = os.path.join(base, "sde/vp-sde/model_epoch_50.pth")
    if os.path.exists(vp_path):
        diff_net = DiffusionNetwork(
            in_channels=1, down_channels=[32, 64, 128, 256], mid_channels=[256, 256, 256],
            up_channels=[256, 128, 64, 32], down_sampling=[True, True, True],
            time_embed_dim=256, y_embed_dim=256,
            num_down_blocks=2, num_mid_blocks=2, num_up_blocks=2,
            down_sampling_factor=2, cont_time=True, use_flash_attention=True
        ).to(device)
        state = torch.load(vp_path, map_location=device, weights_only=False)
        key = 'model_state_dict_score_net' if 'model_state_dict_score_net' in state else 'model_state_dict_diff_net'
        diff_net.load_state_dict(state[key])
        x = torch.randn(1, 1, 28, 28, device=device)
        t = torch.rand(1, device=device)
        with torch.no_grad():
            out = diff_net(x, t)
        assert out.shape == x.shape
        print("  VP-SDE model load: OK")


if __name__ == "__main__":
    tests = [
        ("DDPM", test_ddpm_experiment),
        ("DDIM", test_ddim_experiment),
        ("LDM", test_ldm_experiment),
        ("VP-SDE", test_vp_sde_experiment),
        ("VE-SDE", test_ve_sde_experiment),
        ("Sub-VP-SDE", test_subvp_sde_experiment),
        ("UnCLIP", test_unclip_experiment),
        ("Model Loading", test_model_loading),
    ]

    passed, failed = 0, 0
    print(f"Running experiment validation on {device}...\n")
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All experiment validations passed!")
    else:
        print("Some experiments failed!")
        sys.exit(1)
