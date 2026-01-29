import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchdiff.unclip import SchedulerUnCLIP, ForwardUnCLIP, ReverseUnCLIP
from torchdiff.unclip import CLIPEncoder, CLIPContextProjection, CLIPEmbeddingProjection
from torchdiff.unclip import UnCLIPTransformerPrior, TrainUnCLIPPrior
from torchdiff.unclip import UnClipDecoder, TrainUnClipDecoder
from torchdiff.unclip import UpsamplerUnCLIP, TrainUpsamplerUnCLIP
from torchdiff.unclip import SampleUnCLIP
from torchdiff.utils import DiffusionNetwork, TextEncoder, Metrics


device = 'cuda' # 'cuda'

class CIFAR10WithCaptions(Dataset):
    def __init__(self, cifar_dataset):
        self.dataset = cifar_dataset
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.caption_templates = [
            "A photo of a {}",
            "An image of a {}",
            "A picture of a {}",
            "This is a {}",
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        class_name = self.class_names[label]
        template = self.caption_templates[index % len(self.caption_templates)]
        caption = template.format(class_name)
        return image, caption

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

cifar10_train = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
cifar10_test = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

train_dataset = CIFAR10WithCaptions(cifar10_train)
test_dataset = CIFAR10WithCaptions(cifar10_test)

train_subset_indices = torch.randperm(len(train_dataset))[:10]
test_subset_indices = torch.randperm(len(test_dataset))[:2]
train_subset = Subset(train_dataset, train_subset_indices)
test_subset = Subset(test_dataset, test_subset_indices)

train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, pin_memory=True)
val_loader = DataLoader(test_subset, batch_size=1, shuffle=False, pin_memory=True)

# prior scheduler, forward and reverse modules
pvs = SchedulerUnCLIP()
pfwd = ForwardUnCLIP(pvs)
prwd = ReverseUnCLIP(pvs)

clip_encoder = CLIPEncoder(model_name="openai/clip-vit-base-patch32")

tp = CLIPEmbeddingProjection(
    clip_embed_dim=512,
    trans_embed_dim=320,
    hidden_dim=480,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
)

ip = CLIPEmbeddingProjection(
    clip_embed_dim=512,
    trans_embed_dim=320,
    hidden_dim=480,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
)

p_net = UnCLIPTransformerPrior(
    fwd_unclip=pfwd,
    rwd_unclip=prwd,
    clip_text_proj=tp,
    clip_img_proj=ip,
    trans_embed_dim=320,
    num_layers=4,
    num_att_heads=8,
    ff_dim=512,
    max_sequence_length=2,
    dropout=0.2,
    use_flash = True,
    grad_check = False,
    check_every_n_layers = 2
)

optim = torch.optim.AdamW([p for p in p_net.parameters() if p.requires_grad], lr=1e-4)
loss_fn = nn.MSELoss()

p_trainer = TrainUnCLIPPrior(
    prior_net=p_net,
    clip_net=clip_encoder,
    train_loader=train_loader,
    val_loader=val_loader,
    optim=optim,
    loss_fn=loss_fn,
    max_epochs=10,
    device=device,
    grad_acc=2,
    warmup_steps=2,
    patience=10,
    val_freq=3,
    log_freq=1,
    reduce_clip_embed_dim=True,
    trans_embed_dim=320,
    norm_clip_embed=True,

)

num_params = sum(param.numel() for param in p_trainer.prior_net.parameters())
#print(f"Number of trainable parameters in the Prior model: {num_params:,}")

#losses = p_trainer()


diff_net = DiffusionNetwork(
    in_channels = 3,  # 3 channels for RGB training data
    down_channels = [32, 64, 128],
    mid_channels = [128, 128],
    up_channels = [128, 64, 32],
    down_sampling = [True, True],
    time_embed_dim = 512,
    y_embed_dim = 512,
    num_down_blocks = 2,
    num_mid_blocks =  2,
    num_up_blocks = 2,
    down_sampling_factor = 2,
    cont_time = False, # continuous time is used for SDE based model time sampling here no need to it
    use_flash_attention = True # if true automatically uses flash attention if available
)

optim1 = torch.optim.AdamW(
    [p for p in diff_net.parameters()], lr=1e-4
)

glide = TextEncoder(
    use_pretrained_model=True,
    model_name="bert-base-uncased",
    vocabulary_size = 30522,
    num_layers = 4,
    input_dimension = 512,
    output_dimension = 512,
    num_heads = 4,
    context_length = 77,
    dropout_rate = 0.1,
    qkv_bias = False,
    scaling_value = 4,
    epsilon = 1e-5,
    use_learned_pos = False
)

decoder = UnClipDecoder(
    clip_embed_dim=512,
    diff_net=diff_net,
    fwd_unclip=pfwd,
    rwd_unclip=prwd,
    glide_text_encoder=glide,
    tokenizer=None
)

optim2 = torch.optim.AdamW([p for p in decoder.parameters() if p.requires_grad], lr=1e-5)

decoder_loss = nn.MSELoss()

metrics = Metrics(
    device=device,
    fid=False,
    metrics=False,
    lpips_=False
)

d_trainer = TrainUnClipDecoder(clip_embed_dim=512, decoder_net=decoder, clip_net=clip_encoder,
                               train_loader=train_loader, optim=optim2, loss_fn=decoder_loss, clip_text_proj=tp,
                               clip_img_proj=ip, val_loader=val_loader, metrics_=metrics, max_epochs=2, device=device,
                               store_path="unclip_decoder", patience=10, warmup_steps=2, val_freq=10, use_ddp=False,
                               grad_acc=1, log_freq=1, use_comp=False, norm_range=(-1.0, 1.0),
                               reduce_clip_embed_dim=False, trans_embed_dim=320, norm_clip_embed=True,
                               finetune_clip_proj=False, use_autocast=True)

losses_ = d_trainer()


class DummyUpsampleDataset(Dataset):
    def __init__(self, num_samples=1000, low_res_size=64, high_res_size=256):
        self.num_samples = num_samples
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        low_res_image = torch.rand(3, self.low_res_size, self.low_res_size) * 2 - 1
        high_res_image = torch.rand(3, self.high_res_size, self.high_res_size) * 2 - 1
        return low_res_image, high_res_image




fup = UpsamplerUnCLIP(
    fwd_unclip=pfwd,
    rwd_unclip=prwd,
    in_channels=3,
    out_channels=3,
    model_channels=32,
    num_res_blocks=2,
    channel_mult=(1, 2),
    dropout=0.1,
    time_embed_dim=32,
    low_res_size=64,
    high_res_size=256
)

sup = UpsamplerUnCLIP(
    fwd_unclip=pfwd,
    rwd_unclip=prwd,
    in_channels=3,
    out_channels=3,
    model_channels=32,
    num_res_blocks=2,
    channel_mult=(1, 2),
    dropout=0.1,
    time_embed_dim=32,
    low_res_size=256,
    high_res_size = 1024
)

train_dataset = DummyUpsampleDataset(num_samples=4)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

val_dataset = DummyUpsampleDataset(num_samples=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

# Define optimizer for training
optimizer = torch.optim.AdamW(fup.parameters(), lr=1e-5)

# Define the training objective (loss function)
loss_function = nn.MSELoss()


upt = TrainUpsamplerUnCLIP(
    up_net=sup,
    train_loader=train_loader,
    optim=optimizer,
    loss_fn=loss_function,
    val_loader=val_loader,
    max_epochs=10,
    device=device,
    store_path="upsampler_one",
    patience=10,
    warmup_steps=2,
    val_freq=5,
    use_ddp=False,
    grad_acc=2,
    log_freq=1,
    use_comp=False,
    norm_range=(-1.0, 1.0),
    norm_out = True,
    use_autocast=False
)
losses = upt()



sampler = SampleUnCLIP(
    prior_net = p_net,
    decoder_net = decoder,
    clip_net = clip_encoder,
    low_res_upsampler = fup,
    high_res_upsampler = sup,
    device = device,
    offload_device = 'cpu',
    clip_embed_dim = 512,
    prior_guidance_scale = 4.0,
    decoder_guidance_scale = 8.0,
    batch_size = 1,
    norm_clip_embed = True,
    prior_dim_reduction = True,
    init_img_size = (3, 64, 64),
    use_high_res_upsampler = True,
    norm_range = (-1.0, 1.0)
)

samp = sampler(prompts=["this is a test prompt"])






