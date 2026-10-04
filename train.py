import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import FakEDiT, SpaceTimeSteps

device = "cuda" if torch.cuda.is_available() else "cpu"

T = 1000
batch_size = 128
epochs = 50
lr = 2e-4
warmup = 500

tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
ds = datasets.MNIST("./data", train=True, download=False, transform=tfm)
loader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=True,
)

betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas_bar = torch.cumprod(1.0 - betas, dim=0)
t = SpaceTimeSteps(T, 50).forward()
steps = torch.tensor(t, device=device)

def add_noise(x0, t, noise):
    ab = alphas_bar[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

model = FakEDiT(
    T=1000,
    N=196,
    in_ch=1,
    d_model=256,
    n_heads=8,
    n_blocks=8,
    patch_size=2,
    dropout=0.0,
    image_size=28,
).to(device)
print(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr)

total_steps = epochs * len(loader)

def lr_lambda(s):
    if s < warmup:
        return (s + 1) / warmup
    p = (s - warmup) / (total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

os.makedirs("./model", exist_ok=True)

for epoch in range(epochs):
    model.train()
    total = torch.zeros((), device=device)
    for step, (x0, _) in enumerate(loader):
        if step % 100 == 0:
            print(f"epoch {epoch+1} step {step}/{len(loader)}")
        x0 = x0.to(device, non_blocking=True)
        t = torch.randint(0, T, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)
        xt = add_noise(x0, t, noise)

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            pred = model(xt, t)
            loss = F.mse_loss(pred.float(), noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        total += loss.detach()

    print(f"epoch {epoch+1}/{epochs}  loss {total.item()/len(loader):.4f}")
    torch.save(model.state_dict(), "./model/fake_dit.pt")

print("saved to ./model/fake_dit.pt")
