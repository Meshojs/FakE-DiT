import matplotlib.pyplot as plt
import torch

from model import FakEDiT, SpaceTimeSteps

device = "cuda" if torch.cuda.is_available() else "cpu"

T = 1000
N_STEPS = 50
IMAGE_SIZE = 28
START_T = 600

model = FakEDiT(
    T=T,
    N=196,
    in_ch=1,
    d_model=256,
    n_heads=8,
    n_blocks=8,
    patch_size=2,
    dropout=0.0,
    image_size=IMAGE_SIZE,
).to(device)

model.load_state_dict(torch.load("./model/fake_dit.pt", map_location=device))
model.eval()

betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas_bar = torch.cumprod(1.0 - betas, dim=0)

steps = torch.tensor(
    SpaceTimeSteps(T, N_STEPS).forward(), device=device, dtype=torch.long
)
reverse_steps = steps.flip(0)
reverse_steps = reverse_steps[reverse_steps <= START_T]


@torch.no_grad()
def denoise(x0):
    x0 = x0.to(device)
    noise = torch.randn_like(x0)
    ab = alphas_bar[reverse_steps[0]]
    x = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
    noisy = x.clone()

    for j in range(len(reverse_steps)):
        t = reverse_steps[j].item()
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

        eps = model(x, t_batch)
        ab_t = alphas_bar[t]
        x0_pred = (x - torch.sqrt(1.0 - ab_t) * eps) / torch.sqrt(ab_t)

        if j == len(reverse_steps) - 1:
            x = x0_pred
            break

        ab_prev = alphas_bar[reverse_steps[j + 1].item()]
        x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps

    return noisy.clamp(-1, 1), x.clamp(-1, 1)


image = torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE) * 2 - 1

noisy, clean = denoise(image)

to_img = lambda x: ((x + 1.0) / 2.0).cpu()[0, 0]

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, im, title in zip(
    axes, [image, noisy, clean], ["Input", "Noisy", "Denoised"]
):
    ax.imshow(to_img(im), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("denoised3.png", dpi=150)
plt.show()
