import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_features(images, vgg16):
    # downsample images and return vgg features
    images = (images + 1) * (255 / 2)
    images = F.interpolate(images, size=(224, 224), mode='area')
    features = vgg16(images, resize_images=False, return_lpips=True)
    return features


@torch.no_grad()
def generate_images(G, w):
    # pass noise through generator and convert to numpy
    synth_images = G.synthesis(w, noise_mode='const')
    synth_images = (synth_images + 1) * (255 / 2)
    synth_images = synth_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return synth_images


def find_noise(G, vgg16, target_image, device, num_steps=100, init_lr=0.1,
               num_w_samples=100, std_factor=0.05, l2_lambda=0.1,
               noise_ramp=0.75, lr_rampdown=0.25, lr_rampup=0.05):
    # estimate statistics for w latent
    z_samples = np.random.randn(num_w_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / num_w_samples) ** 0.5

    # generate features of target image
    target_image = target_image.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_image = np.array(target_image, dtype=np.uint8)
    target_image = torch.tensor(target_image, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    target_image = target_image / (255 / 2) - 1
    target_features = get_features(target_image, vgg16)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt], lr=init_lr)
    l2_criterion = torch.nn.MSELoss()

    for step in tqdm(range(num_steps)):
        t = step / num_steps
        w_noise_scale = w_std * std_factor * max(0.0, 1.0 - t / noise_ramp) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup)
        lr = init_lr * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_image = G.synthesis(ws, noise_mode='const')

        synth_features = get_features(synth_image, vgg16)
        perceptual_loss = (target_features - synth_features).square().sum()
        l2_loss = l2_criterion(target_image, synth_image)
        loss = perceptual_loss + l2_lambda * l2_loss

        # optimizer step
        loss.backward()
        optimizer.step()

        # save optimization history
        w_out[step] = w_opt.detach()[0]

    return w_out.repeat([1, G.mapping.num_ws, 1])
