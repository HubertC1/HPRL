import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import wandb

import logging
logging.getLogger('matplotlib.font_manager').disabled = True


def analyze_z_bz(z, bz, save_path="pca_z_bz.png"):
    z_tensor = torch.tensor(z, dtype=torch.float32)
    bz_tensor = torch.tensor(bz, dtype=torch.float32)

    # Merge first two dimensions: [66, 128, 64] -> [8448, 64]
    z_flat = z_tensor.view(-1, z_tensor.size(-1))
    bz_flat = bz_tensor.view(-1, bz_tensor.size(-1))

    # Compute L2 norms
    z_norm = torch.norm(z_flat, dim=1)
    bz_norm = torch.norm(bz_flat, dim=1)

    # Cosine similarity
    cosine_sim = F.cosine_similarity(z_flat, bz_flat, dim=1)
    angles_rad = torch.acos(torch.clamp(cosine_sim, -1.0, 1.0))

    # Scale (length ratio)
    scale_ratio = z_norm / (bz_norm + 1e-8)

    # Summary statistics
    print(f"Z norm mean/std: {z_norm.mean():.4f} / {z_norm.std():.4f}")
    print(f"bZ norm mean/std: {bz_norm.mean():.4f} / {bz_norm.std():.4f}")
    print(f"Mean angle (deg): {angles_rad.mean().item() * 180 / torch.pi:.2f}")
    print(f"Mean scale ratio: {scale_ratio.mean():.4f}")

    wandb.log({
        'eval/z_norm_mean': z_norm.mean(),
        'eval/z_norm_std': z_norm.std(),
        'eval/bz_norm_mean': bz_norm.mean(),
        'eval/bz_norm_std': bz_norm.std(),
        'eval/angle_rad_mean': angles_rad.mean(),
        'eval/angle(deg)': angles_rad.mean().item() * 180 / torch.pi,
        'eval/scale_ratio_mean': scale_ratio.mean()
    })

    # Mix into a pool and do PCA
    pool = torch.cat([z_flat, bz_flat], dim=0).numpy()  # shape: (2 * 8448, 64)
    labels = np.array([0] * z_flat.size(0) + [1] * bz_flat.size(0))  # 0: z, 1: bz

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(pool)  # shape: (16896, 2)

    # Plot and save
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[labels == 0, 0], reduced[labels == 0, 1], alpha=0.5, label='z')
    plt.scatter(reduced[labels == 1, 0], reduced[labels == 1, 1], alpha=0.5, label='bz')
    plt.title("PCA projection of z and bz vectors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    return {
        'z_norm': z_norm,
        'bz_norm': bz_norm,
        'angle_rad': angles_rad,
        'scale_ratio': scale_ratio,
        'pca_components': reduced
    }
