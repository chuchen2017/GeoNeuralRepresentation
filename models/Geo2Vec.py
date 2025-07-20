import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class Geo2Vec_Model(torch.nn.Module):
    def __init__(self, n_poly, z_size=256, hidden_size=256, num_freqs=10,negative_slope=0.01,
                 weight_decay=0.01, log_sampling=False, input_dims=2, polar_fourier=True, num_layers=8):
        super().__init__()
        self.z_size = z_size
        self.input_dims = input_dims
        self.num_layers = num_layers

        self.poly_embedding_layer = torch.nn.Embedding(n_poly, z_size)
        initiation_weight = torch.nn.Parameter(torch.randn_like(self.poly_embedding_layer.weight) * weight_decay)
        self.poly_embedding_layer.weight = initiation_weight
        self.poly_embedding_layer.weight.requires_grad = True

        self.location_encoder = PositionalEncoder(
            input_dims=input_dims,
            num_freqs=num_freqs,
            include_input=True,
            log_sampling=log_sampling,
            polar_fourier=polar_fourier
        )
        self.location_embedding_size = self.location_encoder.output_dims
        self.location_embedding_weight = nn.Linear(self.location_encoder.output_dims, self.location_encoder.output_dims)

        self.negative_slope = negative_slope

        self.model1 = nn.Sequential(
            nn.Linear(self.location_embedding_size + z_size, hidden_size),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(hidden_size, hidden_size)
        )

        self.intermediate_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.intermediate_layers.append(nn.Sequential(
                nn.Linear(hidden_size + self.location_embedding_size + z_size, hidden_size),
                nn.LeakyReLU(negative_slope=self.negative_slope),
                nn.Linear(hidden_size, hidden_size)
            ))

        self.model2 = nn.Linear(hidden_size, 1)

    def forward(self, id, xy):
        if xy.shape[-1] != self.input_dims:
            xy = self.more_axis(xy)  #
        xy = self.location_embedding(xy)
        poly_embedding = id if id.dtype != torch.long else self.poly_embedding_layer(id)
        xy_z = torch.cat([xy, poly_embedding], dim=1)
        x = self.model1(xy_z)
        for layer in self.intermediate_layers:
            x = torch.cat([xy_z, x], dim=1)
            x = layer(x)
        x = self.model2(x)
        return x


    def location_embedding(self, xy):
        xy = self.location_encoder(xy)
        # xy = self.location_embedding_weight(xy)
        return xy

    def more_axis(self, xy):
        x = xy[:, 0]
        y = xy[:, 1]
        x1 = (x + y) / (2 ** 0.5)
        y1 = (y - x) / (2 ** 0.5)
        return torch.stack([x1, y1, x, y], dim=-1)

class Geo2Vec_Dataset(Dataset):
    def __init__(self, training_samples, dataset_ids):
        ids, samples, dists = [], [], []

        for id in dataset_ids:
            s, d = training_samples[id]
            ids.extend([id] * len(s))
            samples.extend(s)
            dists.extend(d)

        self.dataset_ids = torch.tensor(ids, dtype=torch.long)
        self.dataset_samples = torch.tensor(samples, dtype=torch.float32)
        self.dataset_distances = torch.tensor(dists, dtype=torch.float32).view(-1,
                                                                               1)  # Ensure distances are shaped correctly
    def __getitem__(self, index):
        return self.dataset_ids[index], self.dataset_samples[index], self.dataset_distances[index]

    def __len__(self):
        return self.dataset_ids.shape[0]


class SDFLoss(nn.Module):
    def __init__(self,sum=True,  code_reg_weight=1.0):
        """
        Args:
            delta (float): truncation threshold for clamped L1 loss.
            code_reg_weight (float): weight for the latent code regularization (1/σ² in the paper).
        """
        super().__init__()
        self.code_reg_weight = code_reg_weight
        self.sum = sum

    def forward(self, pred_sdf, true_sdf, latent_code=None, code_reg_weight=None):
        """
        Args:
            pred_sdf (Tensor): predicted SDF values, shape [B, N]
            true_sdf (Tensor): ground-truth SDF values, shape [B, N]
            latent_code (Tensor): latent vector per shape, shape [B, D]
        """
        if self.code_reg_weight > 0.0 or code_reg_weight is not None:
            total_loss = torch.sum(torch.abs(pred_sdf - true_sdf)) if self.sum else torch.mean(torch.abs(pred_sdf - true_sdf))
            if code_reg_weight is not None:
                total_loss += torch.mean(latent_code.pow(2)) * code_reg_weight
            else:
                total_loss += torch.mean(latent_code.pow(2)) * self.code_reg_weight
        else:
            total_loss = torch.sum(torch.abs(pred_sdf - true_sdf)) if self.sum else torch.mean(torch.abs(pred_sdf - true_sdf))
        return total_loss

class PositionalEncoder(nn.Module):
    def __init__(self, input_dims, num_freqs, include_input=True, log_sampling=True, polar_fourier=True,L_min=None, L_max=None):
        """
        Args:
            input_dims: Number of input dimensions (e.g., 2 for (x, y))
            num_freqs: Number of frequency bands (L in the paper)
            include_input: Whether to include the original input in the output
            log_sampling: If True, frequencies are spaced logarithmically
        """
        super(PositionalEncoder, self).__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling

        # Precompute and register frequency bands
        freq_bands = self._create_freq_bands(L_min, L_max)
        self.register_buffer("freq_bands", freq_bands)

        self.polar_fourier = polar_fourier
        if polar_fourier:
            self.output_dims = 2 * num_freqs + (2 * num_freqs * input_dims) + (input_dims * 2 if include_input else 0)
        else:
            self.output_dims = (2 * num_freqs * input_dims) + (input_dims if include_input else 0)

    def _create_freq_bands(self,L_min=None,L_max=None):
        if self.log_sampling:
            if L_min is not None and L_max is not None:
                pass
            else:
                return (2.0 ** torch.linspace(0.0, self.num_freqs - 1, self.num_freqs) * torch.pi).unsqueeze(1) # This works for (-1,1)
        else:
            if L_min is not None and L_max is not None:
                pass
            else:
                return torch.linspace(1.0, 6.0, self.num_freqs).unsqueeze(1) # This works for (-1,1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, input_dims)
        Returns:
            Positional encoded tensor of shape (batch, output_dims)
        """
        # Shape: (batch, 1, input_dims)
        x_expanded = x.unsqueeze(1)  # (B, 1, D)

        # Shape: (num_freqs, 1)
        freq_bands = self.freq_bands  # (L, 1)

        # Shape: (batch, num_freqs, input_dims)
        x_freq = x_expanded * freq_bands  # (B, L, D)

        # Apply sin and cos
        sin = torch.sin(x_freq)
        cos = torch.cos(x_freq)

        # If include_input, expand input to match shape
        if self.include_input:
            x_input = x.unsqueeze(1)  # (B, 1, D)
            out = [x_input, sin, cos]
        else:
            out = [sin, cos]
        # Final shape: (batch, output_dims)
        out = torch.cat(out, dim=1).reshape(x.shape[0], -1)
        if self.polar_fourier and self.input_dims == 2:
            pe = self.polar_fourier_encoding(x)  # (B, num_freqs*2)
            out = torch.cat([out, pe], dim=-1)
        return out

    def polar_fourier_encoding(self, xy):
        # xy: (N, 2) tensor
        x, y = xy[:, 0], xy[:, 1]
        r = torch.sqrt(x ** 2 + y ** 2)  # Rotation-invariant radius

        # Fourier features on r
        freqs = r.unsqueeze(1) * self.freq_bands.squeeze(1)  # (N, num_freqs)
        pe = torch.cat([xy, torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return pe  # Shape: (N, num_freqs*2)