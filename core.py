"""Tensor algorithms adapted from PLAN-Lab/EditVid (Apache-2.0).

No ComfyUI or Diffusers imports: the numerical parts are independently testable.
"""
import torch
import torch.nn.functional as F


def correspondence(anchor, current, width, tau, radius, keep_fraction, generator):
    """Forward nearest neighbours filtered by reverse-cycle distance.

    Tile the similarity matrix to avoid quadratic peak storage at high resolution.
    Returns unique destinations, deterministically resolving collisions by score.
    """
    a = F.normalize(anchor.float(), dim=-1)
    b = F.normalize(current.float(), dim=-1)
    n = a.shape[0]
    scores = torch.empty(n, device=a.device)
    targets = torch.empty(n, dtype=torch.long, device=a.device)
    reverse_scores = torch.full((n,), -float("inf"), device=a.device)
    reverse = torch.zeros(n, dtype=torch.long, device=a.device)
    for start in range(0, n, 256):
        similarity = a[start:start + 256] @ b.T
        scores[start:start + 256], targets[start:start + 256] = similarity.max(1)
        values, indices = similarity.max(0)
        better = values > reverse_scores
        reverse[better] = indices[better] + start
        reverse_scores = torch.maximum(reverse_scores, values)
    source = torch.arange(n, device=a.device)
    cycle = reverse[targets]
    distance = (source // width - cycle // width).square() + (source % width - cycle % width).square()
    valid = source[(scores >= tau) & (distance < radius ** 2)]
    count = 0 if keep_fraction <= 0 else min(len(valid), max(1, round(len(valid) * keep_fraction)))
    order = torch.randperm(len(valid), generator=generator, device="cpu")[:count].to(a.device)
    valid = valid[order]
    # Highest confidence wins when multiple source tokens select one destination.
    ranked = valid[torch.argsort(scores[valid], descending=True, stable=True)].tolist()
    selected, seen = [], set()
    destinations = targets.cpu().tolist()
    for index in ranked:
        if destinations[index] not in seen:
            selected.append(index)
            seen.add(destinations[index])
    selected = torch.tensor(selected, device=a.device, dtype=torch.long)
    return selected.cpu(), targets[selected].cpu()


def latent_anchor(x, source, strength, feather=3):
    diff = (x.float() - source.float()).abs().mean(1, keepdim=True)
    flattened = diff.flatten(1)
    low = torch.quantile(flattened, .25, dim=1)[:, None, None, None]
    high = torch.quantile(flattened, .65, dim=1)[:, None, None, None]
    mask = ((diff - low) / (high - low).clamp_min(1e-6)).clamp(0, 1).pow(.7)
    if feather:
        offsets = torch.arange(-feather, feather + 1, device=x.device, dtype=torch.float32)
        kernel = torch.exp(-offsets.square() / (2 * (feather / 2) ** 2))
        kernel /= kernel.sum()
        mask = F.conv2d(mask, kernel[None, None, None, :], padding=(0, feather))
        mask = F.conv2d(mask, kernel[None, None, :, None], padding=(feather, 0))
    weight = strength * (1 - mask.clamp(0, 1))
    return ((1 - weight) * x.float() + weight * source.float()).to(x.dtype)


def integrate(source, sigmas, velocity, invert, anchor_strength=0., trajectory=None, callback=None):
    """RF midpoint inversion / Euler editing, with an exact velocity at sigma zero."""
    x = source.float().clone()
    schedule = sigmas.flip(0) if invert else sigmas
    history = [x.detach().cpu().clone()] if invert else None
    for i, (sigma, next_sigma) in enumerate(zip(schedule[:-1], schedule[1:])):
        dt = next_sigma - sigma
        v = velocity(x, sigma, i, False)
        denoised = x - sigma * v
        if invert:
            midpoint = x + .5 * dt * v
            v = velocity(midpoint, sigma + .5 * dt, i, True)
        x = x + dt * v
        if invert:
            history.append(x.detach().cpu().clone())
        elif anchor_strength and i < len(schedule) - 2:
            x = latent_anchor(x, trajectory[-2 - i].to(x), anchor_strength)
        if callback:
            callback(i, denoised, x, len(schedule) - 1)
    return x, history
