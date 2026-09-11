import torch

from src.utils.tensor_typing import Float


def stratify_loss_by_time(
    batch_t: Float['b'],
    batch_loss: Float['b'],
    num_bins: int = 4,
    loss_name: str = "loss",
) -> dict[str, float]:
    """Bin per-sample losses by timestep range for diagnostic logging."""
    bin_edges = torch.linspace(0, 1, num_bins + 1, device=batch_t.device)

    # Assign each timestep to a bin
    bin_indices = torch.bucketize(batch_t, boundaries=bin_edges)
    bin_indices = torch.clamp(bin_indices, min=1, max=num_bins) - 1

    # Aggregate losses per bin via vectorized bincount
    binned_loss_sum = torch.bincount(bin_indices, weights=batch_loss, minlength=num_bins)
    binned_counts = torch.bincount(bin_indices, minlength=num_bins)

    stratified_losses = {}
    for i in range(num_bins):
        t_range_key = f"{loss_name} t=[{bin_edges[i]:.2f},{bin_edges[i + 1]:.2f})"
        mean_loss = binned_loss_sum[i] / binned_counts[i] if binned_counts[i] > 0 else float("nan")
        stratified_losses[t_range_key] = mean_loss

    return stratified_losses
