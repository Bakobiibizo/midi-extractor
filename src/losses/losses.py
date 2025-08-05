def loss_fn(outputs, targets, weights=None):
    """
    Compute combined loss for onset, frame, and velocity.

    Args:
        outputs: dict of model outputs (onset, frame, velocity)
        targets: dict of ground truth tensors
        weights: optional dict of weighting terms for each head
    """
    if weights is None:
        weights = {"onset": 1.0, "frame": 1.0, "velocity": 0.5}

    onset_loss = F.binary_cross_entropy(outputs["onset"], targets["onset"])
    frame_loss = F.binary_cross_entropy(outputs["frame"], targets["frame"])
    velocity_loss = F.mse_loss(outputs["velocity"], targets["velocity"])

    total = (
        weights["onset"] * onset_loss +
        weights["frame"] * frame_loss +
        weights["velocity"] * velocity_loss
    )

    return total, {
        "onset": onset_loss.item(),
        "frame": frame_loss.item(),
        "velocity": velocity_loss.item(),
        "total": total.item()
    }
