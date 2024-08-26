def dice_loss(pred, target):
    """
    Calculates the Dice loss between predicted and target masks.

    Args:
        pred: A PyTorch tensor of shape (N, C, H, W) containing the predicted masks.
        target: A PyTorch tensor of shape (N, C, H, W) containing the ground truth masks.

    Returns:
        The calculated Dice loss.
    """

    smooth = 1e-5  # Smoothing term to prevent division by zero

    # Flatten the tensors to (N, C, H * W)
    pred = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)

    # Calculate the intersection and union
    intersection = (pred * target).sum(dim=2)
    union = pred.sum(dim=2) + target.sum(dim=2)

    # Calculate the Dice loss
    dice_score = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score.mean()

    return dice_loss
