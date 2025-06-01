import torch
import torch.nn as nn

from .metrics import accuracy_equality, statistical_parity, equal_opportunity, predictive_equality


def fair_bce_loss(outputs, targets, groups, alpha=0.75, fairness_mode="SP", log=False):
    """
    Computes a combined loss function for fairness-aware binary classification.

    This function applies a weighted sum of Binary Cross-Entropy (BCE) loss and a selected
    group fairness loss. The weighting parameter alpha determines the trade-off between
    predictive performance and fairness.

    Args:
        outputs (Tensor): Model outputs (probabilities).
        targets (Tensor): Ground truth binary labels.
        groups (Tensor): Group membership indicators.
        alpha (float): Weighting factor. Must be in [0, 1].
        fairness_mode (str): Fairness criterion to use: "AE", "SP", "EOP", "PE".
        log (bool): Log outputs for debugging.

    Returns:
        Tensor: Combined loss value.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be between 0 and 1. Got: {alpha}")

    fairness_loss_functions = {
        "AE": loss_accuracy_equality,
        "SP": loss_statistical_parity,
        "EOP": loss_equal_opportunity,
        "PE": loss_predictive_equality
    }

    fairness_score_functions = {
        "AE": accuracy_equality,
        "SP": statistical_parity,
        "EOP": equal_opportunity,
        "PE": predictive_equality
    }

    if fairness_mode not in fairness_loss_functions:
        raise ValueError(f"Unsupported fairness mode: {fairness_mode}")

    outputs = outputs.squeeze()
    targets = targets.squeeze()
    groups = groups.squeeze()
    
    bce_loss_fn = nn.BCELoss()
    base_loss = bce_loss_fn(outputs, targets)

    fair_loss = fairness_loss_functions[fairness_mode](outputs, targets, groups)
    fair_score, group_fair_scores = fairness_score_functions[fairness_mode](outputs, targets, groups)

    total_loss = (alpha * base_loss / 1.3863) + ((1 - alpha) * fair_loss / 1)

    if log:
        group_str = ', '.join([
            f'G{int(g.item())}: {v:.3f}' for g, v in zip(torch.unique(groups), group_fair_scores)
        ])
        print(f"[Fairness] {fairness_mode}: {fair_score:.3f} | {fairness_mode} per group: {group_str}")
        print(f"[Loss] BCE: {base_loss.item():.3f} | Fairness Loss: {fair_loss:.3f} | Total Loss: {total_loss:.3f}"
              f" | Alpha: {alpha:.3f}")

    return total_loss


def calculate_alpha(epoch, total_epochs, alpha, alpha_mode="const"):
    """
    Computes the value of the alpha parameter used in the fairness-aware loss function.

    This function supports different scheduling strategies for adjusting the alpha parameter
    during training. The parameter alpha controls the trade-off between predictive performance
    and fairness. Depending on the selected mode, alpha can remain constant or change linearly
    over training epochs.

    Args:
        epoch (int): Current epoch number.
        total_epochs (int): Total number of training epochs.
        alpha (float): Base value of alpha used to define scheduling range.
        alpha_mode (str): Scheduling strategy for alpha. Options are:
                            "const" (constant),
                            "linear_increase",
                            "linear_decrease".

    Returns:
        float: Computed alpha value for the current epoch.
    """
    if alpha_mode == "const":
        return alpha

    alpha_min = 0.5 - alpha / 2
    alpha_max = 0.5 + alpha / 2

    if alpha_mode == "linear_decrease":
        return alpha_max - (alpha_max - alpha_min) * (epoch / total_epochs)
    elif alpha_mode == "linear_increase":
        return alpha_min + (alpha_max - alpha_min) * (epoch / total_epochs)
    else:
        raise ValueError(f"Unsupported alpha mode: {alpha_mode}")


def loss_accuracy_equality(outputs, targets, groups, t=25):
    def soft_threshold(x):
        return torch.sigmoid((x - 0.5) * t)

    soft_preds = soft_threshold(outputs)
    soft_acc = 1.0 - torch.abs(soft_preds - targets)

    unique_groups = torch.unique(groups)
    group_accs = []

    for group in unique_groups:
        mask = (groups == group)
        if mask.sum() == 0:
            continue
        group_accs.append(soft_acc[mask].mean())

    acc_tensor = torch.stack(group_accs)
    loss = torch.max(acc_tensor) - torch.min(acc_tensor)

    return loss


def loss_statistical_parity(outputs, targets, groups):
    unique_groups = torch.unique(groups)
    group_means = []

    for group in unique_groups:
        mask = (groups == group)
        if mask.sum() == 0:
            continue
        mean_pred = outputs[mask].mean()
        group_means.append(mean_pred)

    group_means_tensor = torch.stack(group_means)
    loss_sp = torch.max(group_means_tensor) - torch.min(group_means_tensor)

    return loss_sp


def loss_equal_opportunity(outputs, targets, groups):
    unique_groups = torch.unique(groups)
    group_tprs = []

    for group in unique_groups:
        mask = (groups == group) & (targets == 1)
        if mask.sum() == 0:
            continue
        tpr = outputs[mask].mean()
        group_tprs.append(tpr)

    group_tprs_tensor = torch.stack(group_tprs)
    loss_eop = torch.max(group_tprs_tensor) - torch.min(group_tprs_tensor)

    return loss_eop


def loss_predictive_equality(outputs, targets, groups):
    unique_groups = torch.unique(groups)
    group_fprs = []

    for group in unique_groups:
        mask = (groups == group) & (targets == 0)
        if mask.sum() == 0:
            continue
        fpr = outputs[mask].mean()
        group_fprs.append(fpr)

    group_fprs_tensor = torch.stack(group_fprs)
    loss_pe = torch.max(group_fprs_tensor) - torch.min(group_fprs_tensor)

    return loss_pe
