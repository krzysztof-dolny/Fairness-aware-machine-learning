import torch
import torch.nn as nn

from .metrics import accuracy_equality, statistical_parity, equal_opportunity, predictive_equality


def fair_bce_loss(outputs, targets, groups, alpha=0.75, fairness_mode="AE", val=False, log=False):
    outputs = outputs.squeeze()
    targets = targets.squeeze()
    groups = groups.squeeze()
    
    bce_loss_fn = nn.BCELoss()
    base_loss = bce_loss_fn(outputs, targets)

    if fairness_mode == "AE":
        fair_loss = loss_accuracy_equality(outputs, targets, groups)
        fair_score, group_fair_scores = accuracy_equality(outputs, targets, groups)
    elif fairness_mode == "SP":
        fair_loss = loss_statistical_parity(outputs, groups)
        fair_score, group_fair_scores = statistical_parity(outputs, groups)
    elif fairness_mode == "EOP":
        fair_loss = loss_equal_opportunity(outputs, targets, groups)
        fair_score, group_fair_scores = equal_opportunity(outputs, targets, groups)
    elif fairness_mode == "PE":
        fair_loss = loss_predictive_equality(outputs, targets, groups)
        fair_score, group_fair_scores = predictive_equality(outputs, targets, groups)
    else:
        raise ValueError(f"Unsupported fairness mode: {fairness_mode}")

    total_loss = (alpha * base_loss) + ((1 - alpha) * fair_loss)

    if log and not val:
        group_str = ', '.join([
            f'G{int(g.item())}: {v:.3f}' for g, v in zip(torch.unique(groups), group_fair_scores)
        ])
        print(f"[Fairness] {fairness_mode}: {fair_score:.3f} | {fairness_mode} per group: {group_str}")
        print(f"[Loss] BCE: {base_loss.item():.3f} | Fairness Loss: {fair_loss:.3f} | Total Loss: {total_loss:.3f}"
              f" | Alpha: {alpha:.3f}")

    return total_loss


def calculate_alpha(epoch, total_epochs, alpha, alpha_mode="const", raw_alpha=None):
    if alpha_mode == "const":
        return alpha
    elif alpha_mode == "linear_decrease":
        alpha_start = alpha
        alpha_end = 1 - alpha
        slope = (alpha_end - alpha_start) / total_epochs
        return alpha_start + slope * epoch
    elif alpha_mode == "linear_increase":
        alpha_start = 1 - alpha
        alpha_end = alpha
        slope = (alpha_end - alpha_start) / total_epochs
        return alpha_start + slope * epoch
    # elif alpha_mode == "learnable":
    #    if raw_alpha is None:
    #        raise ValueError("raw_alpha must be provided when using learnable alpha.")
    #    return torch.sigmoid(raw_alpha)
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


def loss_statistical_parity(outputs, groups):
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
