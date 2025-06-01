import torch


def accuracy_equality(outputs, targets, groups):
    pred_labels = (outputs > 0.5).float()
    unique_groups = torch.unique(groups)
    group_ae = []

    for group in unique_groups:
        mask = (groups == group)
        if mask.sum() == 0:
            continue
        ae = (pred_labels[mask] == targets[mask]).float().mean()
        group_ae.append(ae.item())

    ae = max(group_ae) - min(group_ae)

    return ae, group_ae


def statistical_parity(outputs, targets, groups):
    pred_labels = (outputs > 0.5).float()
    unique_groups = torch.unique(groups)
    group_sp = []

    for group in unique_groups:
        mask = (groups == group)
        if mask.sum() == 0:
            continue
        sp = pred_labels[mask].float().mean()
        group_sp.append(sp.item())

    sp = max(group_sp) - min(group_sp)

    return sp, group_sp


def equal_opportunity(outputs, targets, groups):
    pred_labels = (outputs > 0.5).float()
    unique_groups = torch.unique(groups)
    group_eop = []

    for group in unique_groups:
        mask = (groups == group) & (targets == 1)
        if mask.sum() == 0:
            continue
        eop = (pred_labels[mask] == 1).float().mean()
        group_eop.append(eop.item())

    eop = max(group_eop) - min(group_eop)

    return eop, group_eop


def predictive_equality(outputs, targets, groups):
    pred_labels = (outputs > 0.5).float()
    unique_groups = torch.unique(groups)
    group_pe = []

    for group in unique_groups:
        mask = (groups == group) & (targets == 0)
        if mask.sum() == 0:
            continue
        pe = (pred_labels[mask] == 1).float().mean()
        group_pe.append(pe.item())

    pe = max(group_pe) - min(group_pe)

    return pe, group_pe
