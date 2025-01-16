import torch


def monotonicity_measure_func1(q_hats):
    """shape of q_hats: (batch size x num alphas)"""
    q_diff = q_hats[:, 1:] - q_hats[:, :-1]
    gt_zero_counts = torch.sum(q_diff > 0, dim=1)
    gt_zero = torch.where(gt_zero_counts > 0, 1, 0)
    return torch.sum(gt_zero) / len(gt_zero)


def get_hs_mask(q_hats):
    """shape of q_hats: (batch size x num alphas)"""
    q_diff = q_hats[:, 1:] - q_hats[:, :-1]
    q_non_mono = torch.where(q_diff > 0, 1, 0)
    pad_zeros = torch.zeros((q_hats.shape[0], 1)).to(device=q_hats.device)
    q_non_mono1 = torch.concat((q_non_mono, pad_zeros), dim=-1)
    q_non_mono2 = torch.concat((pad_zeros, q_non_mono), dim=-1)
    q_mask = q_non_mono1 + q_non_mono2
    q_mask[q_mask > 0] = 1
    return q_mask

# test
# a = torch.tensor([
#     [3, 2, 1], 
#     [4, 3, 2],
# ])

# b = torch.tensor([
#     [1, 2, 3],
#     [3, 2, 1]
# ])

# print(monotonicity_measure_func1(a))
# print(monotonicity_measure_func1(b))
    