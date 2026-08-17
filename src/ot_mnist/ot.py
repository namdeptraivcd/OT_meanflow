import torch

try:
    from scipy.optimize import linear_sum_assignment

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


@torch.no_grad()
def hard_assignment_pair(e, x):
    """Return x_sigma paired to e by hard balanced assignment."""
    b = e.shape[0]
    e_flat = e.flatten(1)
    x_flat = x.flatten(1)
    cost = torch.cdist(e_flat.float(), x_flat.float()).pow(2)
    if HAS_SCIPY:
        _, col = linear_sum_assignment(cost.detach().cpu().numpy())
        col = torch.as_tensor(col, device=x.device, dtype=torch.long)
        return x[col], col
    return greedy_pair_from_score(-cost)


@torch.no_grad()
def greedy_pair_from_score(score):
    b = score.shape[0]
    used = torch.zeros(b, dtype=torch.bool, device=score.device)
    cols = []
    for i in range(b):
        s = score[i].clone()
        s[used] = -float("inf")
        j = s.argmax()
        used[j] = True
        cols.append(j)
    col = torch.stack(cols)
    return col


@torch.no_grad()
def sinkhorn_plan(cost, epsilon=0.05, iters=80):
    """Balanced entropic OT plan for uniform batches."""
    b = cost.shape[0]
    log_k = -cost.float() / epsilon
    log_u = torch.zeros(b, device=cost.device)
    log_v = torch.zeros(b, device=cost.device)
    log_a = torch.full((b,), -torch.log(torch.tensor(float(b), device=cost.device)), device=cost.device)
    log_b = log_a
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(log_k + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_k + log_u[:, None], dim=0)
    return torch.exp(log_k + log_u[:, None] + log_v[None, :])


@torch.no_grad()
def sinkhorn_hard_pair(e, x, epsilon=0.05, iters=80):
    """Low-entropy Sinkhorn followed by hard balanced rounding."""
    e_flat = e.flatten(1)
    x_flat = x.flatten(1)
    cost = torch.cdist(e_flat.float(), x_flat.float()).pow(2)
    plan = sinkhorn_plan(cost, epsilon=epsilon, iters=iters)
    if HAS_SCIPY:
        _, col = linear_sum_assignment((-plan).detach().cpu().numpy())
        col = torch.as_tensor(col, device=x.device, dtype=torch.long)
    else:
        col = greedy_pair_from_score(plan)
    return x[col], col

