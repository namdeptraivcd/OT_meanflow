import math

import numpy as np
import torch
import torch.nn.functional as F


def class_entropy(probs):
    hist = probs.mean(dim=0)
    ent = -(hist * (hist + 1e-8).log()).sum()
    return ent.item() / math.log(probs.shape[1])


def frechet_distance(feats_a, feats_b):
    a = feats_a.detach().cpu().numpy().astype(np.float64)
    b = feats_b.detach().cpu().numpy().astype(np.float64)
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a = np.cov(a, rowvar=False)
    cov_b = np.cov(b, rowvar=False)
    diff = mu_a - mu_b
    try:
        from scipy.linalg import sqrtm

        covmean = sqrtm(cov_a @ cov_b)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        trace = np.trace(cov_a + cov_b - 2.0 * covmean)
    except Exception:
        vals, vecs = np.linalg.eigh(cov_a @ cov_b)
        vals = np.clip(vals, 0.0, None)
        trace = np.trace(cov_a + cov_b) - 2.0 * np.sqrt(vals).sum()
    return float(diff.dot(diff) + trace)


@torch.no_grad()
def classifier_metrics(classifier, samples, real, device, batch_size=512):
    classifier.eval()
    all_probs = []
    all_gen_feats = []
    all_real_feats = []
    for start in range(0, samples.shape[0], batch_size):
        gen = samples[start : start + batch_size].to(device)
        x = real[start : start + batch_size].to(device)
        logits, gen_feat = classifier(gen, return_features=True)
        _, real_feat = classifier(x, return_features=True)
        all_probs.append(F.softmax(logits, dim=1).cpu())
        all_gen_feats.append(gen_feat.cpu())
        all_real_feats.append(real_feat.cpu())
    probs = torch.cat(all_probs)
    gen_feats = torch.cat(all_gen_feats)
    real_feats = torch.cat(all_real_feats)
    conf = probs.max(dim=1).values.mean().item()
    pred_hist = probs.argmax(dim=1).bincount(minlength=10).float()
    pred_hist = pred_hist / pred_hist.sum().clamp_min(1)
    return {
        "gen_confidence": conf,
        "class_entropy": class_entropy(probs),
        "feature_fid": frechet_distance(gen_feats, real_feats),
        "class_hist": pred_hist.tolist(),
    }

