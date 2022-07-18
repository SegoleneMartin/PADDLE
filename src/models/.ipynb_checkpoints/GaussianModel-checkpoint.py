# Adaptation of the publicly available code of the paper entitled "Leveraging the Feature Distribution in Transfer-based Few-Shot Learning":
# https://github.com/yhu01/PT-MAP
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import get_one_hot


class GaussianModel(object):
    def __init__(self, device, k_eff, n_ways, lam, imbalanced_support=True):
        self.device = device
        self.mus = None  # shape [n_runs][k_eff][n_nfeat]
        self.k_eff = k_eff
        self.lam = lam
        self.n_ways = n_ways
        self.imbalanced_support = imbalanced_support

    def clone(self):
        other = GaussianModel(self.k_eff)
        other.mus = self.mus.clone()
        return self

    def initFromLabelledDatas(self, data, y_s, n_tasks, shot, k_eff, n_queries, n_nfeat):
        if self.imbalanced_support==False:
            self.mus = data.reshape(n_tasks, shot+n_queries,k_eff, n_nfeat)[:,:shot,].mean(1)
        else:
            one_hot = get_one_hot(y_s)
            counts = one_hot.sum(1).view(data.size()[0], -1, 1)
            weights = one_hot.transpose(1, 2).matmul(data)
            self.mus = weights / counts

    def updateFromEstimate(self, estimate, alpha):

        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):

        r = r.to(self.device)
        c = c.to(self.device)
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M).to(self.device)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)

        u = torch.zeros(n_runs, n).to(self.device)
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)

    def getProbas(self, data, y_s, n_tasks, n_sum_query, n_queries, shot):
        # compute squared dist to centroids [n_runs][n_samples][k_eff]
        dist = (data.unsqueeze(2) - self.mus.unsqueeze(1)).norm(dim=3).pow(2)

        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_tasks, n_sum_query)
        c = torch.ones(n_tasks, self.n_ways) * n_queries

        #n_lsamples = self.k_eff * shot
        #n_lsamples = self.n_ways * shot
        n_lsamples = y_s.size()[1]

        # Query probabilities
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, n_lsamples:] = p_xj_test
        
        # Support probabilities
        p_xj[:, :n_lsamples].fill_(0)
        p_xj[:, :n_lsamples].scatter_(2, y_s.unsqueeze(2), 1)

        return p_xj, p_xj_test

    def estimateFromMask(self, data, mask):

        emus = mask.permute(0, 2, 1).matmul(data).div(mask.sum(dim=1).unsqueeze(2))

        return emus