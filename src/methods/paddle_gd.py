import torch.nn.functional as F
from src.utils import Logger
from tqdm import tqdm
import torch
import time
import numpy as np
from .paddle import KM


class PADDLE_GD(KM):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.iter = args.iter
        self.alpha = args.alpha
        self.lr = args.lr
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.n_ways = args.n_ways
        self.criterions = []

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the PADDLE-GD inference (ablation)
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.u : torch.Tensor of shape [n_task, n_query]
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        self.logger.info(" ==> Executing PADDLE with LAMBDA = {}".format(self.alpha))
        
        t0 = time.time()
        y_s_one_hot = F.one_hot(y_s, self.n_ways).to(self.device)

        # Initialize the soft labels u and the prototypes w
        self.u = (self.get_logits(query)).softmax(-1).to(self.device)
        self.u.requires_grad_()
        self.w.requires_grad_()
        optimizer = torch.optim.Adam([self.w, self.u], lr=self.lr)

        all_samples = torch.cat([support.to(self.device), query.to(self.device)], 1)

        for i in tqdm(range(self.iter)):

            w_old = self.w.detach()
            t0 = time.time()
            
            # Data fitting term
            l2_distances = torch.cdist(all_samples, self.w) ** 2  # [n_tasks, ns + nq, K]
            all_p = torch.cat([y_s_one_hot.float(), self.u.float()], dim=1) # [n_task s, ns + nq, K]
            data_fitting = 1 / 2 * (l2_distances * all_p).sum((-2, -1)).sum(0)

            # Complexity term
            marg_u = self.u.mean(1).to(self.device)  # [n_tasks, K]
            marg_ent = - (marg_u * torch.log(marg_u + 1e-12)).sum(-1).sum(0)  # [n_tasks]
            loss = (data_fitting - self.alpha * marg_ent).to(self.device)

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Projection
            with torch.no_grad():
                self.u = self.simplex_project(self.u)
                weight_diff = (w_old - self.w).norm(dim=-1).mean(-1)
                criterions = weight_diff

            t1 = time.time()
            self.record_convergence(new_time=t1-t0, criterions=criterions)

        self.record_info(y_q=y_q)


    def simplex_project(self, u: torch.Tensor, l=1.0):
        """
        Taken from https://www.researchgate.net/publication/283568278_NumPy_SciPy_Recipes_for_Data_Science_Computing_Nearest_Neighbors
        u: [n_tasks, n_q, K]
        """

        # Put in the right form for the function
        matX = u.permute(0, 2, 1).detach().cpu().numpy()

        # Core function
        n_tasks, m, n = matX.shape
        matS = - np.sort(-matX, axis=1)
        matC = np.cumsum(matS, axis=1) - l
        matH = matS - matC / (np.arange(m) + 1).reshape(1, m, 1)
        matH[matH <= 0] = np.inf

        r = np.argmin(matH, axis=1)
        t = []
        for task in range(n_tasks):
            t.append(matC[task, r[task], np.arange(n)] / (r[task] + 1))
        t = np.stack(t, 0)
        matY = matX - t[:, None, :]
        matY[matY < 0] = 0

        # Back to torch
        matY = torch.from_numpy(matY).permute(0, 2, 1).to(self.device)

        assert torch.allclose(matY.sum(-1), torch.ones_like(matY.sum(-1))), \
            "Simplex constraint does not seem satisfied"

        return matY
      