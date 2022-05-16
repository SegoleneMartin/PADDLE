# Adaptation of the publicly available code of the NeurIPS 2020 paper entitled "TIM: Transductive Information Maximization":
# https://github.com/mboudiaf/TIM
import torch.nn.functional as F
from src.utils import get_one_hot, Logger, extract_features
from tqdm import tqdm
import torch
import time
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class KM(object):
    def __init__(self, model, device, log_file, args):
        # matS = np.sort(matX, axis=0)[::-1]ice = device
        self.iter = args.iter
        self.alpha = args.alpha
        self.device = device
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.num_classes = args.num_classes_test

    def init_info_lists(self):
        self.timestamps = []
        self.test_acc = []
        self.losses = []
        self.test_F1 = []

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = (- 2 * samples.matmul(self.weights.transpose(1, 2)) \
                  + (self.weights**2).sum(2).view(n_tasks, 1, -1) \
                  + (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

    def init_weights(self, support, query, y_s, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        t0 = time.time()
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts

    def record_info(self, new_time, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """
        preds_q = self.p.argmax(-1)
        n_tasks, q_shot = preds_q.size()
        self.timestamps.append(new_time)
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)
        union = list(range(self.num_classes))
        for i in range(n_tasks):
            ground_truth = list(y_q[i].reshape(q_shot).cpu().numpy())
            preds = list(preds_q[i].reshape(q_shot).cpu().numpy())
            #union = set.union(set(ground_truth),set(preds))
            f1 = f1_score(ground_truth, preds, average='weighted', labels=union, zero_division=1)
            self.test_F1.append(f1)

    def get_logs(self):
        self.test_F1 = np.array([self.test_F1])
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps,
                'acc': self.test_acc, 'F1': self.test_F1, 'losses': self.losses}

    def run_task(self, task_dic, shot):
        
        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        x_s, x_q = task_dic['x_s'], task_dic['x_q']

        # Transfer tensors to GPU if needed
        support = x_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = x_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Extract features
        #support, query = extract_features(self.model, support, query)
        support = support.to(self.device)
        query = query.to(self.device)

        # Perform normalizations required
        scaler = MinMaxScaler(feature_range=(0, 1))
        query, support = scaler(query, support)

        # Init basic prototypes
        self.init_weights(support=support, y_s=y_s, query=query, y_q=y_q)

        # Run adaptation
        self.run_adaptation(support=support, query=query, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs


class KM_UNBIASED_GD(KM):
    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)
        self.lr = args.lr

    def __del__(self):
        self.logger.del_logger()

    def run_adaptation(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the TIM-ADM inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]


        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        self.logger.info(" ==> Executing KM-UNBIASED with LAMBDA = {}".format(self.alpha))
        
        t0 = time.time()
        y_s_one_hot = F.one_hot(y_s, self.num_classes)
        n_task, num_classes = y_s_one_hot.size(0), y_s_one_hot.size(2)
        n_query = query.size(1)

        # Initialize p
        self.p = self.get_logits(query).softmax(-1)

        self.p.requires_grad_()
        self.weights.requires_grad_()
        optimizer = torch.optim.SGD([self.weights, self.p], lr=self.lr)

        all_samples = torch.cat([support, query], 1)

        for i in tqdm(range(self.iter)):
            t1 = time.time()
            self.record_info(new_time=t1-t0, y_q=y_q)
            t0 = time.time()

            # Data fitting term
            l2_distances = torch.cdist(all_samples, self.weights) ** 2  # [n_tasks, ns + nq, K]
            all_p = torch.cat([y_s_one_hot, self.p], dim=1) # [n_tasks, ns + nq, K]
            data_fitting = 1/2 * (l2_distances * all_p).mean((-2, -1)).sum(0)

            # Complexity term
            marg_p = self.p.mean(1)  # [n_tasks, K]
            marg_ent = - (marg_p * torch.log(marg_p + 1e-12)).sum(-1).sum(0)  # [n_tasks]

            loss = data_fitting - self.alpha * marg_ent

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss)

            # Projection
            with torch.no_grad():
                # self.p = self.simplex_project(self.p)
                self.p = self.p.softmax(-1)
        delattr(self, 'p')
        delattr(self, 'weights')

    def simplex_project(self, p: torch.Tensor, l=1.0):
        """
        Taken from https://www.researchgate.net/publication/283568278_NumPy_SciPy_Recipes_for_Data_Science_Computing_Nearest_Neighbors
        p: [n_tasks, n_q, K]
        """

        # Put in the right form for the function
        matX = p.permute(0, 2, 1).cpu().numpy()

        # Core function
        n_tasks, m, n = matX.shape
        # matS = np.sort(matX, axis=0)[::-1]
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
      

class MinMaxScaler(object):
    """MinMax Scaler

    Transforms each channel to the range [a, b].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, query, support):
        """Fit features

        Parameters
        ----------
        stacked_features : tuple, list
            List of stacked features.

        Returns
        -------
        tensor 
            A tensor with scaled features using requested preprocessor.
        """

        dist = (query.max(dim=1, keepdim=True)[0] - query.min(dim=1, keepdim=True)[0])
        dist[dist == 0.] = 1.
        scale = 1.0 /  dist
        ratio = query.min(dim=1, keepdim=True)[0]
        query.mul_(scale).sub_(ratio)
        support.mul_(scale).sub_(ratio)
        return query, support