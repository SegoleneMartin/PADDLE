# Adaptation of the publicly available code of the NeurIPS 2020 paper entitled "TIM: Transductive Information Maximization":
# https://github.com/mboudiaf/TIM
import torch.nn.functional as F
from src.utils import get_mi, get_cond_entropy, get_entropy, get_one_hot, Logger, extract_features
from tqdm import tqdm
import numpy as np
import torch
import time
import os
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment


class KM(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.iter = args.iter
        self.alpha = args.alpha
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.n_ways = args.n_ways

    def init_info_lists(self):
        self.timestamps = []
        self.test_acc = []
        self.test_F1 = []
        self.losses = []

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

    def get_preds(self, p):
        """
        inputs:
            p : torch.Tensor of shape [n_task, s_shot, feature_dim]

        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        preds = p.argmax(2)
        return preds

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
        one_hot = get_one_hot(y_s).to(self.device)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts
        """
        self.record_info(new_time=time.time()-t0,
                         support=support,
                         query=query,
                         y_s=y_s,
                         y_q=y_q)
        """


    def record_info(self, new_time, support, query, y_s, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """

        preds_q = self.p.argmax(2)
        n_tasks, q_shot = preds_q.size()
        self.timestamps.append(new_time)
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)
        union = list(range(self.n_ways))
        for i in range(n_tasks):
            ground_truth = list(y_q[i].reshape(q_shot).cpu().numpy())
            preds = list(preds_q[i].reshape(q_shot).cpu().numpy())
            f1 = f1_score(ground_truth, preds, average='weighted', labels=union, zero_division=1)
            self.test_F1.append(f1)

    def get_logs(self):
        self.test_F1 = np.array([self.test_F1])
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps,
                'acc': self.test_acc, 'F1': self.test_F1, 'losses': self.losses}

    def normalization(self, z_s, z_q):
        """
            inputs:
                z_s : np.Array of shape [n_task, s_shot, feature_dim]
                z_q : np.Array of shape [n_task, q_shot, feature_dim]
                train_mean: np.Array of shape [feature_dim]
        """


        norm_s = torch.norm(z_s.cpu(), dim=(2))
        norm_q = torch.norm(z_q.cpu(), dim=(2))

        norm = torch.max(torch.max(norm_s), torch.max(norm_q)).cuda()

        z_s = z_s /norm
        z_q = z_q /norm
        return z_s, z_q

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
        

class SOFT_KM(KM):
    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)

    def __del__(self):
        self.logger.del_logger()

    def p_update(self, query):
        """
        inputs:
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
        """
        u = - 1 / 2 * self.get_logits(query).detach()
        self.p = (u).softmax(2)

    def weights_update(self, support, query, y_s_one_hot):
        """
        Corresponds to w_k updates
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, s_shot, n_ways]


        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
    
        num = torch.einsum('bkq,bqd->bkd',torch.transpose(self.p, 1, 2), query) \
                                                + torch.einsum('bkq,bqd->bkd',torch.transpose(y_s_one_hot, 1, 2), support)
        den  = self.p.sum(1) + y_s_one_hot.sum(1)
        self.weights = torch.div(num, den.unsqueeze(2))

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
        t0 = time.time()
        y_s_one_hot = get_one_hot(y_s).to(self.device)

        for i in tqdm(range(self.iter)):
            self.p_update(query)
            self.weights_update(support, query, y_s_one_hot)

        t1 = time.time()
        self.record_info(new_time=t1-t0,
                            support=support,
                            query=query,
                            y_s=y_s,
                            y_q=y_q)
        t0 = time.time()
        

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
        dist[dist==0.] = 1.
        scale = 1.0 /  dist
        ratio = query.min(dim=1, keepdim=True)[0]
        query.mul_(scale).sub_(ratio)
        support.mul_(scale).sub_(ratio)
        return query, support
