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
from copy import deepcopy


class KM(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.iter = args.iter
        self.alpha = args.alpha
        self.tau = args.tau
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.n_ways = args.n_ways

    def init_info_lists(self):
        self.timestamps = []
        self.criterions = []
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

    def A(self, p):
        """
        inputs:

            p : torch.tensor of shape [n_tasks, q_shot, num_class]
                where p[i,j,k] = probability of point j in task i belonging to class k
                (according to our L2 classifier)
        returns:
            v : torch.Tensor of shape [n_task, q_shot, num_class]
        """
        q_shot = p.size(1)
        v = p.sum(1) / q_shot
        return v

    def A_adj(self, v, q_shot):
        """
        inputs:
            V : torch.tensor of shape [n_tasks, num_class]
            q_shot : int
        returns:
            p : torch.Tensor of shape [n_task, q_shot, num_class]
        """
        p = v.unsqueeze(1).repeat(1, q_shot, 1) / q_shot
        return p

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
        one_hot = get_one_hot(y_s)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts

    def record_convergence(self, new_time, criterions):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """
        self.criterions.append(criterions)
        self.timestamps.append(new_time)

    
    def record_info(self, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """
        preds_q = self.p.argmax(2)
        n_tasks, q_shot = preds_q.size()
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
        self.criterions = torch.stack(self.criterions, dim=0).cpu().numpy()
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps, 'criterions':self.criterions,
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

class PADDLE(KM):
    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)

    def __del__(self):
        self.logger.del_logger()

    def p_update(self, query):
        """
        inputs:
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
        """
        
        q_shot = query.size(1)
        u = - 1 / 2 * self.get_logits(query).detach()
        self.p = (u + self.alpha * self.A_adj(self.v, q_shot)).softmax(2)

    def v_update(self):
        """
        inputs:
        """
        self.v = torch.log(self.A(self.p) + 1e-6) + 1

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
        self.logger.info(" ==> Executing PADDLE with LAMBDA = {}".format(self.alpha))
        
        y_s_one_hot = get_one_hot(y_s)
        n_task, n_ways = y_s_one_hot.size(0), y_s_one_hot.size(2)
        self.v = torch.zeros(n_task, n_ways).to(self.device)

        for i in tqdm(range(self.iter)):
            weights_old = deepcopy(self.weights.detach())
            t0 = time.time()
            self.p_update(query)
            self.v_update()
            self.weights_update(support, query, y_s_one_hot)
            weight_diff = (weights_old - self.weights).norm(dim=-1).mean(-1)
            criterions = weight_diff
            t1 = time.time()
            self.record_convergence(new_time=t1-t0, criterions=criterions)
            

        self.record_info(y_q=y_q)
        
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