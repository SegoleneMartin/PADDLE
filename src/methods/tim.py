# Adaptation of the publicly available code of the NeurIPS 2020 paper entitled "TIM: Transductive Information Maximization":
# https://github.com/mboudiaf/TIM
import torch.nn.functional as F
from src.utils import get_mi, get_cond_entropy, get_entropy, get_one_hot, Logger
from tqdm import tqdm
import torch
import time
import numpy as np
from copy import deepcopy


class TIM(object):

    def __init__(self, model, device, log_file, args):
        self.device = device
        self.temp = args.temp
        self.loss_weights = args.loss_weights.copy()
        self.iter = args.iter
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.n_ways = args.n_ways
        self.dataset = args.dataset
        self.used_set_support = args.used_set_support

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        self.timestamps = []
        self.criterions = []
        self.mutual_infos = []
        self.entropy = []
        self.cond_entropy = []
        self.test_acc = []
        self.losses = []

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """

        n_tasks = samples.size(0)
        logits = self.temp * (samples.matmul(self.weights.transpose(1, 2)) \
                              - 1 / 2 * (self.weights**2).sum(2).view(n_tasks, 1, -1) \
                              - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

    def init_weights(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        self.model.eval()
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s).to(self.device)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts
        logits_q = self.get_logits(query).detach()
        q_probs = logits_q.softmax(2)
        return q_probs

    def record_info(self, query, y_q):
        """
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_q : torch.Tensor of shape [n_task, n_query]
        """
        logits_q = self.get_logits(query).detach()
        preds_q = logits_q.argmax(2)
        q_probs = logits_q.softmax(2)
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)

        self.mutual_infos.append(get_mi(probs=q_probs))
        self.entropy.append(get_entropy(probs=q_probs.detach()))
        self.cond_entropy.append(get_cond_entropy(probs=q_probs.detach()))

    def record_convergence(self, new_time, criterions):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query] :
        """
        self.criterions.append(criterions)
        self.timestamps.append(new_time)

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        self.test_F1 = np.array([self.test_F1])
        self.criterions = torch.stack(self.criterions, dim=0).detach().cpu().numpy()
        self.cond_entropy = torch.cat(self.cond_entropy, dim=1).cpu().numpy()
        self.entropy = torch.cat(self.entropy, dim=1).cpu().numpy()
        self.mutual_infos = torch.cat(self.mutual_infos, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps, 'mutual_info': self.mutual_infos,
                'entropy': self.entropy, 'cond_entropy': self.cond_entropy,
                'acc': self.test_acc, 'losses': self.losses, 'F1': self.test_F1, 'criterions':self.criterions}

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the baseline (no transductive inference = SimpleShot)
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        pass

    def run_task(self, task_dic, shot):
        """
        inputs:
            task_dic : dictionnary with n_tasks few-shot tasks
            shot : scalar, number of shots
        """

        # Extract support and query
        y_s = task_dic['y_s']               # [n_task, shot]
        y_q = task_dic['y_q']               # [n_task, n_query]
        x_s = task_dic['x_s']               # [n_task, shot, feature_dim]
        x_q = task_dic['x_q']               # [n_task, n_query, feature_dim]

        # Transfer tensors to GPU if needed
        support = x_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = x_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Perform normalizations required
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)
        support = support.to(self.device)
        query = query.to(self.device)
        
        # Initialize weights lambda
        self.compute_lambda(support=support, query=query)
        # Init basic prototypes
        self.init_weights(support=support, y_s=y_s, query=query)

        # Run method
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs


class TIM_GD(TIM):
    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)
        self.lr = float(args.lr_tim)

    def run_method(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the TIM-GD inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        t0 = time.time()

        self.weights.requires_grad_()
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        self.model.train()

        for i in tqdm(range(self.iter)):

            weights_old = deepcopy(self.weights.detach())
            t0 = time.time()
            logits_s = self.get_logits(support)
            logits_q = self.get_logits(query)

            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            q_probs = logits_q.softmax(2)
            q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            q_ent = - (q_probs.mean(1) * torch.log(q_probs.mean(1) + 1e-12)).sum(1).sum(0)
            loss = self.loss_weights[0] * ce - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()
            weight_diff = (weights_old - self.weights).norm(dim=-1).mean(-1)
            criterions = weight_diff
            self.record_convergence(new_time=t1-t0, criterions=criterions)

        self.model.eval()
        self.record_info(query=query, y_q=y_q)


class ALPHA_TIM(TIM):

    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)
        self.lr = float(args.lr_alpha_tim)
        self.entropies = args.entropies.copy()
        self.alpha_values = args.alpha_values

    def run_method(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the ALPHA-TIM inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        self.logger.info(" ==> Executing ALPHA-TIM adaptation over {} iterations on {} shot tasks with alpha = {}...".format(self.iter, shot, self.alpha_values))

        self.weights.requires_grad_()
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        self.model.train()

        for i in tqdm(range(self.iter)):
            weights_old = deepcopy(self.weights.detach())
            t0 = time.time()
            logits_s = self.get_logits(support)
            logits_q = self.get_logits(query)
            q_probs = logits_q.softmax(2)

            # Cross entropy type
            if self.entropies[0] == 'Shannon':
                ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            elif self.entropies[0] == 'Alpha':
                ce = torch.pow(y_s_one_hot, self.alpha_values[0]) * torch.pow(logits_s.softmax(2) + 1e-12, 1 - self.alpha_values[0])
                ce = ((1 - ce.sum(2))/(self.alpha_values[0] - 1)).mean(1).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Marginal entropy type
            if self.entropies[1] == 'Shannon':
                q_ent = - (q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0)
            elif self.entropies[1] == 'Alpha':
                q_ent = ((1 - (torch.pow(q_probs.mean(1), self.alpha_values[1])).sum(1)) / (self.alpha_values[1] - 1)).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Conditional entropy type
            if self.entropies[2] == 'Shannon':
                q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            elif self.entropies[2] == 'Alpha':
                q_cond_ent = ((1 - (torch.pow(q_probs + 1e-12, self.alpha_values[2])).sum(2)) / (self.alpha_values[2] - 1)).mean(1).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Loss
            loss = self.loss_weights[0] * ce - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()
            weight_diff = (weights_old - self.weights).norm(dim=-1).mean(-1)
            criterions = weight_diff
            self.record_convergence(new_time=t1-t0, criterions=criterions)

        self.model.eval()
        self.record_info(query=query, y_q=y_q)

