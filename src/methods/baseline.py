import torch.nn.functional as F
from src.utils import get_one_hot, Logger
from tqdm import tqdm
import torch
import time

class Baseline(object):

    def __init__(self, model, device, log_file, args):
        self.device = device
        self.temp = args.temp
        self.iter = args.iter
        self.lr = float(args.lr_baseline)
        self.number_tasks = args.batch_size
        self.model = model
        self.log_file = log_file
        self.n_ways = args.n_ways
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.dataset = args.dataset
        self.used_set_support = args.used_set_support

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        self.timestamps = []
        self.test_acc = []

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """

        n_tasks = samples.size(0)
        logits = self.temp * (samples.matmul(self.w.transpose(1, 2)) \
                              - 1 / 2 * (self.w**2).sum(2).view(n_tasks, 1, -1) \
                              - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

    def init_w(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        self.model.eval()
        t0 = time.time()
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s).to(self.device)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support.to(self.device))
        self.w = weights / counts
        self.model.train()

        logits_q = self.get_logits(query).detach()
        q_probs = logits_q.softmax(2)
        return q_probs

    def record_info(self, new_time, query, y_q):
        """
        inputs:
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_q : torch.Tensor of shape [n_task, q_shot]
        """

        logits_q = self.get_logits(query).detach()
        preds_q = logits_q.argmax(2)
        self.timestamps.append(new_time)
        self.test_acc.append((preds_q == y_q).float().mean(1, keepdim=True))

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'acc': self.test_acc}

    def run_method(self, support, query, y_s, y_q):
        
        """
        Corresponds to the BASELINE inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]     # (centroids)
        """

        # Record info if there's no Baseline iteration
        if self.iter == 0:
            t1 = time.time()
            self.model.eval()
            self.record_info(query=query, y_q=y_q)
        else:
            self.logger.info(" ==> Executing Baseline adaptation over {} iterations on {} shot tasks...".format(self.iter, shot))

            self.w.requires_grad_()
            optimizer = torch.optim.Adam([self.w], lr=self.lr)
            y_s_one_hot = get_one_hot(y_s)
            self.model.train()

            for i in tqdm(range(self.iter)):

                logits_s = self.get_logits(support)
                ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
                loss = ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            self.model.eval()
            self.record_info(query=query, y_q=y_q)


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
        support = x_s.to(self.device)  
        query = x_q.to(self.device)  
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Perform normalizations
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)
        
        # Init basic prototypes
        self.init_w(support=support, y_s=y_s, query=query)

        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs
