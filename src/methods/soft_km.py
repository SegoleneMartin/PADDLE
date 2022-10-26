# Adaptation of the publicly available code of the NeurIPS 2020 paper entitled "TIM: Transductive Information Maximization":
# https://github.com/mboudiaf/TIM
import torch.nn.functional as F
from src.utils import get_one_hot, Logger
from tqdm import tqdm
import torch
import time
from .paddle import KM


class SOFT_KM(KM):
    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)

    def __del__(self):
        self.logger.del_logger()

    def u_update(self, query):
        """
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
        updates:
            self.u : torch.Tensor of shape [n_task, n_query, num_classes]
        """
        u = self.get_logits(query).detach()
        self.u = (u).softmax(2)

    def w_update(self, support, query, y_s_one_hot):
        """
        Corresponds to w_k updates
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, shot, n_ways]


        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
    
        num = torch.einsum('bkq,bqd->bkd',torch.transpose(self.u, 1, 2), query) \
                                                + torch.einsum('bkq,bqd->bkd',torch.transpose(y_s_one_hot, 1, 2), support)
        den  = self.u.sum(1) + y_s_one_hot.sum(1)
        self.w = torch.div(num, den.unsqueeze(2))

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the TIM-ADM inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        y_s_one_hot = get_one_hot(y_s).to(self.device)
        self.init_w(support=support, y_s=y_s)                           # initialize basic prototypes

        for i in tqdm(range(self.iter)):
            self.u_update(query)
            self.w_update(support, query, y_s_one_hot)

        self.record_info(query=query, y_q=y_q)
        

