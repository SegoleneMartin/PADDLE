import torch
import numpy as np
import math

class CategoriesSampler():
    
    """
            CategorySampler
            inputs:
                label : All labels of dataset
                n_batch : Number of batches to load
                n_cls : Number of classification ways (n_ways)
                s_shot : Support shot
                n_query : Size of query set
                balanced : 'balanced': Balanced query class distribution: Standard class balanced Few-Shot setting
                           'dirichlet': Dirichlet's distribution over query data: Realisatic class imbalanced Few-Shot setting
                alpha : Dirichlet's concentration parameter

            returns :
                sampler : CategoriesSampler object that will yield batch when iterated
                When iterated returns : batch
                        data : torch.tensor [n_support + n_query, channel, H, W]
                               [support_data, query_data]
                        labels : torch.tensor [n_support + n_query]
                               [support_labels, query_labels]

                        Where :
                            Support data and labels class sequence is :
                                [a b c d e a b c d e a b c d e ...]
                             Query data and labels class sequence is :
                               [a a a a a a a a b b b b b b b c c c c c d d d d d e e e e e ...]
    """
    def __init__(self, label_support, label_query, n_batch, n_cls, num_classes, s_shot, n_query, balanced, used_set_support, alpha = 2):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.s_shot = s_shot
        self.n_query = n_query
        self.balanced = balanced
        self.alpha = alpha
        self.num_classes = num_classes
        self.used_set_support = used_set_support
        
    def create_list_classes(self, label_support, label_query):
        label_support = np.array(label_support)  # all data label
        self.m_ind_support = []  # the data index of each class
        for i in range(max(label_support) + 1):
            ind = np.argwhere(label_support == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind_support.append(ind)
            
        label_query = np.array(label_query)  # all data label
        self.m_ind_query = []  # the data index of each class
        for i in range(max(label_support) + 1):
            ind = np.argwhere(label_query == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind_query.append(ind)
            
        self.list_classes = []
        for i_batch in range(self.n_batch):
            self.list_classes.append(torch.randperm(len(self.m_ind_support))[:self.num_classes])  # random sample num_class indexs
        
            
    
class SamplerSupport:
    def __init__(self, cat_samp):
        self.name = "SamplerSupport"
        self.list_classes = cat_samp.list_classes
        self.n_batch = cat_samp.n_batch
        self.s_shot = cat_samp.s_shot
        self.used_set_support = cat_samp.used_set_support
        self.m_ind_support = cat_samp.m_ind_support

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            support = []
            classes = self.list_classes[i_batch]
            
            if self.used_set_support == 'repr':
                for c in classes:
                    l = self.m_ind_support[c]  # all data indexs of this class
                    #pos = torch.randperm(len(l))[:self.s_shot]  # sample n_per data index of this class
                    pos = torch.randperm(len(l))[:]  # sample n_per data index of this class
                    support.append(l[pos])
                #support = torch.stack(support).t().reshape(-1)
                #support = torch.stack(support).reshape(-1)
                support = torch.cat(support)
            else:
                for c in classes:
                    l = self.m_ind_support[c]  # all data indexs of this class
                    #pos = torch.randperm(len(l))[:self.s_shot]  # sample n_per data index of this class
                    pos = torch.randperm(len(l))[:self.s_shot]  # sample n_per data index of this class
                    support.append(l[pos])
                support = torch.stack(support).t().reshape(-1)
                #support = torch.stack(support).reshape(-1)
                # support = torch.cat(support)
            
            yield support

class SamplerQuery:
    def __init__(self, cat_samp):
        self.name = "SamplerQuery"
        self.list_classes = cat_samp.list_classes
        self.n_batch = cat_samp.n_batch
        self.n_cls = cat_samp.n_cls
        self.m_ind_query = cat_samp.m_ind_query
        self.n_query = cat_samp.n_query
        self.alpha = cat_samp.alpha
        self.balanced = cat_samp.balanced

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):

            classes = self.list_classes[i_batch][:self.n_cls]

            # A MODIFIER si autre data que Inat car on pourrait avoir des superpositions de donnees
            query = []
            
            alpha = self.alpha * np.ones(self.n_cls)
            assert self.balanced in ['balanced', 'dirichlet', 'uniform']
            if self.balanced == 'balanced':
                query_samples = np.repeat(self.n_query // self.n_cls, self.n_cls)
                for c, nb_shot in zip(classes, query_samples):
                    l = self.m_ind_query[c]  # all data indexs of this class
                    pos = torch.randperm(len(l))[:nb_shot]  # sample n_per data index of this class
                    query.append(l[pos])
                query = torch.cat(query)
                
            elif self.balanced == "dirichlet":
                sum_pos = 0
                while sum_pos < self.n_query :
                    query = []
                    query_samples = get_dirichlet_query_dist(alpha, 1, self.n_cls, self.n_query)[0]
                    sum_pos = 0
                    for c, nb_shot in zip(classes, query_samples):
                        l = self.m_ind_query[c]  # all data indexs of this class
                        pos = torch.randperm(len(l))[:nb_shot]  # sample n_per data index of this class
                        sum_pos += min(len(pos),nb_shot)
                        #assert len(pos) == nb_shot
                        query.append(l[pos])
            
                query = torch.cat(query)

                
            elif self.balanced == "uniform":
                complete_possible_samples = self.m_ind_query[classes[0]]
                #print("Len class ", len(self.m_ind_query[classes[0]]))
                for c in classes[1:]:
                    complete_possible_samples = torch.cat((complete_possible_samples, self.m_ind_query[c]))
                    #print("Len class ", len(self.m_ind_query[c]))
                pos = torch.randperm(len(complete_possible_samples))[:self.n_query]
                query = complete_possible_samples[pos]
            
            yield query


def convert_prob_to_samples(prob, n_query):
    prob = prob * n_query
    for i in range(len(prob)):
        if sum(np.round(prob[i])) > n_query:
            while sum(np.round(prob[i])) != n_query:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.floor(prob[i, idx])
            prob[i] = np.round(prob[i])
        elif sum(np.round(prob[i])) < n_query:
            while sum(np.round(prob[i])) != n_query:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.ceil(prob[i, idx])
            prob[i] = np.round(prob[i])
        else:
            prob[i] = np.round(prob[i])
    return prob.astype(int)


def get_dirichlet_query_dist(alpha, n_tasks, n_ways, n_querys):
    alpha = np.full(n_ways, alpha)
    prob_dist = np.random.dirichlet(alpha, n_tasks)
    return convert_prob_to_samples(prob=prob_dist, n_query=n_querys)



class SamplerBasic():
    
    """
            CategorySampler
            inputs:

            returns :
                sampler : CategoriesSampler object that will yield batch when iterated
                When iterated returns : batch
                        data : torch.tensor [n_support + n_query, channel, H, W]
                               [support_data, query_data]
                        labels : torch.tensor [n_support + n_query]
                               [support_labels, query_labels]

                        Where :
                            Support data and labels class sequence is :
                                [a b c d e a b c d e a b c d e ...]
                             Query data and labels class sequence is :
                               [a a a a a a a a b b b b b b b c c c c c d d d d d e e e e e ...]
    """
    def __init__(self, index):
        self.index = index

    def __iter__(self):
            
        yield torch.tensor([self.index])