import numpy as np
from src.utils import warp_tqdm, compute_confidence_interval, load_checkpoint, Logger, extract_mean_features, extract_features, get_features_simple
from src.methods.tim import ALPHA_TIM, TIM_GD
from src.methods.km_unbiased import KM_UNBIASED
from src.methods.km import KM_BIASED
from src.methods.km_unbiased_GD import KM_UNBIASED_GD
from src.methods.ici import ICI
from src.methods.laplacianshot import LaplacianShot
from src.methods.bdcspn import BDCSPN
from src.methods.baseline import Baseline
from src.methods.pt_map import PT_MAP
from src.datasets import Tasks_Generator, get_dataset, get_dataloader, SamplerSupport, SamplerBasic, SamplerQuery, CategoriesSampler
import torch
#from src import datasets
import os
from src.utils import load_pickle, save_pickle
import random

class Evaluator:
    def __init__(self, device, args, log_file):
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)

        
    def extract_features(self, model, model_path, model_tag, used_set, used_set_name, fresh_start, loaders_dic):
        """
        inputs:
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            used_set : Set used between 'test' and 'val'
            n_ways : Number of ways for the task

        returns :
            extracted_features_dic : Dictionnary containing all extracted features and labels
        """

        # Load features from memory if previously saved ...
        print("used_set_name", used_set_name)
        save_dir = os.path.join(model_path, model_tag, used_set_name)
        filepath = os.path.join(save_dir, 'output.plk')
        if os.path.isfile(filepath) and (not fresh_start):
            extracted_features_dic = load_pickle(filepath)
            print(" ==> Features loaded from {}".format(filepath))
            return extracted_features_dic

        # ... otherwise just extract them
        else:
            print(" ==> Beginning feature extraction")
            os.makedirs(save_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():

            all_features = []
            all_labels = []
            for i, (inputs, labels, _) in enumerate(warp_tqdm(loaders_dic[used_set], False)):
                inputs = inputs.to(self.device).unsqueeze(0)
                labels = torch.Tensor([labels])
                outputs, _ = model(inputs, True)
                all_features.append(outputs.cpu())
                all_labels.append(labels)
            all_features = torch.cat(all_features, 0)
            all_labels = torch.cat(all_labels, 0)
            extracted_features_dic = {'concat_features': all_features,
                                      'concat_labels': all_labels
                                      }
        print(" ==> Saving features to {}".format(filepath))
        save_pickle(filepath, extracted_features_dic)
        return extracted_features_dic
                
                

    def run_full_evaluation(self, model):
        """
        Run the evaluation over all the tasks
        inputs:
            model : The loaded model containing the feature extractor
            args : All parameters

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        self.logger.info("=> Runnning full evaluation with method: {}".format(self.args.method))
        load_checkpoint(model=model, model_path=self.args.ckpt_path, type=self.args.model_tag)
        dataset = {}
        loader_info = {'aug': False, 'out_name': False}

        if self.args.target_data_path is not None:  # This mean we are in the cross-domain scenario
            loader_info.update({'path': self.args.target_data_path,
                                'split_dir': self.args.target_split_dir})

        support_set = get_dataset(self.args.used_set_support, args=self.args, **loader_info)
        dataset.update({'support': support_set})
        query_set = get_dataset(self.args.used_set_query, args=self.args, **loader_info)
        dataset.update({'query': query_set})

        print("support len", len(support_set))
        print("query len", len(query_set))
        
        ## Compute train mean
        name_file = 'train_mean_' + self.args.dataset + '_' + self.args.arch + '.pt'
        if os.path.isfile(name_file) == False:
            train_set = get_dataset('train', args=self.args, **loader_info)
            dataset['train_loader'] = train_set
            train_loader = get_dataloader(sets=train_set, args=self.args)
            train_mean, _ = extract_mean_features(model=model,  train_loader=train_loader, args=self.args,
                                                logger=self.logger, device=self.device)
            torch.save(train_mean, name_file)
        else:
            train_mean = torch.load(name_file)

        # Extract features (just load them if already in memory)
        extracted_features_dic_support = self.extract_features(model=model,
                                                       model_path=self.args.ckpt_path, model_tag=self.args.model_tag,
                                                       loaders_dic=dataset, used_set='support',
                                                       used_set_name = self.args.used_set_support,
                                                               fresh_start=self.args.fresh_start)
        if self.args.used_set_support != self.args.used_set_query:
            extracted_features_dic_query = self.extract_features(model=model,
                                                       model_path=self.args.ckpt_path, model_tag=self.args.model_tag,
                                                       loaders_dic=dataset, used_set='query', 
                                                        used_set_name = self.args.used_set_query,
                                                                 fresh_start=self.args.fresh_start)
        else:
            extracted_features_dic_query = extracted_features_dic_support 

        all_features_support = extracted_features_dic_support['concat_features']
        all_labels_support = extracted_features_dic_support['concat_labels'].long()
        all_features_query = extracted_features_dic_query['concat_features']
        all_labels_query = extracted_features_dic_query['concat_labels'].long()

        results = []
        results_F1 = []
        for shot in self.args.shots:
        
            results_task = []
            results_task_F1 = []
            for i in range(int(self.args.number_tasks/self.args.batch_size)):
                sampler = CategoriesSampler(all_labels_support, all_labels_query, self.args.batch_size,
                                        self.args.n_ways, self.args.num_classes_test, shot, self.args.n_query, 
                                        self.args.balanced, self.args.used_set_support, self.args.alpha_dirichlet)
                sampler.create_list_classes(all_labels_support, all_labels_query)
                sampler_support = SamplerSupport(sampler)
                sampler_query = SamplerQuery(sampler)

                test_loader_query = []
                for indices in sampler_query :
                    test_loader_query.append((all_features_query[indices,:], all_labels_query[indices]))

                test_loader_support = []
                for indices in sampler_support :
                    test_loader_support.append((all_features_support[indices,:], all_labels_support[indices]))
      
                task_generator = Tasks_Generator(n_ways=self.args.n_ways, num_classes=self.args.num_classes_test, shot=shot, n_query=self.args.n_query, loader_support=test_loader_support, loader_query=test_loader_query, train_mean=train_mean, log_file=self.log_file)
              
                method = self.get_method_builder(model=model)

                tasks = task_generator.generate_tasks()
                # Run task
                logs = method.run_task(task_dic=tasks, shot=shot)
                acc_mean, acc_conf = compute_confidence_interval(logs['acc'][:, -1])
                F1_mean, F1_conf = compute_confidence_interval(logs['F1'][:, -1])

                results_task.append(acc_mean)
                results_task_F1.append(F1_mean)
                del method
                del tasks
            results.append(results_task)
            results_F1.append(results_task_F1)

        mean_accuracies = np.asarray(results).mean(1)
        mean_F1s = np.asarray(results_F1).mean(1)

        if self.args.method == 'ALPHA-TIM':
            param = self.args.alpha_values[1]
        elif self.args.method == 'KM-UNBIASED':
            param = self.args.alpha
        elif self.args.method == 'KM-BIASED':
            param = self.args.alpha
        elif self.args.method == 'LaplacianShot':
            param = self.args.lmd[0]
        elif self.args.method == 'Baseline':
            param = self.args.iter
        elif self.args.method == 'PT-MAP':
            param = self.args.alpha
        elif self.args.method == 'TIM-GD':
            param = self.args.loss_weights[1]
        elif self.args.method == 'ICI':
            param = self.args.d
        elif self.args.method == 'BDCSPN':
            param = self.args.temp
            
        self.logger.info('----- Final test results -----')
        
        for shot in self.args.shots:
            name_file = 'results_test/{}/{}/{}_shots{}.txt'.format(self.args.dataset, self.args.arch, self.args.method, shot)

            if not os.path.exists('results_test/{}/{}'.format(self.args.dataset, self.args.arch)):
                os.makedirs('results_test/{}/{}'.format(self.args.dataset, self.args.arch))
            if os.path.isfile(name_file) == True:
               f = open(name_file, 'a')
            else:
               f = open(name_file, 'w')
                
            f.write(str(self.args.n_ways)+'\t')
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(shot, self.args.number_tasks,
                                                                                   mean_accuracies[self.args.shots.index(shot)]))
            self.logger.info('{}-shot mean F1 score over {} tasks: {}'.format(shot, self.args.number_tasks,
                                                                                   mean_F1s[self.args.shots.index(shot)]))
            f.write(str(round(100*mean_accuracies[self.args.shots.index(shot)], 1)) +'\t' )
            f.write(str(round(100*mean_F1s[self.args.shots.index(shot)], 1)) +'\t' )

        f.close()
            
        
        #self.logger.info('----- Final test results -----')
        #for shot in self.args.shots:
        #    self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(shot, self.args.number_tasks,
        #                                                                           mean_accuracies[self.args.shots.index(shot)]))
        return mean_accuracies

    def get_method_builder(self, model):
        # Initialize method classifier builder
        method_info = {'model': model, 'device': self.device, 'log_file': self.log_file, 'args': self.args}
        if self.args.method == 'ALPHA-TIM':
            method_builder = ALPHA_TIM(**method_info)
        elif self.args.method == 'KM-UNBIASED':
            method_builder = KM_UNBIASED(**method_info)
        elif self.args.method == 'KM-BIASED':
            method_builder = KM_BIASED(**method_info)
        elif self.args.method == 'TIM-GD':
            method_builder = TIM_GD(**method_info)
        elif self.args.method == 'LaplacianShot':
            method_builder = LaplacianShot(**method_info)
        elif self.args.method == 'BDCSPN':
            method_builder = BDCSPN(**method_info)
        elif self.args.method == 'Baseline':
            method_builder = Baseline(**method_info)
        elif self.args.method == 'PT-MAP':
            method_builder = PT_MAP(**method_info)
        elif self.args.method == 'ICI':
            method_builder = ICI(**method_info)
        elif self.args.method == 'KM-UNBIASED-GD':
            method_builder = KM_UNBIASED_GD(**method_info)
        else:
            self.logger.exception("Method must be in ['KM_UNBIASED', 'KM_UNBIASED_GD', 'KM_BIASED', 'TIM_GD', 'ICI', 'ALPHA_TIM', 'LaplacianShot', 'BDCSPN', 'SimpleShot', 'Baseline', 'Baseline++', 'PT-MAP', 'Entropy_min']")
            raise ValueError("Method must be in ['KM_UNBIASED', 'KM_BIASED', 'TIM_GD', 'ICI', ALPHA_TIM', 'LaplacianShot', 'BDCSPN', 'SimpleShot', 'Baseline', 'Baseline++', 'PT-MAP', 'Entropy_min']")
        return method_builder