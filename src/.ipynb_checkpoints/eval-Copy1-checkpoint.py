import numpy as np
from src.utils import wrap_tqdm, compute_confidence_interval, load_checkpoint, Logger, extract_mean_features
from src.methods.tim import ALPHA_TIM, TIM_GD
from src.methods.paddle import PADDLE
from src.methods.soft_km import SOFT_KM
from src.methods.laplacianshot import LaplacianShot
from src.methods.bdcspn import BDCSPN
from src.methods.simpleshot import SimpleShot
from src.methods.baseline import Baseline, Baseline_PlusPlus
from src.methods.pt_map import PT_MAP
from src.methods.protonet import ProtoNet
from src.methods.entropy_min import Entropy_min
from src.datasets import Tasks_Generator, CategoriesSampler, get_dataset, get_dataloader
import os

class Evaluator:
    def __init__(self, device, args, log_file):
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)

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

        train_set = get_dataset('train', args=self.args, **loader_info)
        dataset['train_loader'] = train_set

        support_set = get_dataset(self.args.used_set_support, args=self.args, **loader_info)
        dataset.update({'repr': support_set})
        query_set = get_dataset(self.args.used_set_query, args=self.args, **loader_info)
        dataset.update({'query': query_set})

        #test_set = get_dataset(self.args.used_set, args=self.args, **loader_info)
        #dataset.update({'test': test_set})

        train_loader = get_dataloader(sets=train_set, args=self.args)
        train_mean, _ = extract_mean_features(model=model,  train_loader=train_loader, args=self.args,
                                              logger=self.logger, device=self.device)

        results = []
        results_hungarian = []
        for shot in self.args.shots:
            sampler = CategoriesSampler(dataset['support'].labels, dataset['query'].labels, self.args.batch_size,
                                        self.args.k_eff, shot, self.args.n_query,
                                        self.args.sampling, self.args.alpha_dirichlet)

            #test_loader = get_dataloader(sets=dataset['test'], args=self.args,
             #                            sampler=sampler, pin_memory=False)
            test_loader = get_dataloader(sets=dataset['test'], args=self.args,
                                         sampler=sampler, pin_memory=False)
            task_generator = Tasks_Generator(k_eff=self.args.k_eff, shot=shot, n_query=self.args.n_query,        loader=test_loader_support, loader=test_loader_query, train_mean=train_mean, log_file=self.log_file)
            results_task = []
            results_task_hungarian = []
            for i in range(int(self.args.number_tasks/self.args.batch_size)):

                method = self.get_method_builder(model=model)

                tasks = task_generator.generate_tasks()

                # Run task
                logs = method.run_task(task_dic=tasks, shot=shot)
                #print(process.memory_info().rss)  # in bytes 

                acc_mean, acc_conf = compute_confidence_interval(logs['acc'][:, -1])
                acc_hungarian_mean, acc_hungarian_conf = compute_confidence_interval(logs['acc_hung'][:, -1])

                results_task.append(acc_mean)
                results_task_hungarian.append(acc_hungarian_mean)
                del method
                del tasks
            results.append(results_task)
            results_hungarian.append(results_task_hungarian)

        mean_accuracies = np.asarray(results).mean(1)
        mean_accuracies_hungarian = np.asarray(results_hungarian).mean(1)

        if self.args.method == 'ALPHA-TIM':
            param = self.args.alpha_values[1]
        elif self.args.method == 'PADDLE':
            param = self.args.alpha
        elif self.args.method == 'LaplacianShot':
            param = self.args.lmd[0]
            
        """
        name_file = 'test_params_tuned/{}_alpha{}_shots{}.txt'.format(self.args.method, self.args.alpha_dirichlet, self.args.shots[0])
        #name_file = 'params_tuning/params_tuning_{}.txt'.format(self.args.method)
        if os.path.isfile(name_file) == True:
            f = open(name_file, 'a')
            print('ok')
        else:
            f = open(name_file, 'w')
            print("not found", name_file, os.getcwd())
            
        f.write(str(param)+'\t')
        self.logger.info('----- Final test results -----')
        for shot in self.args.shots:
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(shot, self.args.number_tasks,
                                                                                   mean_accuracies[self.args.shots.index(shot)]))
            #self.logger.info('{}-shot mean hungarian accuracy over {} tasks: {}'.format(shot, self.args.number_tasks,
            #                                                                       mean_accuracies_hungarian[self.args.shots.index(shot)]))
            f.write(str(mean_accuracies[self.args.shots.index(shot)]) +'\t' )
        f.write('\n')
        f.close()
        """
        
        self.logger.info('----- Final test results -----')
        for shot in self.args.shots:
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(shot, self.args.number_tasks,
                                                                                   mean_accuracies[self.args.shots.index(shot)]))
        return mean_accuracies

    def get_method_builder(self, model):
        # Initialize method classifier builder
        method_info = {'model': model, 'device': self.device, 'log_file': self.log_file, 'args': self.args}
        if self.args.method == 'ALPHA-TIM':
            method_builder = ALPHA_TIM(**method_info)
        elif self.args.method == 'PADDLE':
            method_builder = PADDLE(**method_info)
        elif self.args.method == 'SOFT-KM':
            method_builder = SOFT_KM(**method_info)
        elif self.args.method == 'TIM-GD':
            method_builder = TIM_GD(**method_info)
        elif self.args.method == 'LaplacianShot':
            method_builder = LaplacianShot(**method_info)
        elif self.args.method == 'BDCSPN':
            method_builder = BDCSPN(**method_info)
        elif self.args.method == 'SimpleShot':
            method_builder = SimpleShot(**method_info)
        elif self.args.method == 'Baseline':
            method_builder = Baseline(**method_info)
        elif self.args.method == 'Baseline++':
            method_builder = Baseline_PlusPlus(**method_info)
        elif self.args.method == 'PT-MAP':
            method_builder = PT_MAP(**method_info)
        elif self.args.method == 'ProtoNet':
            method_builder = ProtoNet(**method_info)
        elif self.args.method == 'Entropy-min':
            method_builder = Entropy_min(**method_info)
        else:
            self.logger.exception("Method must be in ['PADDLE', 'SOFT_KM', 'TIM_GD', 'ALPHA_TIM', 'LaplacianShot', 'BDCSPN', 'SimpleShot', 'Baseline', 'Baseline++', 'PT-MAP', 'Entropy_min']")
            raise ValueError("Method must be in ['PADDLE', 'TIM_GD', 'ALPHA_TIM', 'LaplacianShot', 'BDCSPN', 'SimpleShot', 'Baseline', 'Baseline++', 'PT-MAP', 'Entropy_min']")
        return method_builder
