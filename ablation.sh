for n in 4; do \
	# Alpha-TIM
	python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml \
					--method_config config/dirichlet/methods_config/alpha_tim.yaml \
					--opts alpha_values "[8.0,8.0,8.0]" alpha_dirichlet 75 n_ways ${n} convergence_ablation True \
					shots "[20]" number_tasks 1 batch_size 1 iter 10000 lr_alpha_tim 0.001

	# Standard KM-unbiased
	python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml \
					--method_config config/dirichlet/methods_config/km_unbiased.yaml \
					--opts alpha 75 alpha_dirichlet 75 n_ways ${n} convergence_ablation True \
					shots "[20]" number_tasks 1 batch_size 1 iter 1000

	# Gradient-descent version
	python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml \
				--method_config config/dirichlet/methods_config/km_gd_unbiased.yaml \
					--opts alpha 75 alpha_dirichlet 75 n_ways ${n} convergence_ablation True \
					shots "[20]" number_tasks 1 batch_size 1 iter 1000 lr 0.0005


done
