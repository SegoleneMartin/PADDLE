for n in 2 3 4 5 6 7 8 9 10; do \
	# Standard KM-unbiased
	# python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml \
	# 				--method_config config/dirichlet/methods_config/km_unbiased.yaml \
	# 				--opts alpha 75 alpha_dirichlet 75 n_ways ${n}

	# Gradient-descent version
	python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml \
				--method_config config/dirichlet/methods_config/km_gd_unbiased.yaml \
				--opts alpha 75 alpha_dirichlet 75 n_ways ${n}
done
