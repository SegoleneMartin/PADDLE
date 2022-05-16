for n in 5; do \
	# Standard KM-unbiased
	python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml \
					--method_config config/dirichlet/methods_config/km_unbiased.yaml \
					--opts alpha 75 alpha_dirichlet 75 n_ways ${n}

	# Gradient-descent version
	# python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml \
	# 			--method_config config/dirichlet/methods_config/km_gd_unbiased.yaml \
	# 			--opts alpha 75 n_ways ${n}

	# python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml \
	# --method_config config/dirichlet/methods_config/km.yaml \
	# --opts alpha 75 n_ways ${n} num_classes_test 5 
done
