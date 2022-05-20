arch=resnet18

# for n in 2 3 4 5 6 7 8 9 10
#3 4 5 6 7 8 9 10
for n in 5
do
for alpha in 1
do
# for value in 75
# do
# 	python3 -m main --base_config config/dirichlet/base_config/${arch}/tiered/base_config.yaml --method_config config/dirichlet/methods_config/km_unbiased.yaml --opts alpha ${value} alpha_dirichlet ${alpha} n_ways ${n}
# done
for value in [8.0,8.0,8.0]
do
	python3 -m main --base_config config/dirichlet/base_config/${arch}/tiered/base_config.yaml --method_config config/dirichlet/methods_config/alpha_tim.yaml --opts alpha_values ${value} alpha_dirichlet ${alpha} n_ways ${n} 
done
for value in [1.0,0.4,1.0]
do
	python3 -m main --base_config config/dirichlet/base_config/${arch}/tiered/base_config.yaml --method_config config/dirichlet/methods_config/tim.yaml --opts loss_weights ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
for value in [0.8,0.8]
do
	python3 -m main --base_config config/dirichlet/base_config/${arch}/tiered/base_config.yaml --method_config config/dirichlet/methods_config/laplacianshot.yaml --opts lmd ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
for value in 100
do
	python3 -m main --base_config config/dirichlet/base_config/${arch}/tiered/base_config.yaml --method_config config/dirichlet/methods_config/baseline.yaml --opts iter ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
for value in 0.05
do
	python3 -m main --base_config config/dirichlet/base_config/${arch}/tiered/base_config.yaml --method_config config/dirichlet/methods_config/pt_map.yaml --opts alpha ${value} alpha_dirichlet ${alpha} n_ways ${n} batch_size 50
done
for value in 0
do
	python3 -m main --base_config config/dirichlet/base_config/${arch}/tiered/base_config.yaml --method_config config/dirichlet/methods_config/km.yaml --opts alpha ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
# for value in 11
# do
# 	python3 -m main --base_config config/dirichlet/base_config/resnet18/tiered/base_config.yaml --method_config config/dirichlet/methods_config/ici.yaml --opts d ${value} alpha_dirichlet ${alpha} n_ways ${n}
# done
for value in 12
do
	python3 -m main --base_config config/dirichlet/base_config/${arch}/tiered/base_config.yaml --method_config config/dirichlet/methods_config/bdcspn.yaml --opts temp ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
done
done
