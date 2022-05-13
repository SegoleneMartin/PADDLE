for n in 2 3 4 5 6 7 8 9 10
do
for alpha in 1
do
for value in [9.0,9.0,9.0]
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/alpha_tim.yaml --opts alpha_values ${value} alpha_dirichlet ${alpha} n_ways ${n} 
done
for value in [1.0,0.3,1.0]
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/tim.yaml --opts loss_weights ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
for value in [0.7,0.7]
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/laplacianshot.yaml --opts lmd ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
for value in 100
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/baseline.yaml --opts iter ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
for value in 0.008
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/pt_map.yaml --opts alpha ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
for value in 75
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/km_unbiased.yaml --opts alpha ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
for value in 0
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/km.yaml --opts alpha ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
for value in 11
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/ici.yaml --opts d ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
for value in 9
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/bdcspn.yaml --opts temp ${value} alpha_dirichlet ${alpha} n_ways ${n}
done
done
done
