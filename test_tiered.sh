for alpha in 1
do
# for value in [4.0,4.0,4.0] [5.0,5.0,5.0] [6.0,6.0,6.0] [7.0,7.0,7.0] [8.0,8.0,8.0] [9.0,9.0,9.0] [10.0,10.0,10.0] 
# do
# python3 -m main --base_config config/dirichlet/base_config/resnet18/tiered/base_config.yaml --method_config config/dirichlet/methods_config/alpha_tim.yaml --opts alpha_values ${value} alpha_dirichlet ${alpha}
# done
# for value in [1.0,0.6,1.0] [1.0,0.8,1.0] [1.0,1.2,1.0] [1.0,1.4,1.0] [1.0,1.6,1.0]
# do
# python3 -m main --base_config config/dirichlet/base_config/resnet18/tiered/base_config.yaml --method_config config/dirichlet/methods_config/tim.yaml --opts loss_weights ${value} alpha_dirichlet ${alpha}
# done
# for value in [0.4,0.7] [0.5,0.7] [0.6,0.7] [0.7,0.7] [0.8,0.7] [0.9,0.7] [1.0,0.7] [1.1,0.7] [1.2,0.7] [1.3,0.7]
# do
# python3 -m main --base_config config/dirichlet/base_config/resnet18/tiered/base_config.yaml --method_config config/dirichlet/methods_config/laplacianshot.yaml --opts lmd ${value} alpha_dirichlet ${alpha}
# done
# for value in 20 40 60 80 100
# do
# python3 -m main --base_config config/dirichlet/base_config/resnet18/tiered/base_config.yaml --method_config config/dirichlet/methods_config/baseline.yaml --opts iter ${value} alpha_dirichlet ${alpha}
# done
# for value in 0.008 0.02 0.05 0.10 0.15 0.20 0.25
# do
# python3 -m main --base_config config/dirichlet/base_config/resnet18/tiered/base_config.yaml --method_config config/dirichlet/methods_config/pt_map.yaml --opts alpha ${value} alpha_dirichlet ${alpha}
# done
# for value in 50 55 60 65 70 75 80
# do
# python3 -m main --base_config config/dirichlet/base_config/resnet18/tiered/base_config.yaml --method_config config/dirichlet/methods_config/km_unbiased.yaml --opts alpha ${value} alpha_dirichlet ${alpha}
# done
for value in 4 5 6 7 8 9
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/tiered/base_config.yaml --method_config config/dirichlet/methods_config/ici.yaml --opts d ${value} alpha_dirichlet ${alpha}
done
# for value in 6 9 12 15 18
# do
# python3 -m main --base_config config/dirichlet/base_config/resnet18/tiered/base_config.yaml --method_config config/dirichlet/methods_config/bdcspn.yaml --opts temp ${value} alpha_dirichlet ${alpha}
# done
for value in [1.0,0.2,1.0] [1.0,0.3,1.0] [1.0,0.4,1.0] [1.0,0.5,1.0] [1.0,0.6,1.0] 
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/tiered/base_config.yaml --method_config config/dirichlet/methods_config/tim.yaml --opts loss_weights ${value} alpha_dirichlet ${alpha}
done
done
