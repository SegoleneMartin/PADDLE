for alpha in 0.1
do
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/alpha_tim.yaml --opts alpha_values [5.0,18.0,5.0] alpha_dirichlet ${alpha} shots [1]
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/alpha_tim.yaml --opts alpha_values [5.0,19.0,5.0] alpha_dirichlet ${alpha} shots [5]
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/alpha_tim.yaml --opts alpha_values [5.0,19.0,5.0] alpha_dirichlet ${alpha} shots [10]
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/alpha_tim.yaml --opts alpha_values [5.0,19.0,5.0] alpha_dirichlet ${alpha} shots [20]


python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/laplacianshot.yaml --opts lmd [1.3,0.7] alpha_dirichlet ${alpha} shots [1]
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/laplacianshot.yaml --opts lmd [1.3,0.7] alpha_dirichlet ${alpha} shots [5]
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/laplacianshot.yaml --opts lmd [1.3,0.7] alpha_dirichlet ${alpha} shots [10]
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/laplacianshot.yaml --opts lmd [1.3,0.7] alpha_dirichlet ${alpha} shots [20]


python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/km_unbiased.yaml --opts alpha 100 alpha_dirichlet ${alpha} shots [1]
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/km_unbiased.yaml --opts alpha 105 alpha_dirichlet ${alpha} shots [5]
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/km_unbiased.yaml --opts alpha 105 alpha_dirichlet ${alpha} shots [10]
python3 -m main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/km_unbiased.yaml --opts alpha 105 alpha_dirichlet ${alpha} shots [20]
done
