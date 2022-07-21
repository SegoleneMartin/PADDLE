sampling='dirichlet'
shots=[5]
tune_parameters=True
arch='resnet18'
dataset='mini'
used_set_support='val'
used_set_query='val'

### TIM ###
for value in [1.0,0.1,1.0] [1.0,0.15,1.0] [1.0,0.2,1.0] [1.0,0.25,1.0] [1.0,0.3,1.0] \
[1.0,0.35,1.0] [1.0,0.4,1.0] [1.0,0.45,1.0] [1.0,0.5,1.0] [1.0,0.5,1.0]
do
python -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} shots ${shots} \
tune_parameters ${tune_parameters} method tim loss_weights ${value}
done
### ALPHA-TIM ###
for value in [5.0,5.0,5.0] [5.5,5.5,5.5] [6.0,6.0,6.0] [6.5,6.5,6.5] [7.0,7.0,7.0] \
[7.5,7.5,7.5] [8.0,8.0,8.0] [8.5,8.5,8.5] [9.0,9.0,9.0] [9.5,9.5,9.5]
do
python -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} shots ${shots} \
tune_parameters ${tune_parameters} method alpha_tim alpha_values ${value}
done
### BDSCPN ###
for value in for value in 5 8 10 15 20 30 50 70 100 120
doget_log_file
python -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} shots ${shots} \
tune_parameters ${tune_parameters} method bdcspn temp ${value}
done
### LAPLACIAN SHOT ###
for value in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
do
python -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} shots ${shots} \
tune_parameters ${tune_parameters} method laplacianshot lmd ${value}
done
### PT-MAP ###
for value in 0.002 0.005 0.008 0.01 0.02 0.05 0.1 0.2 0.3 0.4
do
python -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} shots ${shots} \
tune_parameters ${tune_parameters} method pt_map alpha ${value}
done
### ICI ###
for value in 2 3 4 5 6 7 8 9 10 11
do
python -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} shots ${shots} \
tune_parameters ${tune_parameters} method ici d ${value}
done
### PADDLE ###
for value in 15 30 40 50 60 70 75 80 85 90
do
python -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} shots ${shots} \
tune_parameters ${tune_parameters} method paddle alpha ${value}
done

