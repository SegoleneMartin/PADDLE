SERVER_IP=mboudiaf@narval.computecanada.ca
SERVER_PATH=~/scratch/sego/data


for dataset in mini_imagenet tiered_imagenet meta_inat; do \
	rsync -avm data/${dataset}.tar.gz ${SERVER_IP}:${SERVER_PATH}/data/ ;\
done