
for dataset in mini_imagenet tiered_imagenet meta_inat; do \
	tar -czvf  data/${dataset}.tar.gz -C data/ ${dataset} ;\
done ;\