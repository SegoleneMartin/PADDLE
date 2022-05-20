python -m src.plots --use_latex True \
					--root results_test \
					--out_dir plots \
					--archs resnet18 \
					--datasets mini tiered \
					--ncols 10 \
					--shots 5 10 20

python -m src.plots --use_latex True \
					--root results_test \
					--out_dir plots \
					--archs resnet18 \
					--datasets inatural \
					--ncols 2 \
					--shots 1
