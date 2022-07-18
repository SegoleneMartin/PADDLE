import argparse
from loguru import logger
from pathlib import Path
import numpy as np

list_methods = ['PADDLE', 'SOFT-KM', 'PT-MAP', 'ICI', \
                'LaplacianShot', 'TIM-GD', 'ALPHA-TIM', 'Baseline', 'BDCSPN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot')
    parser.add_argument("--root", type=str)
    parser.add_argument("--archs", nargs='+', type=str)
    parser.add_argument("--datasets", nargs='+', type=str)
    parser.add_argument("--shots", nargs='+', type=str)
    parser.add_argument("--effective_classes", type=int, default=5)
    args = parser.parse_args()
    return args


def benchmark_plot(args):

    for j, arch in enumerate(args.archs):
        logger.info(f"=========== {arch} =========== ")
        for method_index, method in enumerate(list_methods):
            perf_list = []
            method_complete = True
            for i, dataset in enumerate(args.datasets):
                folder = Path(args.root) / dataset / arch
                for k, shot in enumerate(args.shots):
                    file_path = folder / f'{method}_alpha1_shots{shot}.txt'
                    if file_path.exists():
                        tab = np.genfromtxt(file_path, dtype=float, delimiter='\t')
                        list_classes = tab[..., 0]
                        list_acc = tab[..., 1]
                        if len(list_acc.shape):
                            relevant_index = np.where(list_classes == args.effective_classes)[0]
                            if len(relevant_index):
                                relevant_index = relevant_index[0]
                            else:
                                method_complete = False
                            relevant_acc = list_acc[relevant_index]
                        else:
                            if list_classes != args.effective_classes:
                                method_complete = False
                                break
                            relevant_acc = list_acc
                        perf_list.append(str(relevant_acc))
            if method_complete:
                logger.info(f"Method {method}: {' & '.join(perf_list)} ")


if __name__ == '__main__':
    args = parse_args()
    benchmark_plot(args)