import argparse
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path
import os
from collections import defaultdict
from pathlib import Path
from os.path import join as ospjoin
import pickle 
from scipy.interpolate import make_interp_spline, BSpline


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot')
    parser.add_argument("--use_latex", type=str2bool, default=True)
    parser.add_argument("--root", type=str)
    parser.add_argument("--action", type=str, default="benchmark_plot")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--criterion_threshold", type=float, default=1e-6)
    parser.add_argument("--archs", nargs='+', type=str)
    parser.add_argument("--datasets", nargs='+', type=str)
    parser.add_argument("--shots", nargs='+', type=str)
    parser.add_argument("--ncols", type=int, default=10)
    args = parser.parse_args()
    return args


list_methods = ['PADDLE', 'TIM-GD', 'ALPHA-TIM', 'Baseline', 'BDCSPN', 'SOFT_KM', 'LaplacianShot']
list_name = [r'\textsc{PADDLE}', r'\textsc{TIM}', r'$\alpha$-\textsc{TIM}',
             'Baseline', r'\textsc{BDCSPN}', r'\textsc{PGD}', 'LaplacianShot']
markers = ["^", ".", "v", "1", "p", "*", "X", "d", "P", "<", ">"]
colors = ["#f02d22",
"#90813d",
"#8463cc",
"#b2b03c",
"#c361aa",
"#57ab67",
"#6691ce",
"#cd6c39"]
pretty = defaultdict(str)
pretty['mini'] = r"\textit{mini}-ImageNet"
pretty['tiered'] = r"\textit{tiered}-ImageNet"

pretty["resnet18"] = "ResNet 18"
pretty["wideres"] = "WRN 28-10"

blue = '#2CBDFE'
pink = '#F3A0F2'


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def convergence_plot(args):
    for i, dataset in enumerate(args.datasets):
        for j, arch in enumerate(args.archs):
            for k, shot in enumerate(args.shots):
                folder = ospjoin(args.root)
                _ = plt.Figure(figsize=(1, 1), dpi=300)
                ax = plt.gca()
                methods = [x[:-4] for x in os.listdir(folder)]
                logger.info(methods)
                for method in methods:
                    method_index = list_methods.index(method)
                    u = np.loadtxt(ospjoin(folder, f'{method}.txt'))
                    print(u)
                    with open(ospjoin(folder, method, f'.txt'), 'rb') as f:
                        x = pickle.load(f)['mean']
                        x = np.cumsum(x)
                    criterion_path = ospjoin(folder, method, f'criterion_{shot}.pkl')
                    if os.path.exists(criterion_path):
                        criterion_defined = True
                        with open(criterion_path, 'rb') as f:
                            y = pickle.load(f)['mean']
                    else:
                        break
                        logger.warning(f"Criterion not defined for {method}")
                        criterion_defined = False
                    with open(ospjoin(folder, method, f'full_acc_{shot}.pkl'), 'rb') as f:
                        acc = pickle.load(f)['mean']

                    # Criterion axis 
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    msg = [str(method)]
                    if criterion_defined:
                        index_convergence = np.where(y < args.criterion_threshold)[0]
                        if len(index_convergence):
                            index_convergence = index_convergence[0]
                            time_to_convergence = x[index_convergence]
                            msg.append(f"Time to convergence = {time_to_convergence}")
                            msg.append(f"Acc {np.round(acc[index_convergence], 2)}")
                        else:
                            msg.append(f"Acc {np.round(acc[-1], 2)}")

                    logger.info('\t'.join(msg))

                    if criterion_defined:
                        n = 50
                        ysmooth = moving_average(y, n)
                        # print(x)
                        ax.plot(x[:-n+1], ysmooth,
                                label=list_name[method_index],
                                color=colors[method_index],
                                marker=markers[method_index],
                                markersize=10,
                                linewidth=3,
                                markevery=100)

                    # ax.fill_between(x, criterion['mean'] - criterion['std'], 
                    #                 criterion['mean'] + criterion['std'],
                    #                 color=colors[method_index], alpha=0.4)
                    ax.set_ylabel(r"$\| \boldsymbol{W}^{(\ell+1)} - \boldsymbol{W}^{(\ell)} \|^2$")
                    ax.set_yscale('log')
                    ax.set_xscale('log')
                    ax.set_xlabel("Elapsed time (s)")
                    # ax.set_ylim(7e-8, 1e-1)


                    # color = pink
                    # if 'GD' in method:
                    #     logger.info(acc['mean'])
                    #     ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                    #     ax2.set_ylabel('Accuracy', color=pink)  # we already handled the x-label with ax1
                    #     ax2.plot(x, acc['mean'],
                    #              label=list_name[method_index],
                    #              color=color)
                    #     # ax2.fill_between(x, acc['mean'] - acc['std'], 
                    #     #                  acc['mean'] + acc['std'],
                    #     #                  color=color, alpha=0.4)
                    #     ax2.tick_params(axis='y', labelcolor=color)
                os.makedirs(args.out_dir, exist_ok=True)
                outfilename = ospjoin(args.out_dir,
                                      f"convergence_{dataset}_{arch}_{shot}.pdf")
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(),
                           loc="center",
                           bbox_to_anchor=[1.3, 0.5],  # bottom-right
                           ncol=1,
                           frameon=False)
                plt.savefig(outfilename, bbox_inches="tight")
                logger.info(f"Saved plot at {outfilename}")


def benchmark_plot(args):
    n_datasets = len(args.datasets)
    n_archs = len(args.archs)
    assert n_datasets and n_archs
    if len(args.shots) == 'None': # for inatural
        args.shots = [1]
    fig, axes = plt.subplots(figsize=(8 * len(args.shots), 6 * n_datasets * n_archs),
                             ncols=len(args.shots),
                             nrows=n_datasets * n_archs,
                             dpi=300,
                             sharex=True,
                             )
    for i, dataset in enumerate(args.datasets):
        for j, arch in enumerate(args.archs):
            folder = Path(args.root) / dataset / arch
            min_ = 100.0
            max_ = 0.
            for k, shot in enumerate(args.shots):
                if isinstance(axes, np.ndarray):
                    if n_datasets * n_archs > 1:
                        ax = axes[i * n_archs + j, k]
                    else:
                        ax = axes[k]
                else:
                    ax = axes
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                for method_index, method in enumerate(list_methods):
                    file_path = folder / f'{method}.txt'
                    if file_path.exists():
                        logger.warning(file_path)
                        tab = np.genfromtxt(file_path, dtype=float, delimiter='\t')
                        list_classes = tab[:, 0]
                        list_acc = tab[:, k + 1]
                        keep_index = list_classes <= 10
                        ax.plot(list_classes[keep_index], list_acc[keep_index],
                                marker=markers[method_index],
                                label=list_name[method_index],
                                color=colors[method_index],
                                linewidth=5 if method_index == 0 else 3.5,
                                markersize=15,
                                )
                        if max(list_acc) > max_:
                            max_ = max(list_acc)
                        if min(list_acc) < min_:
                            min_ = min(list_acc)
                # if i == 0 and j == 0:
                #     ax.set_title(rf"{shot} shots")
                if i * n_archs + j == (n_datasets * n_archs - 1):
                    ax.set_xlabel(rf'Number of effective classes $K_{{eff}}$')
                # if k == 0:
                #     ax.set_ylabel('Accuracy')
                #     ax.text(-0.5, 0.5, rf"{pretty[arch]}" + "\n" + rf"{pretty[dataset]}",
                #             rotation=90, ha='center', va='center',
                #             transform=ax.transAxes)
            ax.set_xticks(list_classes[keep_index])

    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="center",
               bbox_to_anchor=[0.53, 1.1],  # bottom-right
               ncol=args.ncols,
               frameon=False)
    os.makedirs(args.out_dir, exist_ok=True)
    outfilename = ospjoin(args.out_dir, f"{'-'.join(args.datasets)}_{'-'.join(args.archs)}_{'-'.join(args.shots)}.pdf")
    plt.savefig(outfilename, bbox_inches="tight")
    logger.info(f"Saved plot at {outfilename}")


if __name__ == "__main__":
    args = parse_args()
    if args.use_latex:
        logger.info("Activating latex")
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"],
                "font.size": 30
            }
        )
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        # \usepackage{amsmath,bm}
    if args.action == 'benchmark_plot':
        benchmark_plot(args)
    else:
        convergence_plot(args)