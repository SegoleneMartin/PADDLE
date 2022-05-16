import argparse
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path
import os
from collections import defaultdict

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
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--archs", nargs='+', type=str)
    parser.add_argument("--datasets", nargs='+', type=str)
    parser.add_argument("--shots", nargs='+', type=str)
    args = parser.parse_args()
    return args


list_methods = ['KM-UNBIASED', 'KM-BIASED', 'TIM-GD', 'ALPHA-TIM', 'Baseline', 'BDCSPN']
list_name = ['Ours', 'K-Means', 'TIM-GD', r'$\alpha$-TIM', 'Baseline', 'BDCSPN']
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


def main(args):
    n_datasets = len(args.datasets)
    n_archs = len(args.archs)
    assert n_datasets and n_archs
    fig, axes = plt.subplots(figsize=(2.5 * len(args.shots), 2 * n_datasets * n_archs),
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
                ax = axes[i * n_archs + j, k]
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                for method_index, method in enumerate(list_methods):
                    file_path = folder / f'{method}_alpha1_shots{shot}.txt'
                    if file_path.exists():
                        tab = np.genfromtxt(file_path, dtype=float, delimiter='\t')
                        list_classes = tab[:, 0]
                        list_acc = tab[:, 1]
                        ax.plot(list_classes, list_acc,
                                marker=markers[method_index],
                                label=list_name[method_index],
                                color=colors[method_index],
                                linewidth=1.5 if method_index == 0 else 0.5)
                        if max(list_acc) > max_:
                            max_ = max(list_acc)
                        if min(list_acc) < min_:
                            min_ = min(list_acc)
                if i == 0 and j == 0:
                    ax.set_title(rf"{shot} shots")
                if i * n_archs + j == (n_datasets * n_archs - 1):
                    ax.set_xlabel(rf'Number of effective classes $K_{{eff}}$')
                if k == 0:
                    ax.set_ylabel('Accuracy')
                    ax.text(-0.5, 0.5, rf"{pretty[arch]}" + "\n" + rf"{pretty[dataset]}",
                            rotation=90, ha='center', va='center',
                            transform=ax.transAxes)
            # print(min_, max_)
            # print(i, j, k)
            # ax.set_ylim(min_ - 0.1, max_ + 0.1)
            # ax.set_yticks(np.linspace(min_, max_, 5))


    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="center",
               bbox_to_anchor=[0.53, 1.05],  # bottom-right
               ncol=6,
               frameon=False)
    os.makedirs(args.out_dir, exist_ok=True)
    outfilename = os.path.join(args.out_dir, f"{'-'.join(args.datasets)}_{'-'.join(args.archs)}_{'-'.join(args.shots)}.pdf")
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
            }
        )
    main(args)