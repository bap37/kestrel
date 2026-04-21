import copy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    get_args,
)
from warnings import warn

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure, FigureBase
from matplotlib.patches import Rectangle
from scipy.stats import binom, gaussian_kde, iqr
from torch import Tensor


##########################
# Simulations Diagnostics Plots
##########################

def plot_surviving_priors(theta,priors,labels,figname):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=13, ncols=1, figsize=(6, 26), sharex=False)

    for x in range(13):
        ax = axes[x]
        ax.hist(theta[:, x].detach().cpu().numpy(), histtype="step", label="Valid")
        ax.hist(priors[:, x].detach().cpu().numpy(), histtype="step", label="Full")
        ax.set_title(labels[x])
        
    axes[0].legend(["Valid", "Full"])
    plt.tight_layout()
    plt.savefig(figname, bbox_inches="tight")



##########################
# Diagnostic plots largely lifted from SBI diagnostics 
# Redesigned to fit my plot style and make saving easier. 
##########################

def plot_tarp(ecp, alpha, savename):

    fig = plt.figure(figsize=(6, 6))
    ax: Axes = plt.gca()

    ax.plot(alpha, ecp, color="blue", label="TARP")
    ax.plot(alpha, alpha, color="black", linestyle="--", label="ideal")
    ax.set_xlabel(r"Credibility Level $\alpha$")
    ax.set_ylabel(r"Expected Coverage Probability")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    plt.savefig(savename, bbox_inches="tight")
    return fig, ax

def plot_loss(inference, savename):
    
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6))
    ax: Axes = plt.gca()

    train_losses = inference._summary["training_loss"]
    val_losses = inference._summary["validation_loss"]

    plt.plot(train_losses, label="train", lw=3,)
    plt.plot(val_losses, label="validation", lw=3, ls='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(frameon=False, labelcolor="linecolor", fontsize=14)

    ax.spines[['bottom', 'left']].set_color('dimgrey')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_lw(2)
    ax.tick_params(axis="both", which="both", colors="dimgrey", labelsize=15)

    plt.savefig(savename, bbox_inches="tight")


#################################
# Posterior calibration plots
#################################

def sbc_rank_plot(
    ranks: Union[Tensor, np.ndarray, List[Tensor], List[np.ndarray]],
    num_posterior_samples: int,
    num_bins: Optional[int] = None,
    plot_type: str = "cdf",
    parameter_labels: Optional[List[str]] = None,
    ranks_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot simulation-based calibration ranks as empirical CDFs or histograms.

    Additional options can be passed via the kwargs argument, see _sbc_rank_plot.

    Args:
        ranks: Tensor of ranks to be plotted shape (num_sbc_runs, num_parameters), or
            list of Tensors when comparing several sets of ranks, e.g., set of ranks
            obtained from different methods.
        num_bins: number of bins used for binning the ranks, default is
            num_sbc_runs / 20.
        plot_type: type of SBC plot, histograms ("hist") or empirical cdfs ("cdf").
        parameter_labels: list of labels for each parameter dimension.
        ranks_labels: list of labels for each set of ranks.
        colors: list of colors for each parameter dimension, or each set of ranks.

    Returns:
        fig, ax: figure and axis objects.

    """

    return _sbc_rank_plot(
        ranks,
        num_posterior_samples,
        num_bins,
        plot_type,
        parameter_labels,
        ranks_labels,
        colors,
        fig=fig,
        ax=ax,
        figsize=figsize,
        **kwargs,
    )

def _sbc_rank_plot(
    ranks: Union[Tensor, np.ndarray, List[Tensor], List[np.ndarray]],
    num_posterior_samples: int,
    num_bins: Optional[int] = None,
    plot_type: str = "cdf",
    parameter_labels: Optional[List[str]] = None,
    ranks_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    num_repeats: int = 50,
    line_alpha: float = 0.8,
    show_uniform_region: bool = True,
    uniform_region_alpha: float = 0.3,
    xlim_offset_factor: float = 0.1,
    num_cols: int = 4,
    params_in_subplots: bool = False,
    show_ylabel: bool = False,
    sharey: bool = False,
    fig: Optional[FigureBase] = None,
    legend_kwargs: Optional[Dict] = None,
    ax=None,
    figsize: Optional[tuple] = None,
) -> Tuple[Figure, Axes]:

    # -------------------------
    # Input handling
    # -------------------------
    if isinstance(ranks, (Tensor, np.ndarray)):
        ranks_list = [ranks]
    else:
        assert isinstance(ranks, List)
        ranks_list = ranks

    for idx, rank in enumerate(ranks_list):
        assert isinstance(rank, (Tensor, np.ndarray))
        if isinstance(rank, Tensor):
            ranks_list[idx] = rank.numpy()

    plot_types = ["hist", "cdf"]
    assert plot_type in plot_types

    if legend_kwargs is None:
        legend_kwargs = dict()

    num_sbc_runs, num_parameters = ranks_list[0].shape
    num_ranks = len(ranks_list)

    if num_ranks > 1 or plot_type == "hist":
        params_in_subplots = True

    for r in ranks_list:
        assert r.shape == ranks_list[0].shape

    if parameter_labels is None:
        parameter_labels = [f"dim {i + 1}" for i in range(num_parameters)]

    if ranks_labels is None:
        #ranks_labels = [f"rank set {i + 1}" for i in range(num_ranks)]
        ranks_labels = ['Ranks' for i in range(num_ranks)]

    if num_bins is None:
        num_bins = num_sbc_runs // 20

    # -------------------------
    # Subplot case
    # -------------------------
    if params_in_subplots:

        # +1 subplot for legend
        num_plots = num_parameters + 1
        num_cols_eff = min(num_cols, num_plots)
        num_rows = int(np.ceil(num_plots / num_cols_eff))

        if figsize is None:
            figsize = (num_cols_eff * 4, num_rows * 4)

        if fig is None or ax is None:
            fig, ax = plt.subplots(
                num_rows,
                num_cols_eff,
                figsize=figsize,
                sharey=sharey,
            )

        axes = np.atleast_2d(ax).flatten()

        legend_handles = []
        legend_labels = []

        # -------------------------
        # Plot each parameter
        # -------------------------
        for jj in range(num_parameters):
            current_ax = axes[jj]
            plt.sca(current_ax)

            for ii, ranki in enumerate(ranks_list):

                if plot_type == "cdf":
                    _plot_ranks_as_cdf(
                        ranki[:, jj],
                        num_posterior_samples,
                        ranks_label=None,
                        color=f"C{ii}" if colors is None else colors[ii],
                        xlabel=f"{parameter_labels[jj]}",
                        show_ylabel=(jj == 0),
                        alpha=line_alpha,
                    )

                    if jj == 0:
                        line = current_ax.lines[-1]
                        legend_handles.append(line)
                        legend_labels.append(ranks_labels[ii])

                    if ii == 0 and show_uniform_region:
                        _plot_cdf_region_expected_under_uniformity(
                            num_sbc_runs,
                            alpha=uniform_region_alpha,
                        )

                elif plot_type == "hist":
                    _plot_ranks_as_hist(
                        ranki[:, jj],
                        num_bins,
                        num_posterior_samples,
                        ranks_label=None,
                        color="firebrick" if colors is None else colors[ii],
                        xlabel=f"{parameter_labels[jj]}",
                        show_ylabel=show_ylabel,
                        alpha=line_alpha,
                        xlim_offset_factor=xlim_offset_factor,
                    )

                    if jj == 0:
                        patch = plt.Rectangle(
                            (0, 0), 1, 1,
                            color="firebrick" if colors is None else colors[ii],
                            alpha=line_alpha,
                        )
                        legend_handles.append(patch)
                        legend_labels.append(ranks_labels[ii])

                    if ii == 0:
                        _plot_hist_region_expected_under_uniformity(
                            num_sbc_runs,
                            num_bins,
                            num_posterior_samples,
                            alpha=uniform_region_alpha,
                        )

                else:
                    raise ValueError(
                        f"plot_type {plot_type} not defined, use one in {plot_types}"
                    )

            #current_ax.set_title(parameter_labels[jj])

        # -------------------------
        # Legend subplot
        # -------------------------
        legend_ax = axes[num_parameters]
        legend_ax.axis("off")

        legend_ax.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="center",
            **legend_kwargs,
        )

        # Hide unused axes
        for k in range(num_parameters + 1, len(axes)):
            axes[k].axis("off")

    # -------------------------
    # Single panel case
    # -------------------------
    else:
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize or (8, 5))

        plt.sca(ax)

        legend_handles = []
        legend_labels = []

        ranki = ranks_list[0]

        for jj in range(num_parameters):
            _plot_ranks_as_cdf(
                ranki[:, jj],
                num_posterior_samples,
                ranks_label=None,
                color=f"C{jj}" if colors is None else colors[jj],
                xlabel="Posterior Rank",
                show_ylabel=(jj == num_parameters - 1),
                alpha=line_alpha,
            )

            line = ax.lines[-1]
            legend_handles.append(line)
            legend_labels.append(parameter_labels[jj])

        if show_uniform_region:
            _plot_cdf_region_expected_under_uniformity(
                num_sbc_runs,
                alpha=uniform_region_alpha,
            )

        ax.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            **legend_kwargs,
        )

    return fig, ax

def _plot_ranks_as_hist(
    ranks: np.ndarray,
    num_bins: int,
    num_posterior_samples: int,
    ranks_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: str = "firebrick",
    alpha: float = 0.8,
    show_ylabel: bool = False,
    num_ticks: int = 3,
    xlim_offset_factor: float = 0.1,
) -> None:
    """Plot ranks as histograms on the current axis.

    Args:
        ranks: SBC ranks in shape (num_sbc_runs, )
        num_bins: number of bins for the histogram, recommendation is num_sbc_runs / 20.
        num_posteriors_samples: number of posterior samples used for ranking.
        ranks_label: label for the ranks, e.g., when comparing ranks of different
            methods.
        xlabel: label for the current parameter.
        color: histogram color, default from Talts et al.
        alpha: histogram transparency.
        show_ylabel: whether to show y-label "counts".
        show_legend: whether to show the legend, e.g., when comparing multiple ranks.
        num_ticks: number of ticks on the x-axis.
        xlim_offset_factor: factor for empty space left and right of the histogram.
        legend_kwargs: kwargs for the legend.
    """
    xlim_offset = 0#int(num_posterior_samples * xlim_offset_factor)
    plt.hist(
        ranks,
        bins=num_bins,
        label=ranks_label,
        color=color,
        alpha=alpha,
    )

    if show_ylabel:
        plt.ylabel("counts")
    else:
        plt.yticks([])

    plt.xlim(-xlim_offset, num_posterior_samples + xlim_offset)
    plt.xticks(np.linspace(0, num_posterior_samples, num_ticks))
    plt.xlabel("Posterior Rank" if xlabel is None else xlabel)

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_ranks_as_cdf(
    ranks: np.ndarray,
    num_posterior_samples: int,
    ranks_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: Optional[str] = None,
    alpha: float = 0.8,
    show_ylabel: bool = True,
    num_ticks: int = 5,
) -> None:
    """Plot ranks as a true empirical CDF (continuous, no binning)."""

    # Normalize ranks to [0, 1]
    ranks = np.asarray(ranks) / num_posterior_samples

    # Sort for ECDF
    x = np.sort(ranks)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.plot(
        x,
        y,
        label=ranks_label,
        color=color,
        alpha=alpha,
    )

    if show_ylabel:
        plt.ylabel("Empirical CDF")
        plt.yticks(np.linspace(0, 1, 5))
    else:
        plt.yticks(np.linspace(0, 1, 5), [])

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks(np.linspace(0, 1, num_ticks))
    plt.xlabel("Posterior Rank" if xlabel is None else xlabel)

def _plot_cdf_region_expected_under_uniformity(
    num_sbc_runs: int,
    alpha: float = 0.2,
    color: str = "gray",
    num_points: int = 500,
) -> None:
    """Continuous confidence band for ECDF under uniformity."""

    x = np.linspace(0, 1, num_points)

    lower = binom.ppf(0.005, num_sbc_runs, x) / num_sbc_runs
    upper = binom.ppf(0.995, num_sbc_runs, x) / num_sbc_runs

    plt.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=alpha,
        label="Expected Under Uniformity",
    )


def _plot_hist_region_expected_under_uniformity(
    num_sbc_runs: int,
    num_bins: int,
    num_posterior_samples: int,
    alpha: float = 0.2,
    color: str = "gray",
) -> None:
    """Plot region of empirical cdfs expected under uniformity."""

    lower = binom(num_sbc_runs, p=1 / (num_bins + 1)).ppf(0.005)
    upper = binom(num_sbc_runs, p=1 / (num_bins + 1)).ppf(0.995)

    # Plot grey area with expected ECDF.
    plt.fill_between(
        x=np.linspace(0, num_posterior_samples, num_bins),
        y1=np.repeat(lower, num_bins),
        y2=np.repeat(upper, num_bins),  # pyright: ignore[reportArgumentType]
        color=color,
        alpha=alpha,
        label="Expected Under Uniformity",
    )


