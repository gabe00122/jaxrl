import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union, NamedTuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class LeagueEntry(NamedTuple):
    """Coerced record describing a league rating sample."""

    name: str
    step: int
    mu: float
    sigma: float


class LiveRankingsPlot:
    """
    Live-updating probability density plot of rankings between runs, directly from a list of
    `LeagueEntry` records.
    - One curve per run token (`LeagueEntry.name`), derived from its rating mean and variance
    - X = training step; Y = rating mean with shaded credible interval from the reported sigma
    - Call `update(entries)` whenever ratings change (e.g., each round or every N rounds)
    - Use `update_from_csv(path)` to render a saved CSV snapshot of league entries
    """
    def __init__(
        self,
        title: str = "Policy rankings"
    ):
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(9, 6))

    def update(self, league: Iterable[LeagueEntry]) -> None:
        self.ax.clear()

        plot_records: List[Dict[str, float]] = []
        for name, step, mu, sigma in league:
            plot_records.append(
                {
                    "step": step,
                    "mu": mu,
                    "sigma": sigma,
                    "run": name,
                }
            )

        plot_df = pd.DataFrame(plot_records)
        order = list(dict.fromkeys(plot_df["run"].tolist()))  # preserve insertion order
        palette = sns.color_palette("husl", n_colors=len(order) or 1)

        sns.lineplot(
            data=plot_df,
            x="step",
            y="mu",
            hue="run",
            hue_order=order,
            linewidth=2.0,
            ax=self.ax,
            palette=palette,
        )

        z = 1.96  # approx 95% credible interval assuming normality
        for color, run_name in zip(palette, order):
            run_df = plot_df[plot_df["run"] == run_name]
            steps = run_df["step"].to_numpy()
            mu = run_df["mu"].to_numpy()
            sigma = run_df["sigma"].to_numpy()
            lower = mu - z * sigma
            upper = mu + z * sigma
            self.ax.fill_between(steps, lower, upper, color=color, alpha=0.2)

        self.ax.legend(title="Run", loc="best", fontsize=9)

        self.ax.set_xlabel("Checkpoint step")
        self.ax.set_ylabel("Rating (mu)")
        self.ax.set_title(self.title)
        self.ax.grid(True, alpha=0.2)
        self.fig.tight_layout()

        self.fig.canvas.draw_idle()

    def update_from_csv(self, csv_path: Union[str, Path]) -> None:
        """Convenience helper to load a CSV and render its entries."""

        entries = load_league_csv(csv_path)
        self.update(entries)


REQUIRED_LEAGUE_COLUMNS = ("name", "step", "mu", "sigma")


def save_league_csv(entries: Sequence[LeagueEntry], path: Union[str, Path]) -> Path:
    """Persist a sequence of league entries to CSV and return the file path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entries = list(entries)

    if entries:
        data = [entry._asdict() for entry in entries]
    else:
        data = {column: [] for column in REQUIRED_LEAGUE_COLUMNS}

    df = pd.DataFrame(data, columns=REQUIRED_LEAGUE_COLUMNS)
    df.to_csv(path, index=False)
    return path


def load_league_csv(path: Union[str, Path]) -> List[LeagueEntry]:
    """Load league entries from CSV into the common `LeagueEntry` format."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"League CSV not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []

    missing = set(REQUIRED_LEAGUE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in league CSV: {sorted(missing)}")

    entries: List[LeagueEntry] = []
    for row in df.itertuples(index=False):
        entries.append(
            LeagueEntry(
                name=str(getattr(row, "name")),
                step=int(getattr(row, "step")),
                mu=float(getattr(row, "mu")),
                sigma=float(getattr(row, "sigma")),
            )
        )

    return entries


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for plotting league ratings from a CSV file."""

    parser = argparse.ArgumentParser(description="Render a probability density plot for league ratings.")
    parser.add_argument(
        "csv",
        nargs="?",
        default="rankings.csv",
        help="Path to the ratings CSV (default: ratings.csv)",
    )
    parser.add_argument(
        "--title",
        default="Policy rankings",
        help="Plot title to display.",
    )
    args = parser.parse_args(argv)

    plot = LiveRankingsPlot(title=args.title)
    plot.update_from_csv(args.csv)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
