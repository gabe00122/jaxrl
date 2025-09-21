from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class LiveRankingsPlot:
    """
    Live-updating line plot of rankings between runs, directly from `league`.
    - One line per run token (`PolicyRecord.name`)
    - X = checkpoint step, Y = mu or (mu - 3*sigma)
    - Call `update(league)` whenever ratings change (e.g., each round or every N rounds)
    """
    def __init__(
        self,
        conservative: bool = True,
        smooth: Optional[int] = None,   # e.g., 3 for moving avg per run
        pause_sec: float = 0.1,
        title: str = "Policy rankings (live)"
    ):
        self.conservative = conservative
        self.smooth = smooth if (smooth is not None and smooth > 1) else None
        self.pause_sec = pause_sec
        self.title = title

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.lines: Dict[str, any] = {}   # run_name -> Line2D

    @staticmethod
    def _score(mu: np.ndarray, sigma: np.ndarray, conservative: bool) -> np.ndarray:
        return mu - 3.0 * sigma if conservative else mu

    def _group_league(self, league) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns {run_name: (steps_sorted, mu_sorted, sigma_sorted)}.
        """
        rows: List[Tuple[str, int, float, float]] = [
            (p.name, int(p.step), float(p.rating.mu), float(p.rating.sigma)) for p in league
        ]
        if not rows:
            return {}

        df = pd.DataFrame(rows, columns=["name", "step", "mu", "sigma"])
        # If duplicates exist per (name, step), keep last (latest rating)
        df = df.sort_values(["name", "step"]).groupby(["name", "step"], as_index=False).tail(1)

        groups: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for run_name, g in df.groupby("name"):
            g = g.sort_values("step")
            steps = g["step"].to_numpy()
            mu = g["mu"].to_numpy()
            sigma = g["sigma"].to_numpy()
            groups[run_name] = (steps, mu, sigma)
        return groups

    def _maybe_smooth(self, y: np.ndarray) -> np.ndarray:
        if self.smooth is None or len(y) <= 1:
            return y
        k = int(self.smooth)
        if k <= 1:
            return y
        # simple causal moving average
        kernel = np.ones(k, dtype=float) / k
        # pad to keep same length and avoid phase shift
        y_pad = np.pad(y, (k-1, 0), mode="edge")
        return np.convolve(y_pad, kernel, mode="valid")

    def update(self, league) -> None:
        groups = self._group_league(league)

        self.ax.clear()
        any_data = False

        for run_name, (steps, mu, sigma) in groups.items():
            y = self._score(mu, sigma, self.conservative)
            y = self._maybe_smooth(y)

            line, = self.ax.plot(steps, y, marker="o", linewidth=1.5, label=str(run_name))
            self.lines[run_name] = line
            any_data = True

        ylabel = "Conservative rating (mu - 3·sigma)" if self.conservative else "Rating (mu)"
        self.ax.set_xlabel("Checkpoint step")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(self.title)
        self.ax.grid(True, alpha=0.3)
        if any_data:
            self.ax.legend(loc="best", fontsize=9)
        self.fig.tight_layout()

        # redraw without blocking the loop
        self.fig.canvas.draw_idle()
        plt.pause(self.pause_sec)
