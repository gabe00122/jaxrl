
# live_training_visualizer.py
# A lightweight Rich-based live visualizer for ML training loops.
# - Tracks arbitrary metrics you update (e.g., loss, accuracy, reward, lr, etc.)
# - Displays steps/sec and grad-steps/sec (15s & 60s windows)
# - Shows totals, uptime, and simple stats (avg/min/max) for each metric
#
# Usage:
#     from live_training_visualizer import TrainingVisualizer
#
#     with TrainingVisualizer(metrics=["loss", "accuracy", "reward"], total_steps=100000) as viz:
#         for step in range(100000):
#             # ... compute your metrics here ...
#             viz.step()                   # call on each environment step / training step
#             if (step + 1) % 4 == 0:
#                 viz.grad_step()          # call when you complete a gradient update
#             viz.update({
#                 "loss": loss_value,
#                 "accuracy": acc_value,
#                 "reward": reward_value,
#             })
#
# Works in any terminal. Press Ctrl+C to stop.
#
# Notes:
# - "steps" is whatever you define (env steps, tokens, or optimizer steps). If you want both,
#   call viz.step() for the faster inner loop and viz.grad_step() when you actually step the optimizer.
# - Metric values can be floats or ints. We maintain a rolling average over the last N values per metric.

from __future__ import annotations

import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from rich.console import Console, RenderableType, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.rule import Rule


@dataclass
class ThroughputCounter:
    window_seconds: float = 60.0
    stamps: deque = field(default_factory=lambda: deque(maxlen=200_000))

    def record(self, n: int = 1, now: Optional[float] = None):
        t = time.time() if now is None else now
        for _ in range(n):
            self.stamps.append(t)

    def rate(self, window: float) -> float:
        now = time.time()
        w = min(window, self.window_seconds)
        # drop old stamps
        cutoff = now - w
        while self.stamps and self.stamps[0] < cutoff:
            self.stamps.popleft()
        return len(self.stamps) / max(w, 1e-9)


@dataclass
class MetricBuffer:
    name: str
    maxlen: int = 500
    values: deque = field(default_factory=lambda: deque(maxlen=500))

    def update(self, x: float):
        self.values.append(float(x))

    @property
    def latest(self) -> Optional[float]:
        return self.values[-1] if self.values else None

    def avg(self, k: int = 100) -> Optional[float]:
        if not self.values:
            return None
        if k <= 0:
            k = 1
        data = list(self.values)[-k:]
        return sum(data) / len(data) if data else None

    def min(self) -> Optional[float]:
        return float('nan') if not self.values else float(__import__('builtins').min(self.values))

    def max(self) -> Optional[float]:
        return float('nan') if not self.values else float(__import__('builtins').max(self.values))


class TrainingVisualizer:
    def __init__(
        self,
        metrics: Iterable[str] = (),
        total_steps: Optional[int] = None,
        title: str = "Live Training",
        refresh_hz: float = 10.0,
        metric_history: int = 500,
        console: Optional[Console] = None,
    ):
        self.console = console or Console()
        self.title = title
        self.total_steps = total_steps
        self.refresh_hz = float(refresh_hz)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # counters
        self._steps_total = 0
        self._grad_steps_total = 0
        self._steps_tp = ThroughputCounter(window_seconds=300.0)
        self._grad_tp = ThroughputCounter(window_seconds=300.0)

        # metrics
        self._metric_buffers: Dict[str, MetricBuffer] = {
            m: MetricBuffer(m, maxlen=metric_history) for m in metrics
        }

        # guard
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None

    # ---- public API ----
    def step(self, n: int = 1):
        now = time.time()
        with self._lock:
            self._steps_total += n
            self._steps_tp.record(n=n, now=now)

    def grad_step(self, n: int = 1):
        now = time.time()
        with self._lock:
            self._grad_steps_total += n
            self._grad_tp.record(n=n, now=now)

    def update(self, metrics: Dict[str, float]):
        with self._lock:
            for k, v in metrics.items():
                if k not in self._metric_buffers:
                    self._metric_buffers[k] = MetricBuffer(k)
                self._metric_buffers[k].update(float(v))

    def start(self):
        if self._thread is not None:
            return
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ---- rendering ----
    def _render_header(self) -> RenderableType:
        now = time.time()
        uptime = 0.0 if self._start_time is None else (now - self._start_time)
        h = Text(self.title, style="bold")
        sub = Text(f"  uptime: {uptime:,.1f}s", style="dim")
        return Group(Rule(), Text.assemble(h, sub), Rule())

    def _render_throughput_panel(self) -> Panel:
        with self._lock:
            steps_total = self._steps_total
            grad_total = self._grad_steps_total
            steps_r15 = self._steps_tp.rate(15.0)
            steps_r60 = self._steps_tp.rate(60.0)
            grad_r15 = self._grad_tp.rate(15.0)
            grad_r60 = self._grad_tp.rate(60.0)
            pct = None
            if self.total_steps is not None and self.total_steps > 0:
                pct = min(100.0, 100.0 * steps_total / self.total_steps)

        t = Table.grid(pad_edge=True, expand=True)
        t.add_column(justify="left")
        t.add_column(justify="right")
        t.add_row("Steps total", f"{steps_total:,}")
        if self.total_steps:
            t.add_row("Progress", f"{pct:5.1f}% ({steps_total:,} / {self.total_steps:,})")
        t.add_row("Grad steps total", f"{grad_total:,}")
        t.add_row("Steps/sec (15s)", f"{steps_r15:,.2f}")
        t.add_row("Steps/sec (60s)", f"{steps_r60:,.2f}")
        t.add_row("Grad steps/sec (15s)", f"{grad_r15:,.2f}")
        t.add_row("Grad steps/sec (60s)", f"{grad_r60:,.2f}")
        return Panel(t, title="Throughput", border_style="cyan")

    def _render_metrics_panel(self) -> Panel:
        table = Table(expand=True, show_lines=False)
        table.add_column("Metric", style="bold", no_wrap=True)
        table.add_column("Latest", justify="right")
        table.add_column("Avg(100)", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        with self._lock:
            items = sorted(self._metric_buffers.items(), key=lambda kv: kv[0].lower())
            for name, buf in items:
                latest = buf.latest
                avg100 = buf.avg(100)
                mn = buf.min()
                mx = buf.max()
                table.add_row(
                    name,
                    "-" if latest is None else f"{latest: .6g}",
                    "-" if avg100 is None else f"{avg100: .6g}",
                    "-" if not buf.values else f"{mn: .6g}",
                    "-" if not buf.values else f"{mx: .6g}",
                )

        return Panel(table, title="Metrics", border_style="magenta")

    def _build_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
        )
        layout["body"].split_row(
            Layout(name="throughput", ratio=1),
            Layout(name="metrics", ratio=2),
        )
        layout["header"].update(self._render_header())
        layout["throughput"].update(self._render_throughput_panel())
        layout["metrics"].update(self._render_metrics_panel())
        return layout

    def _run(self):
        refresh = max(0.05, 1.0 / self.refresh_hz)
        layout = self._build_layout()
        with Live(layout, console=self.console, refresh_per_second=self.refresh_hz, screen=True):
            while not self._stop.is_set():
                # update panels
                layout["header"].update(self._render_header())
                layout["throughput"].update(self._render_throughput_panel())
                layout["metrics"].update(self._render_metrics_panel())
                time.sleep(refresh)


# --- Demo ---
# Run this file directly to see a simulated training loop.
if __name__ == "__main__":
    import math
    import random

    total_steps = 10_000
    grad_every = 8  # simulate 1 grad step per 8 env steps

    with TrainingVisualizer(
        metrics=["loss", "accuracy", "reward", "lr"],
        total_steps=total_steps,
        title="Demo: Live Training (Rich)",
        refresh_hz=12,
    ) as viz:
        try:
            t0 = time.time()
            for step in range(total_steps):
                # Simulate some work
                time.sleep(0.005 + random.random() * 0.003)

                # Step & maybe grad step
                viz.step()
                if (step + 1) % grad_every == 0:
                    viz.grad_step()

                # Fake metrics (just to demo movement)
                # loss decays, accuracy increases, reward oscillates, lr cosine decays
                progress = (step + 1) / total_steps
                loss = 2.0 * math.exp(-3.0 * progress) + 0.02 * random.random()
                acc = 0.5 + 0.5 * progress + 0.05 * (random.random() - 0.5)
                reward = 100.0 * math.sin(progress * 8.0) + 200.0 + 5.0 * random.random()
                lr = 3e-4 * (0.5 * (1 + math.cos(math.pi * progress)))

                viz.update({
                    "loss": loss,
                    "accuracy": max(0.0, min(1.0, acc)),
                    "reward": reward,
                    "lr": lr,
                })
        except KeyboardInterrupt:
            pass
        finally:
            # ensure the UI cleans up nicely
            pass
