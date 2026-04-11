import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_ranking_plot(df: pd.DataFrame, output_file: str) -> None:
    title = "Policy rankings"
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.clear()

    order = list(dict.fromkeys(df["run"].tolist()))  # preserve insertion order
    palette = sns.color_palette("husl", n_colors=len(order) or 1)

    sns.lineplot(
        data=df,
        x="step",
        y="mu",
        hue="run",
        hue_order=order,
        linewidth=2.0,
        ax=ax,
        palette=palette,
    )

    z = 1.96  # approx 95% credible interval assuming normality
    for color, run_name in zip(palette, order):
        run_df = df[df["run"] == run_name]
        steps = run_df["step"].to_numpy()
        mu = run_df["mu"].to_numpy()
        sigma = run_df["sigma"].to_numpy()
        lower = mu - z * sigma
        upper = mu + z * sigma
        ax.fill_between(steps, lower, upper, color=color, alpha=0.2)

    ax.legend(title="Run", loc="best", fontsize=9)

    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Rating (mu)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    plt.savefig(output_file)


def main():
    df = pd.read_csv("./rankings.csv")
    save_ranking_plot(df, "./output.png")


if __name__ == "__main__":
    main()
