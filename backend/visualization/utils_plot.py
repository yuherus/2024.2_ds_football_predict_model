import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def save_fig(fig_dir, filename):
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}/{filename}")
    plt.clf()

def plot_label_distribution(df, result_col, fig_dir):
    sns.countplot(data=df, x=result_col, order=['Home Win', 'Draw', 'Away Win'])
    plt.title("Label Distribution (Win/Draw/Loss)")
    plt.xlabel("Match Result")
    plt.ylabel("Count")
    save_fig(fig_dir, "label_distribution.png")

def plot_numerical_distributions(df, numerical_features, fig_dir):
    for col in numerical_features:
        if col in df.columns:
            sns.histplot(df[col].dropna(), kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            save_fig(fig_dir, f"distribution_{col}.png")

def plot_correlation_matrix(df, numerical_features, fig_dir):
    corr = df[numerical_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    save_fig(fig_dir, "feature_correlation_matrix.png")

def plot_result_by_season(df, season_col, result_col, fig_dir):
    season_results = df.groupby([season_col, result_col]).size().unstack(fill_value=0)
    season_results.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title("Match Outcomes per Season")
    plt.xlabel("Season")
    plt.ylabel("Number of Matches")
    plt.xticks(rotation=45)
    plt.legend(title='Result')
    plt.tight_layout()
    save_fig(fig_dir, "results_by_season.png")

def plot_team_stats(df, team_col, stat_cols, fig_dir):
    for col in stat_cols:
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df, x=team_col, y=col, errorbar=None)
        plt.title(f"{col.replace('_', ' ').title()} per Team")
        plt.xticks(rotation=90)
        plt.tight_layout()
        save_fig(fig_dir, f"team_{col}.png")

def plot_squad_value_distribution(df, value_col, fig_dir):
    # Clean numeric value from string like '€15m', '£10K' etc.
    df['value_num'] = pd.to_numeric(
    df[value_col].str.replace(r'[€£mMbBkK]', '', regex=True),
    errors='coerce' 
)
    sns.histplot(df['value_num'].dropna(), bins=30, kde=True)
    plt.title("Squad Market Value Distribution")
    plt.xlabel("Value (in millions)")
    save_fig(fig_dir, "squad_value_distribution.png")

def plot_missing_values(df, fig_dir):
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing.index, y=missing.values)
    plt.xticks(rotation=90)
    plt.ylabel("Missing Ratio")
    plt.title("Missing Value Ratio per Column")
    plt.tight_layout()
    save_fig(fig_dir, "missing_values_ratio.png")
