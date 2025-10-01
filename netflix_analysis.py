"""
netflix_analysis.py

Clean, runnable script extracted from the provided document.
Requires: pandas, numpy, matplotlib, seaborn

Usage:
    - Place netflix.csv in the same folder as this script or update CSV_PATH below.
    - Run: python netflix_analysis.py
    - In Jupyter/Colab, you can run cells selectively to view individual plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: improve default plot aesthetics
sns.set(style="whitegrid", rc={"figure.figsize": (10, 6)})

# -------------------------
# Configuration
# -------------------------
CSV_PATH = "netflix.csv"  # change if your file is elsewhere
SAVE_PLOTS = False        # set True to save plot images instead of showing interactively
PLOT_DIR = "plots"        # directory to save plots if SAVE_PLOTS = True

# -------------------------
# Utility: save or show plots
# -------------------------
import os
if SAVE_PLOTS and not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def show_or_save(fig, name):
    if SAVE_PLOTS:
        path = os.path.join(PLOT_DIR, name)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved plot: {path}")
    else:
        plt.show()

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(CSV_PATH)

# quick look
print("Dataset shape:", df.shape)
print(df.dtypes)
print(df.head(3))

# -------------------------
# Data type conversions & cleaning
# -------------------------
# Convert columns to categorical where appropriate
for col in ['type', 'country', 'rating']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Convert date_added to datetime with errors coerced (some NaN possible)
if 'date_added' in df.columns:
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Fill missing values for director and cast (categorical fields)
if 'director' in df.columns:
    df['director'] = df['director'].fillna('Unknown')
if 'cast' in df.columns:
    df['cast'] = df['cast'].fillna('Unknown')

# For other small nulls, you can inspect and decide — here we print counts
print("\nMissing values per column:")
print(df.isnull().sum())

# -------------------------
# Basic numeric summary
# -------------------------
if 'release_year' in df.columns:
    print("\nRelease year summary:")
    print(df['release_year'].describe())

# -------------------------
# Categorical summaries
# -------------------------
print("\nType counts:")
print(df['type'].value_counts(dropna=False))

if 'country' in df.columns:
    print("\nTop 10 countries:")
    print(df['country'].value_counts().head(10))

if 'rating' in df.columns:
    print("\nRating counts (top):")
    print(df['rating'].value_counts().head(10))

# -------------------------
# Univariate plots
# -------------------------
# 1) Movies vs TV Shows pie
if 'type' in df.columns:
    type_counts = df['type'].value_counts()
    fig = plt.figure(figsize=(6,6))
    plt.pie(type_counts.values, labels=type_counts.index, startangle=90, autopct='%1.1f%%')
    plt.title('Movies vs TV Shows')
    show_or_save(fig, "movies_vs_tv_pie.png")

# 2) Countplot for rating
if 'rating' in df.columns:
    fig = plt.figure(figsize=(12,5))
    order = df['rating'].value_counts().index
    sns.countplot(x='rating', data=df, order=order)
    plt.title("Count of Content by Rating")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    show_or_save(fig, "rating_countplot.png")

# 3) Distribution / histogram of release_year
if 'release_year' in df.columns:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    sns.histplot(df['release_year'], kde=True, bins=30, ax=axes[0])
    axes[0].set_title('Distribution of Release Years')
    axes[0].set_xlabel('Release Year')
    axes[0].set_ylabel('Density')

    axes[1].hist(df['release_year'].dropna(), bins=30, edgecolor='black')
    axes[1].set_title('Histogram of Release Years')
    axes[1].set_xlabel('Release Year')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    show_or_save(fig, "release_year_dist.png")

# Top 10 release years barplot
if 'release_year' in df.columns:
    top_years = df['release_year'].value_counts().nlargest(10)
    fig = plt.figure(figsize=(10,6))
    sns.barplot(x=top_years.index.astype(str), y=top_years.values)
    plt.title('Top 10 Most Frequent Release Years')
    plt.xlabel('Release Year')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    show_or_save(fig, "top10_release_years.png")

# -------------------------
# Top directors (excluding 'Unknown')
# -------------------------
if 'director' in df.columns:
    top_directors = df.loc[df['director'] != 'Unknown', 'director'].value_counts().head(10)
    if not top_directors.empty:
        fig = plt.figure(figsize=(12,8))
        sns.barplot(x=top_directors.values, y=top_directors.index)
        plt.title('Top 10 Most Frequent Directors on Netflix')
        plt.xlabel('No of Titles')
        plt.ylabel('Director')
        show_or_save(fig, "top10_directors.png")

# -------------------------
# Bivariate analysis: Type vs Rating
# -------------------------
if {'type', 'rating'}.issubset(df.columns):
    fig = plt.figure(figsize=(14,6))
    order = df['rating'].value_counts().index
    sns.countplot(x='rating', hue='type', data=df, order=order)
    plt.title('Count of Movies vs TV Shows by Rating')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Type')
    show_or_save(fig, "type_vs_rating.png")

# -------------------------
# Boxplot: Rating vs Release Year
# -------------------------
if {'rating', 'release_year'}.issubset(df.columns):
    fig = plt.figure(figsize=(14,8))
    # For better readability, keep only ratings that occur frequently
    rating_counts = df['rating'].value_counts()
    frequent_ratings = rating_counts[rating_counts > 50].index.tolist()
    subset = df[df['rating'].isin(frequent_ratings)]
    sns.boxplot(x='rating', y='release_year', data=subset)
    plt.title('Distribution of Release Years by Rating (frequent ratings)')
    plt.xlabel('Rating')
    plt.ylabel('Release Year')
    plt.xticks(rotation=45)
    show_or_save(fig, "boxplot_rating_releaseyear.png")

# -------------------------
# Correlation (numeric) & Pairplot
# -------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns:", numeric_cols)

if len(numeric_cols) > 0:
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation matrix (numeric columns)')
    show_or_save(fig, "correlation_matrix.png")

    # Pairplot only if there are >=2 numeric columns
    if len(numeric_cols) >= 2:
        # NOTE: pairplot can be slow for large datasets; sample if needed
        sample = df[numeric_cols].sample(1000, random_state=42) if len(df) > 1000 else df[numeric_cols]
        sns.pairplot(sample)
        plt.suptitle('Pairplot for Numeric Variables', y=1.02)
        show_or_save(plt.gcf(), "pairplot_numeric.png")
else:
    print("No numeric columns (other than possibly release_year). Pairplot/heatmap omitted or not informative.")

# -------------------------
# Simple business-insight style counts (text output)
# -------------------------
print("\n--- Business-style summaries ---")
if 'type' in df.columns:
    print("Type split (Movies vs TV Shows):")
    print(df['type'].value_counts(normalize=False))

if 'country' in df.columns:
    print("\nTop 10 countries producing content:")
    print(df['country'].value_counts().head(10))

if 'rating' in df.columns:
    print("\nTop ratings (counts):")
    print(df['rating'].value_counts().head(10))

# Example: % of TV-MA + TV-14
if 'rating' in df.columns:
    total = len(df)
    for r in ['TV-MA', 'TV-14']:
        if r in df['rating'].cat.categories:
            print(f"\nCount of {r}: {df['rating'].value_counts().get(r,0)} ({df['rating'].value_counts(normalize=True).get(r,0)*100:.1f}%)")

print("\nScript complete.")
