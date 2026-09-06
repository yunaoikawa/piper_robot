#!/usr/bin/env python3
"""Dot-only Door code-size/distance figure. Requires seaborn==0.13.2.

Uses tracked audit reports only; does not access hardware or private logs.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ASSETS = Path(__file__).resolve().parent / 'assets/code_as_learning_machine'


def plot():
    complexity = json.loads((ASSETS / 'door_code_complexity_report.json').read_text())
    distances = json.loads((ASSETS / 'door_approach_distance_report.json').read_text())
    by_stage = {row['historical_stage']: row for row in distances['configurations']}
    rows = []
    for row in complexity['configurations']:
        distance = by_stage[row['historical_stage']]
        rows.append({'stage': row['display_stage'],
                     'code_lines': row['code_lines_without_comments_or_docstrings'],
                     'distance_mm': distance['distance_mm']})
    data = pd.DataFrame(rows)
    with sns.axes_style('whitegrid'), sns.plotting_context('talk'), plt.rc_context({
        'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
        'svg.hashsalt': 'door_code_size_distance_dotplot_v1',
    }):
        fig, ax = plt.subplots(figsize=(9, 5.8), layout='constrained')
        sns.scatterplot(data=data, x='code_lines', y='distance_mm',
                        color='#2B6CB0', s=110, linewidth=0.8,
                        edgecolor='white', legend=False, ax=ax)
        ax.set(xlim=(950, 3400), ylim=(10, 20),
               xlabel='Code size (Python lines)',
               ylabel='Estimated EE-position distance (mm)')
        ax.set_yticks(range(10, 21, 2))
        ax.set_xticks(range(1000, 3500, 500))
        ax.grid(color='#D7E1EA', linewidth=0.7)
        sns.despine(ax=ax)
        offsets = {'D1': (-24, 15), 'D2': (25, 15),
                   'D3': (-24, 15), 'D4': (24, -24)}
        for row in rows:
            ax.annotate(row['stage'], (row['code_lines'], row['distance_mm']),
                        xytext=offsets[row['stage']], textcoords='offset points',
                        ha='center', fontsize=16, color='#17324D')
        for extension in ('png', 'svg'):
            path = ASSETS / f'door_code_size_distance_dotplot.{extension}'
            metadata = {'Date': None} if extension == 'svg' else {}
            fig.savefig(path, dpi=220, metadata=metadata)
            if extension == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    return fig, ax


if __name__ == '__main__':
    figure, _ = plot()
    plt.close(figure)
