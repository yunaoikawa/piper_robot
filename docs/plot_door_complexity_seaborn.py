#!/usr/bin/env python3
"""Dot-only Door code/API-size and pose-error figures. Requires seaborn==0.13.2.

Uses tracked audit reports only; does not access hardware or private logs.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ASSETS = Path(__file__).resolve().parent / 'assets/code_as_learning_machine'


def load_data():
    complexity = json.loads((ASSETS / 'door_code_complexity_report.json').read_text())
    distances = json.loads((ASSETS / 'door_approach_distance_report.json').read_text())
    by_stage = {row['historical_stage']: row for row in distances['configurations']}
    rows = []
    for row in complexity['configurations']:
        distance = by_stage[row['historical_stage']]
        rows.append({'stage': row['display_stage'],
                     'code_lines': row['code_lines_without_comments_or_docstrings'],
                     'external_api_count': row['external_api_name_count'],
                     'distance_mm': distance['distance_mm'],
                     'orientation_deg': distance['orientation_difference_deg']})
    return pd.DataFrame(rows)


def save_figure(fig, stem, output_dir=ASSETS):
    for extension in ('png', 'svg'):
        path = output_dir / f'{stem}.{extension}'
        metadata = {'Date': None} if extension == 'svg' else {}
        fig.savefig(path, dpi=220, metadata=metadata)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


def plot():
    data = load_data()
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
        for row in data.to_dict('records'):
            ax.annotate(row['stage'], (row['code_lines'], row['distance_mm']),
                        xytext=offsets[row['stage']], textcoords='offset points',
                        ha='center', fontsize=16, color='#17324D')
        save_figure(fig, 'door_code_size_distance_dotplot')
    return fig, ax


def plot_api_errors(*, save=True):
    """Preserve actual coordinates; D3/D4 coincide in both panels."""
    data = load_data()
    with sns.axes_style('whitegrid'), sns.plotting_context('talk'), plt.rc_context({
        'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
        'svg.hashsalt': 'door_api_pose_errors_dotplot_v1',
    }):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), layout='constrained')
        settings = (
            ('distance_mm', 'EE position', 'Estimated position error (mm)', (10, 20), range(10, 21, 2)),
            ('orientation_deg', 'EE orientation', 'Orientation difference (deg)', (0, 12), range(0, 13, 2)),
        )
        for ax, (metric, title, ylabel, ylim, yticks) in zip(axes, settings):
            sns.scatterplot(data=data, x='external_api_count', y=metric,
                            color='#2B6CB0', s=110, linewidth=0.8,
                            edgecolor='white', legend=False, ax=ax)
            ax.set(xlim=(25, 62), ylim=ylim, title=title,
                   xlabel='External API count\n(distinct call targets)', ylabel=ylabel)
            ax.set_xticks([30, 40, 50, 60])
            ax.set_yticks(yticks)
            ax.grid(color='#D7E1EA', linewidth=0.7)
            sns.despine(ax=ax)
            # Group annotations only, never merge or jitter the observations.
            for (x, y), group in data.groupby(['external_api_count', metric], sort=False):
                label = ' / '.join(group['stage'])
                offset = {'D1': (-27, -21), 'D2': (27, 15)}.get(label, (0, 17))
                ax.annotate(label, (x, y), xytext=offset,
                            textcoords='offset points', ha='center',
                            fontsize=16, color='#17324D')
        if save:
            save_figure(fig, 'door_api_pose_errors_dotplot')
    return fig, axes


if __name__ == '__main__':
    figure, _ = plot()
    plt.close(figure)
    figure, _ = plot_api_errors()
    plt.close(figure)
