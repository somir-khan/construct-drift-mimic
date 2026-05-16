import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV = os.path.abspath(os.path.join(_DIR, '..', '..', 'outputs', 'surface_features_by_period.csv'))

plt.rcParams.update({
    'font.family':       'sans-serif',
    'axes.labelsize':    10,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
})

if not os.path.exists(_CSV):
    raise FileNotFoundError(f"Expected CSV not found: {_CSV}")

df_raw = pd.read_csv(_CSV)
df_raw = df_raw.rename(columns={'period_label': 'period', 'n_notes': 'n'})

print("Loaded CSV:")
print(df_raw.to_string())
print()

df = df_raw[~df_raw['period'].str.contains('2020')].copy()

period_order = ['MIMIC-III', '2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019']
df = df.set_index('period').loc[period_order].reset_index()

assert len(df) == 5, f"Expected 5 rows after filtering, got {len(df)}"

label_map = {
    'MIMIC-III':   'MIMIC-III\n(2001\u201312)',
    '2008 - 2010': 'MIMIC-IV\n2008\u201310',
    '2011 - 2013': 'MIMIC-IV\n2011\u201313',
    '2014 - 2016': 'MIMIC-IV\n2014\u201316',
    '2017 - 2019': 'MIMIC-IV\n2017\u201319',
}
windows = [label_map[p] for p in df['period']]
x = np.arange(len(windows))


def _draw_bracket(ax, x_start=2.5, x_end=4.5, label='Analysis windows',
                  y_bracket=-0.26, y_text=-0.34):
    """Draw a |-| bracket below the x-axis in data-x / axes-fraction-y space."""
    transform = ax.get_xaxis_transform()
    ax.annotate(
        '',
        xy=(x_end, y_bracket), xytext=(x_start, y_bracket),
        xycoords=transform, textcoords=transform,
        arrowprops=dict(arrowstyle='|-|', color='#b8860b', lw=1.0,
                        mutation_scale=5),
        annotation_clip=False,
    )
    t = ax.text(
        (x_start + x_end) / 2, y_text, label,
        transform=transform,
        fontsize=7.5, ha='center', va='top', color='#b8860b',
    )
    t.set_clip_on(False)


def _format_panel(ax, label, windows, label_x=-0.13):
    _x = np.arange(len(windows))
    ax.set_xticks(_x)
    ax.set_xticklabels(windows, fontsize=8, rotation=15, ha='right')
    ax.axvspan(2.5, 4.5, alpha=0.07, color='gold', zorder=0)
    _draw_bracket(ax)
    ax.text(label_x, 1.04, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')


fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

bars = ax_a.bar(x, df['mean_length'], color='#4878CF', alpha=0.85, width=0.6)
ax_a.set_ylim(bottom=8000)
ax_a.set_ylabel('Mean length (characters)')
for bar, v in zip(bars, df['mean_length']):
    ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
              f'{round(v):,}', fontsize=7.5, ha='center', va='bottom', color='#333333')
_format_panel(ax_a, '(a)', windows)

bars = ax_b.bar(x, df['section_rate'], color='#6ACC65', alpha=0.85, width=0.6)
ax_b.set_ylim(0.70, 1.01)
ax_b.set_ylabel('Fraction of notes')
for bar, v in zip(bars, df['section_rate']):
    ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
              f'{v:.1%}', fontsize=7.5, ha='center', va='bottom', color='#333333')
ax_b.axhline(df['section_rate'].iloc[0], color='gray', linewidth=1,
             linestyle='--', alpha=0.6, label='MIMIC-III baseline')
ax_b.legend(fontsize=7.5, loc='lower right', framealpha=0.8)
_format_panel(ax_b, '(b)', windows)

ax_c.plot(x, df['jaccard'], color='#D65F5F', marker='o',
          linewidth=2, markersize=6, zorder=3)
ax_c.fill_between(x, df['jaccard'], alpha=0.12, color='#D65F5F')
ax_c.set_ylim(0.60, 1.05)
ax_c.set_ylabel('Jaccard similarity\nvs. MIMIC-III baseline', labelpad=4)
for xi, v in zip(x, df['jaccard']):
    ax_c.text(xi, v + 0.018, f'{v:.3f}', fontsize=7.5, ha='center', color='#D65F5F')

_format_panel(ax_c, '(c)', windows)

ax_d.set_facecolor('#f5f5f5')
bars = ax_d.bar(x, df['numeric_density'], color='#999999', alpha=0.75, width=0.6)
ax_d.set_ylim(0.085, 0.105)
ax_d.set_ylabel('Proportion of numeric tokens')
for bar, v in zip(bars, df['numeric_density']):
    ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
              f'{v:.4f}', fontsize=7.5, ha='center', va='bottom', color='#555555')
ax_d.axhline(df['numeric_density'].iloc[0], color='#555555', linewidth=1,
             linestyle='--', alpha=0.7)

_format_panel(ax_d, '(d)', windows, label_x=-0.22)

plt.tight_layout()
fig.subplots_adjust(hspace=0.58, wspace=0.42)

pdf_path = os.path.join(_DIR, 'fig3_surface_features.pdf')
png_path = os.path.join(_DIR, 'fig3_surface_features.png')
eps_path = os.path.join(_DIR, 'fig3_surface_features.eps')
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(eps_path, format='eps', bbox_inches='tight')
print("Saved fig3_surface_features.pdf / .png / .eps")

print("\nData used:")
print(df[['period', 'n', 'mean_length', 'section_rate', 'jaccard',
          'numeric_density', 'phi_density']].to_string(index=False))
plt.close()