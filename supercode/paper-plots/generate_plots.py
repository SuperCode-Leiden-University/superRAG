import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import pprint
import numpy as np

# ---------------------------------------------------------------------------------------- #
# data in readable format
benchmark_results = {
    "Qwen2.5-Coder-3B": {
        "baseline llm stats": {  # median values and std
            "llm runtime (s)": (142.06, 20.46),
            #"llm cputime (s)": (4159.48, 484.89),
            "llm energy (Wh)": (5.92, 0.70),
            #"llm emissions (g CO2eq)": (1.59, 0.19),
        },
        "benchmark llm stats": {  # median values and std
            "llm runtime (s)": (295.09, 32.52),
            #"llm cputime (s)": (5109.90, 1147.88),
            "llm energy (Wh)": (10.28, 1.24),
            #"llm emissions (g CO2eq)": (2.75, 0.33),
        },
        "baseline score": { # average values and std
            "val pass@1 (%) ": (43.29, 4.68),
            "test pass@1 (%)": (32.56, 2.33),
        },
        "benchmark score": {
            "val pass@1 (%) ": (41.22, 1.66),
            "test pass@1 (%)": (27.80, 0.83),
        },
    },
    "Qwen2.5-Coder-7B": {
        "baseline llm stats": {  # values and std
            "llm runtime (s)": (170.70, 16.51),
            #"llm cputime (s)": (6269.61, 622.32),
            "llm energy (Wh)": (8.73, 1.18),
            #"llm emissions (g CO2eq)": (2.34, 0.32),
        },
        "benchmark llm stats": {  # values and std
            "llm runtime (s)": (345.82, 30.24),
            #"llm cputime (s)": (7145.77, 1393.01),
            "llm energy (Wh)": (13.89, 2.17),
            #"llm emissions (g CO2eq)": (3.72, 0.58),
        },
        "baseline score": {
            "val pass@1 (%) ": (41.46, 3.49),
            "test pass@1 (%)": (26.10, 2.45),
        },
        "benchmark score": {
            "val pass@1 (%) ": (61.71, 1.61),
            "test pass@1 (%)": (41.95, 1.24),
        },
    },
    "Qwen2.5-Coder-14B": {
        "baseline llm stats": {  # values and std
            "llm runtime (s)": (439.83, 39.49),
            #"llm cputime (s)": (12510.36, 1123.37),
            "llm energy (Wh)": (22.56, 2.09),
            #"llm emissions (g CO2eq)": (6.04, 0.56),
        },
        "benchmark llm stats": {  # values and std
            "llm runtime (s)": (519.34, 72.95),
            #"llm cputime (s)": (16748.24, 2699.28),
            "llm energy (Wh)": (27.80, 3.99),
            #"llm emissions (g CO2eq)": (7.44, 1.07),
        },
        "baseline score": {
            "val pass@1 (%) ": (68.41, 2.48),
            "test pass@1 (%)": (48.54, 1.75),
        },
        "benchmark score": {
            "val pass@1 (%) ": (56.34, 1.99),
            "test pass@1 (%)": (37.20, 1.93),
        },
    },
}

# ---------------------------------------------------------------------------------------- #
# convert data in a more convenient format
plot_vals = {"model_names": [], }
# {'model_names': [...],
#  'llm_stats': {'llm_runtime (s)':     {'vals': [...], 'stds': [...]},
#                'llm_cputime (s)':     {'vals': [...], 'stds': [...]},
#                'llm_energy (Wh)':     {'vals': [...], 'stds': [...]},
#                'llm_co2eq (g CO2eq)': {'vals': [...], 'stds': [...]} },
#  'baseline':  {'comp_pass_1 (%)':     {'vals': [...], 'stds': [...]},
#                'test_pass_1 (%)':     {'vals': [...], 'stds': [...]} },
#  'benchmark': {'comp_pass_1 (%)':     {'vals': [...], 'stds': [...],
#                'test_pass_1 (%)':     {'vals': [...], 'stds': [...]} },
#  'gain':      {'comp_pass_1 (%)':     {'vals': [...], 'stds': [...]},
#                'test_pass_1 (%)':     {'vals': [...], 'stds': [...]} }}

for m, model in enumerate(benchmark_results.keys()):
    # print(f"m = {m}")
    plot_vals["model_names"].append(
        model
        #r"\begin{flushright} " + model + r" \end{flushright}"
    )

    for (group_key, group) in benchmark_results[model].items():
        if m == 0: plot_vals[group_key] = {}

        for (key, val_std) in group.items():
            if m == 0: plot_vals[group_key][key] = {"vals": [], "stds": []}
            plot_vals[group_key][key]["vals"].append(val_std[0])
            plot_vals[group_key][key]["stds"].append(val_std[1])

    if m == 0: plot_vals["gain"] = {}
    for k, key in enumerate(plot_vals["baseline score"].keys()):
        # print(f"k = {k}, key = {key}")
        plot_vals["gain"][key] = {"vals": [], "stds": []}
        plot_vals["gain"][key]["vals"] = np.subtract(
            plot_vals["benchmark score"][key]["vals"],
            plot_vals["baseline score"][key]["vals"]
        )
# pprint.pp(plot_vals)

# ---------------------------------------------------------------------------------------- #
plt.rc('font', family='serif')
matplotlib.rc('text')#, usetex=True)
matplotlib.rc('legend')
#matplotlib.rcParams['text.latex.preamble'] = r'\boldmath' # makes the numbers bold




#############################################################################################
n_metrics = 4
model_names = plot_vals["model_names"]
models_ax = np.arange(len(model_names))

fig, axs = plt.subplots(nrows=n_metrics, ncols=1, sharex=True, figsize=(5, 7))

row = 0
for (group_key, group) in plot_vals.items():
    if group_key == "model_names": continue
    elif "benchmark" in group_key: row-=len(group)
    elif group_key == "gain": break

    for (metric_key, metric) in group.items():
        print(f"k = {row},\tgroup = {group_key},\tmetric = {metric_key}")
        axs[row].errorbar(models_ax, metric["vals"], yerr=metric["stds"], label=group_key, fmt='--o')
        axs[row].set_title(metric_key)

        axs[row].grid(alpha=0.5, axis='y')
        row+=1

#axs[n_metrics-1].spines.right.set_visible(False)
#axs[n_metrics-1].tick_params(axis="x", length=0, labeltop=False)  # remove ticks and labels for the left plot
#axs[n_metrics-1].set_xticks([]) # remove ticks from ax1 entirely
axs[n_metrics-1].set_xticks(models_ax, model_names)
axs[n_metrics-1].legend(
    bbox_to_anchor=(0, -0.5),
    loc='upper left'
)


fig.tight_layout() # tighten everything
fig.subplots_adjust(wspace=0)  # adjust space between Axes

#plt.show()
plt.savefig("multi-metric_plot.png", dpi=500)

#############################################################################################
"""
# plot the data
labels = [
    #r'\textbf{
    "agent, val",
#}',
    #r'\textbf{
    "agent, test",
#}',
    #r'\textbf{
    "baseline, val",
#}',
    #r'\textbf{
    "baseline, test"
#}'
]
#comp_label = r'\textbf{only unit tests}'
comp_label = "v"
#test_label = r'\textbf{unit tests and validation set}'
test_label = "unit tests and validation set"

model_names = plot_vals["model_names"]
data = [
    plot_vals["benchmark"]["comp_pass_1 (%)"]["vals"], # comp_pass_bench,
    plot_vals["benchmark"]["test_pass_1 (%)"]["vals"], # test_pass_bench,
    plot_vals["baseline" ]["comp_pass_1 (%)"]["vals"], # comp_pass_base,
    plot_vals["baseline" ]["test_pass_1 (%)"]["vals"], # test_pass_base
]
data_std = [
    plot_vals["benchmark"]["comp_pass_1 (%)"]["stds"], # comp_pass_bench_std,
    plot_vals["benchmark"]["test_pass_1 (%)"]["stds"], # test_pass_bench_std,
    plot_vals["baseline" ]["comp_pass_1 (%)"]["stds"], # comp_pass_base_std,
    plot_vals["baseline" ]["test_pass_1 (%)"]["stds"], # test_pass_base_std
]
comp_gain = plot_vals["gain"]["comp_pass_1 (%)"]["vals"]
test_gain = plot_vals["gain"]["test_pass_1 (%)"]["vals"]

width = 0.22 # width of the bars in the plot
x = np.arange(len(model_names))
dx = [ width*i for i in [-0.65, 1.15, -1.15, 0.65]] # relative positions of the bars in the plot (for each model)
#xx = 0.5*(x[-2]+x[-3]) # position between pure and hybrid mode
hatches = ['\\\\\\', '///', '\\', '/']

#########################################################################################################
# plot the performance for all models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 6.5), sharey=True, gridspec_kw={'width_ratios': [1, 10]})

# plot the same data on both Axes
for ax in [ax1, ax2]:
    for j in range(4):
        ax.barh(x+dx[j], data[j], width, xerr=data_std[j], label=labels[j], edgecolor='#404040', hatch=hatches[j], capsize=4)
        #ax.plot([0, 100], [xx, xx], c="black", lw=0.7) # separation line between pure and hybrid mode

# zoom / limit the view to different portions of the data
ax1.set_xlim(0, 2)
ax2.set_xlim(15, 100)

# hide the spines between ax and ax2
ax1.invert_yaxis()
ax1.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)
#ax1.xaxis.tick_top()
#ax2.xaxis.tick_bottom()
ax1.tick_params(axis="x", length=0, labeltop=False)  # remove ticks and labels for the left plot
ax2.tick_params(axis="y", length=0, labeltop=False)  # remove ticks and labels for the right plot
ax1.set_xticks([]) # remove ticks from ax1 entirely
#ax2.set_yticks([]) # remove ticks from ax2 entirely

d1 = 0.3; d2 = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-d1, -d2), (d1, d2)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0.75, 0.5, 0.75, 0.5], [0, 0, 1, 1], transform=ax1.transAxes, **kwargs) # add the // to the axis

ax2.grid(axis='x', alpha=0.5)
ax1.set_yticks(x, model_names)
ax1.set_ylabel(#r'\textbf{
    "Model"
#}'
)
ax2.set_xlabel(#r'\textbf{
    r"Performance (\%)"
#}'
)
ax2.legend(bbox_to_anchor=(0, 1), loc='lower center')

fig.tight_layout() # tighten everything
fig.subplots_adjust(wspace=0)  # adjust space between Axes

#plt.show()
plt.savefig("performance_plot.pdf", dpi=500)
#"""

#########################################################################################################
# show the delta between benchmark-results and baseline
"""
fig2, ax3 = plt.subplots(1, 1, figsize=(4.5, 2.8))
ax3.scatter(comp_gain, x, label=comp_label, marker="o")
ax3.scatter(test_gain, x, label=test_label, marker="^")
#ax3.plot([0, 100], [xx, xx], c="black", lw=0.7)  # separation line between pure and hybrid mode

# zoom / limit the view to different portions of the data
max_gain = np.ceil(max(max(comp_gain), max(test_gain)))+1
ax3.set_xlim(-5, max_gain)

ax3.invert_yaxis()

ax3.grid(axis='x', alpha=0.5)
ax3.set_xticks(np.arange(0, max_gain, 2.0))
ax3.set_yticks(x, model_names)
ax3.set_ylabel(#r'\textbf{
    "Model"
#}'
)
ax3.set_xlabel(#r'\textbf{
    r"Gain (\% points, agent - baseline)"
#}'
)
ax3.legend(bbox_to_anchor=(0, 1), loc='lower center')

fig2.tight_layout() # tighten everything
fig2.subplots_adjust(wspace=0)  # adjust space between Axes

#plt.show()
plt.savefig("gain_plot.pdf", dpi=500)
#"""




#########################################################################################################
# Evolution scatterplot with arrows
"""
# Example data
baseline = np.array([[x, y], ...]) # Baseline datapoints coordinates
improved = np.array([[x, y], ...]) # Improved datapoints coordinates

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot baseline and improved points
ax.scatter(baseline[:, 0], baseline[:, 1], color='blue', label='Baseline', s=100)
ax.scatter(improved[:, 0], improved[:, 1], color='green', label='Improved', s=100)

# Connect points with arrows
for i in range(len(baseline)):
    ax.annotate(
        '',
        xy=(improved[i, 0], improved[i, 1]),  # End point (improved)
        xytext=(baseline[i, 0], baseline[i, 1]),  # Start point (baseline)
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5, shrinkA=5, shrinkB=5), # shrinkA/B is to avoid overlaps
    )

# Add labels and legend
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Improvement from Baseline to Improved (with Arrows)')
ax.legend()
ax.grid(True)

# Show plot
plt.tight_layout()
plt.show()
"""