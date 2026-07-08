import matplotlib.pyplot as plt
import matplotlib
import numpy as np

performance_values = [
    # model, values: baseline (ex, test) + bench (ex, test), variance
    [   r'\textbf{\begin{flushright} Qwen2.5-Coder-3B-Instruct \end{flushright}}',
        [76.59, 73.78, 80.12, 76.46],
        [1.34, 1.14, 1.26, 1.76]
    ], [r'\textbf{\begin{flushright} Qwen3-4B (with thinking) \end{flushright}}',
        [86.59, 84.76, 90.24, 88.41],
        [0,0,0,0]
    ], [r'\textbf{\begin{flushright} Qwen3-Coder-Next-80B-A3B \end{flushright}}',
        [93.66, 92.68, 98.41, 96.34],
        [0.08, 0.15, 0.22, 0.24]
    ], [r'\textbf{\begin{flushright} Qwen2.5-Coder-3B-Instruct\\+Qwen3-4B \end{flushright}}',
        [78.05, 75.00, 90.24, 84.15],
        [0,0,0,0]
    ], [r'\textbf{\begin{flushright} Qwen2.5-Coder-3B-Instruct\\+Qwen3-4B (3 iterations) \end{flushright}}',
        [78.90, 75.37, 90.73, 84.63],
        [0.91, 1.69, 0.95, 2.31]
    ],
] # this format is good for checking the values, but it's inconvenient for the plot

# reorganizing the data to make it easier to create and manipulate the plot
model_names = []
comp_pass_base  = [] ; comp_pass_base_std  = []
test_pass_base  = [] ; test_pass_base_std  = []
comp_pass_bench = [] ; comp_pass_bench_std = []
test_pass_bench = [] ; test_pass_bench_std = []

for m, model in enumerate(performance_values):
    model_names.append(model[0])
    comp_pass_base.append( model[1][0]) ; comp_pass_base_std.append( model[2][0])
    test_pass_base.append( model[1][1]) ; test_pass_base_std.append( model[2][1])
    comp_pass_bench.append(model[1][2]) ; comp_pass_bench_std.append(model[2][2])
    test_pass_bench.append(model[1][3]) ; test_pass_bench_std.append(model[2][3])

labels = [
    r'\textbf{exemples_pass with compiler feedback}',
    r'\textbf{test_pass with compiler feedback}',
    r'\textbf{exemples_pass baseline}',
    r'\textbf{test_pass baseline}'
]
data = [comp_pass_bench, test_pass_bench, comp_pass_base, test_pass_base]
data_std = [comp_pass_bench_std, test_pass_bench_std, comp_pass_base_std, test_pass_base_std]

width = 0.22 # width of the bars in the plot
x = np.arange(len(performance_values))
dx = [ width*i for i in [-0.65, 1.15, -1.15, 0.65]] # relative positions of the bars in the plot (for each model)
hatches = ['\\\\\\', '///', '\\', '/']

plt.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
#matplotlib.rc('legend')
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath' # makes the numbers bold

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 6), sharey=True, gridspec_kw={'width_ratios': [1, 10]})

# plot the same data on both Axes
xx = 0.5*(x[-2]+x[-3])
for ax in [ax1, ax2]:
    for j in range(4):
        ax.barh(x+dx[j], data[j], width, xerr=data_std[j], label=labels[j], edgecolor='#404040', hatch=hatches[j], capsize=4)
        ax.plot([0, 100], [xx, xx], c="black", lw=0.7)

# zoom-in / limit the view to different portions of the data
ax1.set_xlim(0, 2)
ax2.set_xlim(57, 100)

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
ax1.set_ylabel(r'\textbf{Model}')
ax2.set_xlabel(r'\textbf{Performance (\%)}')
ax2.legend(bbox_to_anchor=(-0.58, 1)) # loc='upper center')

fig.tight_layout() # tighten everything
fig.subplots_adjust(wspace=0)  # adjust space between Axes

#plt.show()
plt.savefig("performance_plot.png", dpi=500)
