import matplotlib.pyplot as plt
import numpy as np

performance_values = [
    # model, values: baseline (ex, test) + bench (ex, test), variance
    ["Qwen2.5-Coder-3B-Instruct\n(1 sample, 1 iteration)",
        [76.22, 73.78, 79.88, 76.22],
        [0,0,0,0]
    ], ["Qwen3-4B\n(1 sample, 1 iteration)",
        [86.59, 84.76, 90.24, 88.41],
        [0,0,0,0]
    ], ["Qwen2.5-Coder-3B-Instruct\n+Qwen3-4B\n(1 sample, 1 iteration)",
        [78.05, 75.00, 90.24, 84.15],
        [0,0,0,0]
    ], ["Qwen2.5-Coder-3B-Instruct\n+Qwen3-4B\n(5 samples, 3 iterations)",
        [78.90, 75.37, 90.73, 84.63],
        [0.91, 1.69, 0.95, 2.31]
    ], ["Qwen3-Coder-Next\n(5 samples, 1 iteration)",
        [93.66, 92.68, 98.41, 96.34],
        [0.08, 0.15, 0.22, 0.24]
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

labels = ["exemples_pass with compiler feedback", "test_pass with compiler feedback", "exemples_pass baseline", "test_pass baseline"]
data = [comp_pass_bench, test_pass_bench, comp_pass_base, test_pass_base]
data_std = [comp_pass_bench_std, test_pass_bench_std, comp_pass_base_std, test_pass_base_std]

width = 0.2
x = np.arange(len(performance_values))
dx = [ width*i for i in [-0.6, 1, -1, 0.6]]
hatches = ['\\\\\\', '///', '\\', '/']

fig = plt.figure(figsize=(10, 4))

# plot data in grouped manner of bar type
for j in range(4):
    plt.bar(x+dx[j], data[j], width, yerr=data_std[j], label=labels[j], edgecolor='#404040', hatch=hatches[j])

plt.grid(axis='y', alpha=0.5)
plt.xticks(x, model_names)
plt.xlabel("Model")
plt.ylabel("Performance (%)")
plt.legend(loc='lower center')
plt.tight_layout()
#plt.show()
plt.savefig("performance_plot.png", dpi=500)

"""
plt.bar(x-0.2, y1, width, color='cyan')
plt.bar(x, y2, width, color='orange')
plt.bar(x+0.2, y3, width, color='green')
plt.xticks(x, ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'])
plt.xlabel("Teams")
plt.ylabel("Scores")
plt.legend(["Round 1", "Round 2", "Round 3"])
plt.show()
"""
