import benchmark_connected_vs_ordinary as bench

# experiment_names = ["mnist", "fmnist", "cifar10", "cifar100"]
experiment_names = ["mnist"]


experiment_configs = [
    [True, True],
    [True, False],
    [False, True],
    [False, False],
]

###### This is for comparing multi-invex with other possible architectures.
for exp_idx in [0,1,2]:
    for exp in experiment_names:
        for i in range(len(experiment_configs)):
            inv, conn = experiment_configs[i]
            print()
            print(f"Experiment {exp}: inv-{inv} conn-{conn} exp-{exp_idx} cuda:0")
            print("---------------------------------------------------------------")
            bench.benchmark(exp, inv, conn, exp_idx, cuda=0, linear_clf=False)
            
####### This is for comparing when Regions==Classes
experiment_configs = [
    [True, True],
    [True, False],
]            
for exp_idx in [0,1,2]:
    for exp in experiment_names:
        for i in range(len(experiment_configs)):
            inv, conn = experiment_configs[i]
            print()
            print(f"Experiment {exp}: inv-{inv} conn-{conn} exp-{exp_idx} cuda:0")
            print("---------------------------------------------------------------")
            bench.benchmark(exp, inv, conn, exp_idx, cuda=0, linear_clf=True)