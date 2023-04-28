import copy

import ordpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import antropy as ant
from sklearn.metrics import f1_score


def plot_entropy_per_ts(entropy_name, batches_anomalies,
                        entropies, boundary_bottom,
                        boundary_up, ts):
    color = [
        'red' if i in batches_anomalies else 'blue' for i in range(
            len(entropies))]

    plt.bar(list(range(len(entropies))), entropies, color=color)
    plt.axvline(x=int(len(entropies) // 2), color='orange',
                linestyle='--', label='train-test split')
    plt.axhline(y=boundary_bottom, color='orange', linestyle='--')
    plt.axhline(y=boundary_up, color='orange', linestyle='--')

    plt.legend()
    plt.ylabel(entropy_name)
    plt.xlabel('Batch')
    plt.savefig(f'results/imgs/{ts}_entropy_{entropy_name}.png')
    plt.clf()


def entropy_modelling():
    entropies_no_anomalies = []

    # loop through datasets
    for dataset, subsets in [('kpi', ['train'])]:
        for tss in subsets:
            hcs = pd.DataFrame([])
            train_data_path = 'datasets/' + dataset + '/' + tss + '/'

            res = pd.DataFrame([])
            for filename in os.listdir(train_data_path):
                print(filename)
                f = os.path.join(train_data_path, filename)
                # loop through windows 5 to 100
                # For me 65 was overall the best
                for wdw in range(100, 5, -5):
                    # loop through entropy variants
                    # for entropy_name in ['spectral_entropy', 'value_decomposition_entropy', 'approximate_entropy',
                    #                      'sample_entropy', 'permutation_entropy', 'renyi', 'tsallis']:
                    for entropy_name in ['value_decomposition_entropy']:
                        # For me 1.5 was overall the best
                        for factor in np.arange(0.5, 4.1, 0.1):
                            factor = round(factor, 1)
                            print(f'Doing entropy {wdw}, {factor}')
                            ts = pd.read_csv(f)

                            ts2 = None
                            if dataset == 'kpi':
                                ts2 = pd.read_csv(
                                    'datasets/' + dataset + '/test/' + filename)
                            ts.rename(
                                columns={
                                    'timestamps': 'timestamp',
                                    'anomaly': 'is_anomaly'},
                                inplace=True)

                            batches_anomalies = []
                            entropy_differences = []
                            collected_entropies = []
                            y_predicted = []
                            y_true = []

                            if dataset != 'kpi':
                                ts1, ts2 = np.array_split(copy.deepcopy(ts), 2)
                            else:
                                ts1 = ts

                            step = wdw
                            try:
                                for start in range(0, ts1.shape[0], step):
                                    window = ts1.iloc[start:start + step]

                                    if True:
                                        if entropy_name == 'spectral_entropy':
                                            collected_entropies.append(ant.spectral_entropy(window['value'].to_numpy(),
                                                                                            sf=100, method='welch',
                                                                                            normalize=True))
                                        elif entropy_name == 'value_decomposition_entropy':
                                            collected_entropies.append(
                                                ant.svd_entropy(window['value'].to_numpy(), normalize=True))
                                        elif entropy_name == 'approximate_entropy':
                                            collected_entropies.append(
                                                ant.app_entropy(window['value'].to_numpy()))
                                        elif entropy_name == 'sample_entropy':
                                            collected_entropies.append(
                                                ant.sample_entropy(window['value'].to_numpy()))
                                        elif entropy_name == 'hjorth_entropy':
                                            collected_entropies.append(
                                                ant.hjorth_params(window['value'].to_numpy()))
                                        elif entropy_name == 'permutation_entropy':
                                            collected_entropies.append(ant.perm_entropy(
                                                window['value'].to_numpy(), normalize=True))
                                        elif entropy_name == 'tsallis':
                                            collected_entropies.append(
                                                ordpy.tsallis_entropy(window['value'].to_numpy()))
                                        elif entropy_name == 'renyi':
                                            collected_entropies.append(
                                                ordpy.renyi_entropy(window['value'].to_numpy()))

                                        if len(collected_entropies) > 1:
                                            entropy_differences.append(
                                                abs(collected_entropies[-1] - collected_entropies[-2]))

                                        if window['is_anomaly'].tolist().count(
                                                1) > 0:
                                            batches_anomalies.append(
                                                start // step)
                                        else:
                                            entropies_no_anomalies.append(
                                                collected_entropies[-1])

                                entropies = copy.deepcopy(collected_entropies)
                                batches_anomalies = []
                                entropies = []
                                mean_entropy = np.mean(
                                    np.array([v for v in collected_entropies if pd.notna(v)]))
                                std_entropy = np.std(
                                    np.array([v for v in collected_entropies if pd.notna(v)]))
                                boundary_bottom = mean_entropy - factor * std_entropy
                                boundary_up = mean_entropy + factor * std_entropy
                                for start in range(0, ts2.shape[0], step):
                                    window = ts2.iloc[start:start + step]
                                    if window.shape[0] == step:
                                        if entropy_name == 'spectral_entropy':
                                            entropies.append(ant.spectral_entropy(window['value'].to_numpy(),
                                                                                  sf=100, method='welch', normalize=True))
                                        elif entropy_name == 'value_decomposition_entropy':
                                            entropies.append(
                                                ant.svd_entropy(
                                                    window['value'].to_numpy(),
                                                    normalize=True))
                                        elif entropy_name == 'approximate_entropy':
                                            entropies.append(ant.app_entropy(
                                                window['value'].to_numpy()))
                                        elif entropy_name == 'sample_entropy':
                                            entropies.append(ant.sample_entropy(
                                                window['value'].to_numpy()))
                                        elif entropy_name == 'hjorth_entropy':
                                            entropies.append(ant.hjorth_params(
                                                window['value'].to_numpy()))
                                        elif entropy_name == 'permutation_entropy':
                                            entropies.append(
                                                ant.perm_entropy(window['value'].to_numpy(), normalize=True))
                                        elif entropy_name == 'tsallis':
                                            entropies.append(
                                                ordpy.tsallis_entropy(window['value'].to_numpy()))
                                        elif entropy_name == 'renyi':
                                            entropies.append(
                                                ordpy.renyi_entropy(window['value'].to_numpy()))

                                        if window['is_anomaly'].tolist().count(
                                                1) > 0:
                                            # batches_anomalies.append(len(collected_entropies) + (start // step))
                                            batches_anomalies.append(
                                                start // step)

                                        if mean_entropy:

                                            if boundary_bottom <= entropies[-1] <= boundary_up:
                                                y_predicted.append(0)
                                            else:
                                                y_predicted.append(1)

                                            if window['is_anomaly'].tolist().count(
                                                    1) > 0:
                                                y_true.append(1)
                                            else:
                                                y_true.append(0)

# Plot entropy model performance per batch
                                if plot:
                                    plot_entropy_per_ts(entropy_name, batches_anomalies,
                                                        entropies, boundary_bottom,
                                                        boundary_up, filename.replace('.csv', ''))

                                res = res.append({
                                    'ts': filename.replace('.csv', ''),
                                    'window': wdw,
                                    'factor': factor,
                                    'dataset': dataset,
                                    'subset': ts,
                                    'f1': f1_score(y_true, y_predicted, average='binary')
                                }, ignore_index=True)
                                res.to_csv(
                                    f'results/scores/entropy_vs_window_{dataset}_{tss}_per_ts.csv')
                            except Exception as e:
                                pass
