import numpy as np
import pandas as pd
from drift_detection.entropy import *
from queue import deque
from pyhht.emd import EMD
from scipy.signal import argrelmax, argrelmin


def showPlot(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(data)), data)
    plt.show()

# the ETFE class to extract features


class ETFE():
    def __init__(self, entropy_type=''):
        self.window_len = 100
        self.window_data = deque(maxlen=self.window_len)
        self.count = 0
        self.entropy_list = []
        self.average_entropy_list = []
        self.entropy_type = entropy_type

    def cal_imfs(self, series):

        decomposer = EMDBE(series)
        imfs = decomposer.decompose()

        return imfs

    def cal_entropy(self, series):
        n_series = np.shape(series)[0]
        entropys = []
        for i in range(n_series):
            u = series[i]
            m = 2
            r = 0.2 * np.std(u)
            if self.entropy_type == '':
                en = PeEn(u, 3, 1)
            elif self.entropy_type == 'PeEn':
                en = PeEn(u, 4, 1)
            elif self.entropy_type == 'WPeEn':
                en = WeightedPeEn(u, 4, 1)
            elif self.entropy_type == 'FsEn':
                en = FsEn(u, m, r)
            elif self.entropy_type == 'IncrEn':
                en = IncrEn(u, m, 2)
            elif self.entropy_type == 'ApEn':
                en = ApEn(u, m, r)
            elif self.entropy_type == 'SampEn':
                en = SampEn(u, m, r)

            entropy = en.cal_entropy()
            entropys.append(entropy)
        return entropys

    def feed(self, data):
        self.window_data.append(data)
        self.count += 1
        if len(self.window_data) < self.window_len:
            return

        if self.count % 5 == 0:
            window_data = np.array(self.window_data)
            imfs = self.cal_imfs(window_data)
            entropy = self.cal_entropy(imfs[:2])

            self.entropy_list.append(entropy)
            return entropy
        else:
            return None


# EMD with extrema extension
class EMDBE():

    def __init__(self, series):
        self.series = series

    def addBound(self):

        series = self.series
        n = 8
        max_index = argrelmax(series)[0]
        min_index = argrelmin(series)[0]
        max_data = [series[i] for i in max_index]
        min_data = [series[i] for i in min_index]

        lefted_num = 0
        if max_index[0] < min_index[0]:
            if series[0] > min_data[0]:
                extra_data = []
                for i in range(max_index[0] + 1, n + 1 + 1):
                    extra_data.append(series[i])
                lefted_series = []
                for i in range(len(extra_data)):
                    index = len(extra_data) - 1 - i
                    lefted_series.append(extra_data[index])
                for i in range(max_index[0], len(series)):
                    lefted_series.append(series[i])
                if len(lefted_series) - len(series) > 0:
                    lefted_num = len(lefted_series) - len(series)

            elif series[0] <= min_data[0]:
                lefted_series = []
                extra_data = []
                for i in range(1, n + 1):
                    extra_data.append(series[i])

                for i in range(len(extra_data)):
                    index = len(extra_data) - 1 - i
                    lefted_series.append(extra_data[index])
                for i in range(len(series)):
                    lefted_series.append(series[i])
                lefted_num = n

        elif max_index[0] > min_index[0]:
            if series[0] < max_data[0]:
                extra_data = []
                for i in range(min_index[0] + 1, n + 1 + 1):
                    # ex_index = 2 * max_index[0] - i
                    extra_data.append(series[i])
                lefted_series = []
                for i in range(len(extra_data)):
                    index = len(extra_data) - 1 - i
                    lefted_series.append(extra_data[index])
                for i in range(min_index[0], len(series)):
                    lefted_series.append(series[i])
                if len(lefted_series) - len(series) > 0:
                    lefted_num = len(lefted_series) - len(series)

            elif series[0] >= max_data[0]:
                lefted_series = []
                extra_data = []
                for i in range(1, n + 1):
                    extra_data.append(series[i])

                for i in range(len(extra_data)):
                    index = len(extra_data) - 1 - i
                    lefted_series.append(extra_data[index])
                for i in range(len(series)):
                    lefted_series.append(series[i])

                lefted_num = n

        # print(lefted_series)
        # print('lefted ', lefted_num)

        series = np.array(lefted_series)
        max_index = argrelmax(series)[0]
        min_index = argrelmin(series)[0]
        max_data = [series[i] for i in max_index]
        min_data = [series[i] for i in min_index]

        # ---------------------from right---------------------
        righted_num = 0
        if max_index[-1] > min_index[-1]:
            if series[-1] > min_data[-1]:
                extra_data = []
                for i in range(n):
                    extra_data.append(series[max_index[-1] - 1 - i])
                righted_series = []

                for i in range(max_index[-1] + 1):
                    righted_series.append(series[i])
                for i in range(len(extra_data)):
                    righted_series.append(extra_data[i])
                if len(righted_series) - len(series) > 0:
                    righted_num = len(righted_series) - len(series)

            elif series[-1] <= min_data[-1]:
                extra_data = []
                for i in range(n):
                    extra_data.append(series[-1 - 1 - i])
                righted_series = []

                for i in range(len(series)):
                    righted_series.append(series[i])
                for i in range(len(extra_data)):
                    righted_series.append(extra_data[i])

                righted_num = n

        elif max_index[-1] < min_index[-1]:

            if series[-1] < max_data[-1]:
                extra_data = []
                for i in range(n):
                    extra_data.append(series[min_index[-1] - 1 - i])

                righted_series = []

                for i in range(min_index[-1] + 1):
                    righted_series.append(series[i])

                for i in range(len(extra_data)):
                    righted_series.append(extra_data[i])
                if len(righted_series) - len(series) > 0:
                    righted_num = len(righted_series) - len(series)

            elif series[-1] >= max_data[-1]:

                extra_data = []

                for i in range(n):
                    extra_data.append(series[-1 - 1 - i])

                righted_series = []

                for i in range(len(series)):
                    righted_series.append(series[i])

                for i in range(len(extra_data)):
                    righted_series.append(extra_data[i])

                righted_num = n

        else:
            righted_series = np.array(lefted_series)

        # print(righted_series)
        # print('righted ', righted_num)

        return righted_series, lefted_num, righted_num

    def decompose(self):
        series, lefted_num, righted_num = self.addBound()
        series = np.array(series)
        emd = EMD(series)

        imfs = emd.decompose()
        length = len(imfs[0])

        # print('series len ',len(series), ' lefted num ', lefted_num, ' righted num ',righted_num)
        imfs = imfs[:, lefted_num:length - righted_num]

        # imf1 = imfs[0]
        # imf1_len = len(imf1)
        # imf1 = imf1[lefted_num:imf1_len-righted_num]
        # print(len(imf1))

        # emd2 = EMD(np.array(self.series))
        # imf2s = emd2.decompose()
        # imf2 = imf2s[0]
        #
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(range(len(imf1)), imf1, c='b')
        # ax.plot(range(len(imf2)), imf2, c='r')
        # ax.legend(['bounded', 'origin'])
        # plt.show()
        return imfs
