from scipy.signal import argrelmax, argrelmin
from pyhht.emd import EMD
import numpy as np

# EMD with extrema extension
class EMDext():

    def __init__(self, series):
        self.series = series

    def addBound(self):
        """
        极值对称法补充边界
        :return:
        righted_series 补充后的序列
        lefted_num 左边填充了多少个数据
        righted_num 右边填充了多少个数据

        """
        series = self.series
        n = 8 # 延拓多少个点
        max_index = argrelmax(series)[0]
        min_index = argrelmin(series)[0]
        max_data = [series[i] for i in max_index]
        min_data = [series[i] for i in min_index]

        # ----------------------------从左开始延拓-------------------------
        # 左边第一个是极大值
        lefted_num = 0
        lefted_series = []
        if max_index[0] < min_index[0]:

            if series[0] > min_data[0]:
                # 以极大值作为对称中心
                extra_data = []
                for i in range(max_index[0]+1, n+1+1):
                    extra_data.append(series[i])

                for i in range(len(extra_data)):
                    index = len(extra_data) -1 - i
                    lefted_series.append(extra_data[index])
                for i in range(max_index[0], len(series)):
                    lefted_series.append(series[i])
                if len(lefted_series) - len(series) > 0:
                    lefted_num = len(lefted_series) - len(series)

            elif series[0] <= min_data[0]:
                # 以左端点为中心延拓
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
                # 以极小值作为对称中心
                extra_data = []
                for i in range(min_index[0]+1, n+1+1):
                    # ex_index = 2 * max_index[0] - i
                    extra_data.append(series[i])
                for i in range(len(extra_data)):
                    index = len(extra_data) -1 - i
                    lefted_series.append(extra_data[index])
                for i in range(min_index[0], len(series)):
                    lefted_series.append(series[i])
                if len(lefted_series) - len(series) > 0:
                    lefted_num = len(lefted_series) - len(series)
            elif series[0] >= max_data[0]:
                # 以左端点为中心延拓

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

        # ---------------------从右边开始延拓---------------------
        # 右边第一个是极大值
        righted_num = 0
        righted_series = []
        if max_index[-1] > min_index[-1]:
            if series[-1] > min_data[-1]:
                # 以极大值作为对称中心
                extra_data = []
                for i in range(n):
                    extra_data.append(series[max_index[-1]-1-i])


                for i in range(max_index[-1]+1):
                    righted_series.append(series[i])
                for i in range(len(extra_data)):
                    righted_series.append(extra_data[i])
                if len(righted_series) - len(series) > 0:
                    righted_num = len(righted_series) - len(series)

            elif series[-1] <= min_data[-1]:
                # 以左端点作为对称中心
                extra_data = []
                for i in range(n):
                    extra_data.append(series[-1 -1 - i])

                for i in range(len(series)):
                    righted_series.append(series[i])
                for i in range(len(extra_data)):
                    righted_series.append(extra_data[i])

                righted_num = n

        # 右边第一个是极小值
        elif max_index[-1] < min_index[-1]:

            if series[-1] < max_data[-1]:
                # 以极小值作为对称中心
                extra_data = []
                for i in range(n):
                    extra_data.append(series[min_index[-1] - 1 - i])

                for i in range(min_index[-1] + 1):
                    righted_series.append(series[i])

                for i in range(len(extra_data)):
                    righted_series.append(extra_data[i])
                if len(righted_series) - len(series) > 0:
                    righted_num = len(righted_series) - len(series)


            elif series[-1] >= max_data[-1]:

                # 以左端点作为对称中心
                extra_data = []

                for i in range(n):
                    extra_data.append(series[-1 - 1 - i])

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

        imfs = imfs[:, lefted_num:length - righted_num]

        return imfs