import numpy as np
import math
import matplotlib.pyplot as plt


class Entropy():
    def __init__(self):
        pass

    def cal_entropy(self):
        pass


class ApEn(Entropy):
    def __init__(self, u, m, r):
        self.u = u           # 时间序列
        self.r = r           # 相似度比较的阅知
        self.m = m           # 子序列长度
        self.N = len(u)      # 时间序列长度

    def _max_dist(self, xi, xj):
        return np.max([np.abs(a - b) for a, b in zip(xi, xj)])

    def _phi(self, m):
        X = []
        for i in range(self.N - m + 1):
            Xi = []
            for j in range(i, i + m):
                Xi.append(self.u[j])
            X.append(Xi)
        C = []
        for Xi in X:
            num = 0
            for Xj in X:
                if self._max_dist(Xi, Xj) <= self.r:
                    num += 1
            C.append(num / (self.N - m + 1))
        phi = np.sum(np.log(C)) / (self.N - m + 1)
        return phi

    def cal_entropy(self):
        return np.abs(self._phi(self.m + 1) - self._phi(self.m))


class SampEn(Entropy):
    def __init__(self, u, m, r):
        self.u = u
        self.r = r
        self.m = m
        self.N = len(u)

    def _max_dist(self, xi, xj):
        return np.max([np.abs(a - b) for a, b in zip(xi, xj)])

    def _phi(self, m):
        X = []
        for i in range(self.N - m + 1):
            Xi = []
            for j in range(i, i + m):
                Xi.append(self.u[j])
            X.append(Xi)
        B = []
        for Xi in X:
            num = 0
            for Xj in X:
                if self._max_dist(Xi, Xj) <= self.r:
                    num += 1
            B.append((num - 1) / (self.N - m))
        phi = np.sum(B) / (self.N - m + 1)
        return phi

    def cal_entropy(self):
        return -np.log(self._phi(self.m + 1) / self._phi(self.m))


class FsEn(Entropy):
    def __init__(self, u, m, r):
        self.u = u
        self.r = r
        self.m = m
        self.N = len(u)

    def _max_dist(self, xi, xj):
        return np.max([np.abs(a) - np.abs(b) for a, b in zip(xi, xj)])

    def _phi(self, m):
        X = []              # 重构的子序列
        for i in range(self.N - m + 1):
            Xi = []
            for j in range(i, i + m):
                Xi.append(self.u[j])
            Xi = Xi - np.mean(Xi)
            X.append(Xi)
        dij = []  # 满足相似度阈值条件的个数与总的统计数目之间的比值
        for Xi in X:
            di = []
            for Xj in X:
                di.append(self._max_dist(Xi, Xj))
            dij.append(di)
        A = []
        for i in range(self.N - m + 1):
            Ai = []
            for j in range(self.N - m + 1):
                if i == j:
                    continue
                Ai.append(np.exp(-np.log(2) * np.square(dij[i][j] / self.r)))
            A.append(Ai)
        C = []
        for Ai in A:
            C.append(np.sum(Ai) / (self.N - m))
        phi = np.sum(C) / (self.N - m + 1)
        return phi

    def cal_entropy(self):
        return np.log(self._phi(self.m)) - np.log(self._phi(self.m + 1))


class PeEn(Entropy):
    def __init__(self, u, m, l):
        self.u = u
        self.m = m
        self.l = l
        self.N = len(u)

    def cal_entropy(self, std=True):
        '''

        Parameters
        ----------
        std 是否标准化输出

        Returns
        -------

        '''
        X = []
        for i in range(self.N - self.m + 1):
            Xi = []
            for j in range(i, i + (self.m - 1 + 1) * self.l, self.l):
                Xi.append(self.u[j])
            X.append(Xi)

        J = []
        for Xi in X:
            dict = {}
            for i in range(len(Xi)):
                dict[i + 1] = Xi[i]
            sorted_dict = sorted(dict.items(), key=lambda x: x[1])
            j = []
            for x, y in sorted_dict:
                j.append(x)
            J.append(j)

        J_dict = {}
        for j in J:
            temp = map(str, j)
            temp = ''.join(temp)
            if temp not in J_dict.keys():
                J_dict[temp] = 1
            else:
                J_dict[temp] += 1

        H = []
        for key in J_dict.keys():
            count = J_dict[key]
            p = count / len(J)
            H.append(-p * np.log(p))

        Hp = np.sum(H)
        if std:
            return Hp / np.log(math.factorial(self.m))
        else:
            return Hp


class WeightedPeEn(Entropy):
    def __init__(self, u, m, l):
        self.u = u
        self.m = m
        self.l = l
        self.N = len(u)

    def cal_entropy(self, std=True):
        '''

        Parameters
        ----------
        std 是否标准化输出

        Returns
        -------

        '''
        X = []              # 重构的子序列
        for i in range(self.N - self.m + 1):
            Xi = []
            for j in range(i, i + (self.m - 1 + 1) * self.l, self.l):
                Xi.append(self.u[j])
            X.append(Xi)

        J = []
        for Xi in X:
            dict = {}
            var = np.var(Xi)
            for i in range(len(Xi)):
                dict[i + 1] = Xi[i]
            sorted_dict = sorted(dict.items(), key=lambda x: x[1])
            j = []
            dict_var = {}
            for x, y in sorted_dict:
                j.append(x)
            key = map(str, j)
            key = ''.join(key)
            dict_var[key] = var
            J.append(dict_var)

        # print(J)

        J_dict = {}
        for j in J:
            for key in j:
                if key not in J_dict:
                    value = j[key]
                    J_dict[key] = [value]
                else:
                    value = j[key]
                    J_dict[key].append(value)

        weighted_sum = 0
        for key in J_dict.keys():
            weighted_sum += np.sum(J_dict[key])

        H = []
        for key in J_dict.keys():
            w_sum = np.sum(J_dict[key])
            p = w_sum / weighted_sum
            H.append(-p * np.log(p))

        Hp = np.sum(H)
        if std:
            return Hp / np.log(math.factorial(self.m))
        else:
            return Hp


class IncrEn(Entropy):
    def __init__(self, u, m, r):
        self.u = u
        self.m = m
        self.r = r
        self.N = len(u)

    def sgn(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def cal_q(self, x, r, std):
        n_std = int(x / std)
        if n_std >= r:
            return r
        else:
            return n_std

    def cal_entropy(self):
        X = []
        diff_u = np.diff(self.u)
        for i in range(self.N - self.m):
            Xi = []
            for j in range(i, i + self.m):
                Xi.append(diff_u[j])
            X.append(Xi)

        # 重构向量对应的模型向量 example:[[(s1,q1),(s2,q2)], [(s1,q1),(s2,q2)]]
        # 其中[(s1,q1),(s2,q2)]代表一阶差分序列中每个子序列对应的模式向量
        X_w = []
        std = np.std(np.abs(diff_u))   # 窗口内一阶差的标准差

        for i in range(self.N - self.m):
            u = X[i]
            u_w = []
            for j in range(self.m):
                s = self.sgn(u[j])
                q = self.cal_q(np.abs(u[j]), self.r, std)
                u_w.append((s, q))
            X_w.append(u_w)

        key_list = []     # 所有的w组合 [1111, 1110, -12-13]
        for w in X_w:
            key = []
            for wi in w:
                for a in wi:
                    key.append(a)
            key_list.append(''.join(map(str, key)))

        # print(key_list)

        J_dict = {}
        for key in key_list:
            if key not in J_dict.keys():
                J_dict[key] = 1
            else:
                J_dict[key] += 1

        H = []
        for key in J_dict.keys():
            count = J_dict[key]
            p = count / len(key_list)
            H.append(-p * np.log(p))

        en = np.sum(H)
        return en
