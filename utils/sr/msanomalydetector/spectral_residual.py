"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from datetime import datetime

import pandas as pd
import numpy as np
import antropy as ant
from scipy import stats

from msanomalydetector.util import *
import msanomalydetector.boundary_utils as boundary_helper
import plotly.graph_objects as go


class SpectralResidual:
    def __init__(self, series, use_drift, threshold, mag_window, score_window,
                 sensitivity, detect_mode, dataset,
                 filename, drift_detector, batch_size=32, drift_count_limit=10, 
                 anomaly_window=65, data_in_memory_sz=3000):
        self.__series__ = series
        self.__values__ = self.__series__['value'].tolist()
        self.__threshold__ = threshold
        self.__mag_window = mag_window
        self.__score_window = score_window
        self.__sensitivity = sensitivity
        self.__detect_mode = detect_mode
        self.__anomaly_frame = None
        self.__batch_size = batch_size
        if self.__batch_size <= 0:
            self.__batch_size = len(series)

        self.__batch_size = max(12, self.__batch_size)
        self.__batch_size = min(len(series), self.__batch_size)

        self.anomaly_window = anomaly_window
        self.data_in_memory_zs = data_in_memory_sz
        self.history = pd.DataFrame([])
        self.dynamic_threshold = True
        self.dataset = dataset
        self.filename = filename
        # TODO update: hardcoded for now
        self.entropy_factor = 1.5
        self.entropy_window = 65
        self.drift_detector = drift_detector
        self.is_drift = False
        self.drift_alerting_cts = 0
        self.drift_count_limit = drift_count_limit
        self.dynamic_thresholds = []
        self.use_drift_adaptation = use_drift

    def fit(self):
        self.svd_entropies = []
        self.__anomaly_frame = self.__detect()

        for start in range(0, len(self.__anomaly_frame), self.anomaly_window):
                try:
                    self.svd_entropies.append(
                        ant.svd_entropy(self.__anomaly_frame[start:start + self.anomaly_window]['value'].tolist(),
                                        normalize=True))
                except Exception as e:
                    print(str(e))
        self.mean_entropy = np.mean([v for v in self.svd_entropies if pd.notna(v)])
        self.boundary_bottom = self.mean_entropy - \
                               self.entropy_factor * \
                               np.std([v for v in self.svd_entropies if pd.notna(v)])
        self.boundary_up = self.mean_entropy + \
                           self.entropy_factor * \
                           np.std([v for v in self.svd_entropies if pd.notna(v)])
        result = self.__anomaly_frame
        if self.use_drift_adaptation:
            self.drift_detector.record(result['score'])
        return result

    def predict(self, window_step, current_points):
        self.__anomaly_frame = self.__detect()
        try:
            entropy = ant.svd_entropy(window_step['value'].tolist()[-current_points], normalize=True)
        except:
            entropy = (self.boundary_bottom + self.boundary_up) / 2

        result = self.__anomaly_frame
        for idx, row in result.iterrows():
            if isinstance(row['timestamp'], str):
                t = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S').timestamp()
            else:
                t = row['timestamp']

            if self.use_drift_adaptation:
                self.drift_detector.update(error=row['score'], t=t)
                response = self.drift_detector.monitor()
                if response == self.drift_detector.drift:
                    self.drift_alerting_cts += 1
                if self.drift_alerting_cts == self.drift_count_limit:
                    pass

        if entropy < self.boundary_bottom or entropy > self.boundary_up:
            extent = stats.percentileofscore(self.svd_entropies, entropy) / 100.0
            extent = 1.5 - max(extent, 1.0 - extent)
            threshold_adapted = self.__threshold__ * extent
            self.dynamic_thresholds += [threshold_adapted] * current_points
            result['isAnomaly_e'] = np.where(result['score'] > threshold_adapted, True, False)
        else:
            self.dynamic_thresholds += [self.__threshold__] * current_points
            result['isAnomaly_e'] = result['isAnomaly']

        self.history = self.history.append(result[-current_points:], ignore_index=True)
        return result[-current_points:]

    def plot(self, datatest, threshold_type='static'):
        fig = go.Figure()

        datatest.set_index('timestamp', inplace=True)

        self.history.set_index('timestamp', inplace=True)

        fig.add_trace(go.Scatter(x=self.history.index, y=self.history['score'].tolist(), name='Residual scores'))
        if threshold_type == 'dynamic':
            fig.add_trace(
                go.Scatter(x=self.history.index,
                           y=self.dynamic_thresholds,
                           name='Threshold'))
        else:
            fig.add_trace(
                go.Scatter(x=self.history.index,
                           y=[self.__threshold__ for _ in range(self.history.shape[0])],
                           name='Threshold'))

        x_fp, y_fp = [], []
        x_fn, y_fn = [], []
        for idx, (tm, row) in enumerate(self.history.iterrows()):
            threshold = self.__threshold__ if threshold_type != 'dynamic' else self.dynamic_thresholds[idx]
            if tm in datatest.index:
                if row['score'] > threshold and datatest.loc[tm, 'is_anomaly'] == 0:
                    x_fp.append(tm)
                    y_fp.append(row['score'])
                if row['score'] < threshold and datatest.loc[tm, 'is_anomaly'] == 1:
                    x_fn.append(tm)
                    y_fn.append(row['score'])

        if x_fp:
            fig.add_trace(go.Scatter(x=x_fp, y=y_fp, name='FP', mode="markers"))
        if x_fn:
            fig.add_trace(go.Scatter(x=x_fn, y=y_fn, name='FN', mode="markers"))

        fig.update_layout(showlegend=True, title='Saliency map')
        return fig

    def detect(self, window_step):
        self.__anomaly_frame = self.__detect()

        self.history = pd.concat([self.history, self.__anomaly_frame[-window_step:]])

        return self.__anomaly_frame

    def __detect(self):
        anomaly_frames = []
        for i in range(0, len(self.__series__), self.__batch_size):
            start = i
            end = i + self.__batch_size
            end = min(end, len(self.__series__))
            if end - start >= 12:
                anomaly_frames.append(self.__detect_core(self.__series__[start:end]))
            else:
                ext_start = max(0, end - self.__batch_size)
                ext_frame = self.__detect_core(self.__series__[ext_start:end])
                anomaly_frames.append(ext_frame[start-ext_start:])

        return pd.concat(anomaly_frames, axis=0, ignore_index=True)

    def __detect_core(self, series):
        values = series['value'].values
        extended_series = SpectralResidual.extend_series(values)
        mags = self.spectral_residual_transform(extended_series)
        anomaly_scores = self.generate_spectral_score(mags)
        anomaly_frame = pd.DataFrame({Timestamp: series['timestamp'].values,
                                      Value: values,
                                      Mag: mags[:len(values)],
                                      AnomalyScore: anomaly_scores[:len(values)]})
        # here we compare with tau
        anomaly_frame[IsAnomaly] = np.where(anomaly_frame[AnomalyScore] > self.__threshold__, True, False)

        if self.__detect_mode == DetectMode.anomaly_and_margin:
            anomaly_index = anomaly_frame[anomaly_frame[IsAnomaly]].index.tolist()
            anomaly_frame[ExpectedValue] = self.calculate_expected_value(values, anomaly_index)
            boundary_units = boundary_helper.calculate_boundary_unit_entire(values,
                                                                           anomaly_frame[IsAnomaly].values)
            anomaly_frame[AnomalyScore] = boundary_helper.calculate_anomaly_scores(
                values=values,
                expected_values=anomaly_frame[ExpectedValue].values,
                units=boundary_units,
                is_anomaly=anomaly_frame[IsAnomaly].values
            )

            margins = [boundary_helper.calculate_margin(u, self.__sensitivity) for u in boundary_units]
            anomaly_frame['unit'] = boundary_units

            anomaly_frame[LowerBoundary] = anomaly_frame[ExpectedValue].values - margins
            anomaly_frame[UpperBoundary] = anomaly_frame[ExpectedValue].values + margins
            isLowerAnomaly = np.logical_and(anomaly_frame[IsAnomaly].values,
                                                      anomaly_frame[LowerBoundary].values > values)
            isUpperAnomaly = np.logical_and(anomaly_frame[IsAnomaly].values,
                                                      values > anomaly_frame[UpperBoundary].values)
            anomaly_frame[IsAnomaly] = np.logical_or(isLowerAnomaly, isUpperAnomaly)

        return anomaly_frame

    def generate_spectral_score(self, mags):
        ave_mag = average_filter(mags, n=self.__score_window)
        safeDivisors = np.clip(ave_mag, EPS, ave_mag.max())

        raw_scores = np.abs(mags - ave_mag) / safeDivisors
        scores = np.clip(raw_scores / 10.0, 0, 1.0)

        return scores

    def spectral_residual_transform(self, values):
        """
        This method transform a time series into spectral residual series
        :param values: list.
            a list of float values.
        :return: mag: list.
            a list of float values as the spectral residual values
        """

        trans = np.fft.fft(values)
        mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
        eps_index = np.where(mag <= EPS)[0]
        mag[eps_index] = EPS

        mag_log = np.log(mag)
        mag_log[eps_index] = 0

        # average filter is h_q(f), mag window is a hyperparameter
        spectral = np.exp(mag_log - average_filter(mag_log, n=self.__mag_window))

        trans.real = trans.real * spectral / mag
        trans.imag = trans.imag * spectral / mag
        trans.real[eps_index] = 0
        trans.imag[eps_index] = 0

        return mag

    @staticmethod
    def predict_next(values):
        """
        Predicts the next value by sum up the slope of the last value with previous values.
        Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
        where g(x_i,x_j) = (x_i - x_j) / (i - j)
        :param values: list.
            a list of float numbers.
        :return : float.
            the predicted next value.
        """

        if len(values) <= 1:
            raise ValueError(f'data should contain at least 2 numbers')

        try:
            # list case
            v_last = values[-1]
            n = len(values)

            slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]
            return values[1] + sum(slopes)
        except:
            # pandas series case
            v_last = values.iloc[-1]
            n = values.shape[0]
            slopes = [(v_last - v) / (n - 1 - i) for i, v in values.iloc[:-1].items()]

            return values.iloc[1] + sum(slopes)

    @staticmethod
    def extend_series(values, extend_num=5, look_ahead=5):
        """
        extend the array data by the predicted next value
        :param values: list.
            a list of float numbers.
        :param extend_num: int, default 5.
            number of values added to the back of data.
        :param look_ahead: int, default 5.
            number of previous values used in prediction.
        :return: list.
            The result array.
        """

        if look_ahead < 1:
            raise ValueError('look_ahead must be at least 1')

        # here extend num is a hyperparameter = 5 (tau or m)
        extension = [SpectralResidual.predict_next(values[-look_ahead - 2:-1])] * extend_num
        return np.concatenate((values, extension), axis=0)

    @staticmethod
    def calculate_expected_value(values, anomaly_index):
        values = deanomaly_entire(values, anomaly_index)
        length = len(values)
        fft_coef = np.fft.fft(values)
        fft_coef.real = [v if length * 3 / 8 >= i or i >= length * 5 / 8 else 0 for i, v in enumerate(fft_coef.real)]
        fft_coef.imag = [v if length * 3 / 8 >= i or i >= length * 5 / 8 else 0 for i, v in enumerate(fft_coef.imag)]
        exps = np.fft.ifft(fft_coef)
        return exps.real
