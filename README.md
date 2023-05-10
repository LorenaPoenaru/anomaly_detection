# How to Retrain? Techniques for Anomaly Detection Models for AIOps Data

This repo includes signal reconstruction models and retraining twchniques for univariate time series data.
Our work attempts to investigate the most applicable maintanance approach for various kinds of signal reconstruction models on online streaming data.


### Project structure:
```
.
├── datasets                    extract you data heretru. Each behcnmark folder includes TS for dry run
│   ├── NAB_realAWSCloudwatch
│   └── Yahoo_A1Benchmark  
├── results                     Results (as .csvs  and plots) get saved to this folder
│   ├── fedd_results            Results of a FEDD drift detector (reusable for retraining)
│   ├── scores    
│   └── imgs     
├── dynamic_models              Dynamic signal reconstruction models: LSTM AE, SR CNN. See README within folder
├── static_models               Static signal reconstruction models: FFT, PCI, SR. See README within folder
├── utils                       utils used during training
│   ├── drift_detection
│   └── sr
```

### Datasets

* `NAB`: https://github.com/numenta/NAB/tree/master/data
* `Yahoo`: https://yahooresearch.tumblr.com/post/114590420346/a-benchmark-dataset-for-time-series-anomaly
Please, send a request through the official website to obtain Yahoo data

### Drift detectors

* `FEDD` 
We use own adaptation of https://github.com/GustavoHFMO/IDPSO-ELM-S/blob/008477e80c37ed5d0ff7f2d75394d85542b046c0/detectores/FEDD.py

### Models
 #### Static
* `PCI` 

    *Yufeng, Y., Zhu, Y., Li, S., Wan, D.: Time series outlier detection based
    on sliding window prediction. Mathematical Problems in Engineering
    2014 (10 2014)*

    We use an adoptation of https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/pci . Our adoptation sits in utils/pci.py
* `FFT`

    *Rasheed, F., Peng, P., Alhajj, R., Rokne, J.: Fourier transform based spa-
    tial outlier mining. In: Proceedings of the 10th International Conference
    on Intelligent Data Engineering and Automated Learning. p. 317–324
    (2009)*

    We use an adoptation of https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/fft . Our adoptation sits in utils/fft.py
* `SR`

    *Ren, H., Xu, B., Wang, Y., Yi, C., Huang, C., Kou, X., Xing, T.,
    Yang, M., Tong, J., Zhang, Q.: Time-series anomaly detection service
    at microsoft. In: Proceedings of the 25th ACM SIGKDD Interna-
    tional Conference on Knowledge Discovery &amp; Data Mining. p.
    3009–3017. KDD ’19, Association for Computing Machinery, New
    York, NY, USA (2019). https://doi.org/10.1145/3292500.3330680, https:
    //doi.org/10.1145/3292500.3330680*

    We use https://github.com/y-bar/ml-based-anomaly-detection

 #### Dynamic
* `LSTM AE` 

    *Provotar, O., Linder, Y.M., Veres, M.: Unsupervised anomaly detection
    in time series using lstm-based autoencoders. 2019 IEEE International
    Conference on Advanced Trends in Information Theory (ATIT) pp. 513–
    517 (2019)*

    We use own implementation

* `SR CNN`

    *Ren, H., Xu, B., Wang, Y., Yi, C., Huang, C., Kou, X., Xing, T.,
    Yang, M., Tong, J., Zhang, Q.: Time-series anomaly detection service
    at microsoft. In: Proceedings of the 25th ACM SIGKDD Interna-
    tional Conference on Knowledge Discovery &amp; Data Mining. p.
    3009–3017. KDD ’19, Association for Computing Machinery, New
    York, NY, USA (2019). https://doi.org/10.1145/3292500.3330680, https:
    //doi.org/10.1145/3292500.3330680*

    We adopt https://github.com/microsoft/anomalydetector

