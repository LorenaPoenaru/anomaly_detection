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
We use own adaptation of https://github.com/GustavoHFMO/IDPSO-ELM-S/blob/008477e80c37ed5d0ff7f2d75394d85542b046c0/detectores/FEDD.py that can be found under utils/drift_detection

### Models
 ### Static
 * `PCI` https://www.hindawi.com/journals/mpe/2014/879736/
 * `FFT` https://dl.acm.org/doi/10.5555/1789574.1789615
 * `SR` https://arxiv.org/pdf/1906.03821.pdf
