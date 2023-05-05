# Experiments instructions

Please, run single jupyter notebook for the whole set of experiments on static models. This type of models does not require re-training sicne they do not pre-train on a separate set.

## The models included are:

* `PCI`
We use an adoptation of https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/pci . Our adoptation sits in utils/pci.py
* `FFT` 
We use an adoptation of https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/fft . Our adoptation sits in utils/fft.py
* `SR`
We use https://github.com/y-bar/ml-based-anomaly-detection

We install it via pip in the notebook
`%pip install sranodec`

We also tried original implementation from https://github.com/microsoft/anomalydetector (msanomalydetector). To install, please, run setup.py from the SR adopted folder placed in utils
 `pip install utils/sr/.` 
 You can then import from `msanomalydetector` in the notebook.
