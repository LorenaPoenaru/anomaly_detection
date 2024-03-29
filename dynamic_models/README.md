# Experiments overview

## Data

Datasets should be stored into folders 'NAB_realAWSClodwatch' and 'Yahoo_A1Benchmark'.

## Models Specifications

LSTM-AE does not require parameter changes when running on Yahoo or NAB. SR-CNN does require the following parameter changes:
- lr = 1e-5 for NAB and lr = 1e-6 for Yahoo. The lr parameter is given at the beginning of Main.

When running SR-CNN, create a folder called 'snapshot' where the trained model will be saved.

Besides the parameters, the original SR-CNN implementation was changed to accomodate our experiments as follows:
- changed function _get_path_;
- changed data reading from predefined functions to pandas;
- results collected in a dataframe;
- changed one parameter in the loss function in the in the srcnn/utils.py;
- changed the code such that it can be run on CPU instead of GPU with CUDA;
- merged generate_data.py, train.py and evalue.py into one script.


## data_preparation_srcnn

The data for SR-CNN needs to be split into a folder called train and a folder called test.
Run this notebook to prepare data for SR-CNN.

## lstmae_fedd_FH

Training routine for LSTM AE with FEDD drift detector and Full History retraining approach

## lstmae_fedd_SW

Training routine for LSTM AE with FEDD drift detector and Sliding Window retraining approach

## lstmae_periodic_FH

Training routine for LSTM AE with periodic Full History retraining approach

## lstmae_periodic_SW

Training routine for LSTM AE with periodic Sliding Window retraining approach

## lstmae_Static

Training routin for LSTM AE without retraining

## srcnn_periodic_FH

Training routine for SR CNN with periodic Full History retraining approach

## srcnn_periodic_SW

Training routine for SR CNN with periodic Sliding Window retraining approach

## srcnn_Static

Training routine for SR CNN without retraining
