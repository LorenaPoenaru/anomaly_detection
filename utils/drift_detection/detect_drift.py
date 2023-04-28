from drift_detection.ETFE import ETFE
import subprocess
import pandas as pd
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri


class DriftDetector(ETFE):

    def __init__(self, entropy_type='SampEn'):
        super().__init__(entropy_type)
        entropy_delta = None

    def process_train_data(self, data):
        for point in data:
            entropy = self.feed(point)

    def is_drifted(self, window):
        entropies = []

        for point in window:
            entropy = self.feed(point)
            if entropy:
                entropies.append(entropy)

        print(entropies)
        df = pd.DataFrame([])
        df['entropy'] = entropies
        df.to_csv('entropies.csv')
        if entropies:

            glr_test_stat = subprocess.call("Rscript GLR.R", shell=True)
            print(glr_test_stat)
            if glr_test_stat > self.entropy_delta:
                # drift occured
                return True
        return False
    
    
if __name__ == '__main__':
    dd = DriftDetector()
    dd.is_drifted([206,
                    13,
                    130,
                    37,
                    169,
                    105,
                    202,
                    146,
                    0,
                    97,
                    205,
                    168,
                    154,
                    18,
                    49,
                    58,
                    38,
                    22,
                    145,
                    90,
                    110,
                    212,
                    120,
                    108,
                    237,
                    127,
                    188,
                    59,
                    124,
                    145])