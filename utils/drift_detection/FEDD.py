import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf


class FEDD():
    def __init__(self, Lambda, w, c):
        '''
        Method to create an ECDD model
        :param Lambda: float with the value of lambda
        :param w: float with the alert level
        :param c float with the detection level
        '''
        self.Lambda = Lambda
        self.w = w
        self.c = c
        self.initial_feature_vector = 0 
        self.average_zero = 0
        self.deviation_zero = 0
        self.deviation_z = 0
        self.zt = 0
        self.below_warn = 0
        self.warn = 0
        self.nodrift = "NoDrift"
        self.alert = "Alert"
        self.change = "Change"
        self.change_detector = True
    
    def storing_agv_dev(self, initial_feature_vector, MI0, SIGMA0):
        '''
        This method aims to store a concept of an error
        :param initial_feature_vector: initial feature vector
        :param MI0: average of errors
        :param SIGMA0: deviation of errors
        '''
        self.initial_feature_vector = initial_feature_vector
        self.average_zero = MI0
        self.deviation_zero = SIGMA0
        #print(MI0)
        #print(SIGMA0)
    
    def update_ewma(self, error, t):
        '''
        Function for calculating the EWMA estimators based on the equations in the paper
        :param error: the error to be verified
        :param t: timestamp
        '''
        
        
        # calculating the moving average
        if(t == 1):
            self.zt = (1-self.Lambda) * self.average_zero + self.Lambda * error
        elif(self.change_detector == True):
            self.change_detector = False
            self.zt = (1-self.Lambda) * self.average_zero + self.Lambda * error
        else:
            self.zt = (1-self.Lambda) * self.zt + self.Lambda * error
        
        
        # calculating the deviation of the moving average
        parte1 = (self.Lambda/(2-self.Lambda))
        parte2 = (1-self.Lambda)
        parte3 = (2*t)
        parte4 = (1 - (parte2**parte3))
        parte5 = (parte1 * parte4 * self.deviation_zero)
        self.deviation_z = np.sqrt(parte5)

    
    def monitoring(self):
        '''
        Function to check whether concept drift was detected
        '''

        # condition for detecting drift
        if(self.zt > self.average_zero + (self.c * self.deviation_z)):
            self.change_detector = True
            #self.below_warn = 0
            return self.change
        
        # condition for detecting an warning
        elif(self.zt > self.average_zero + (self.w * self.deviation_z)):
            #self.below_warn += 1
            
            #if(self.below_warn == 10):
            #    self.below_warn = 0
            
            return self.alert
        
        # condition for not detecting anything
        else:   
            return self.nodrift
        
    def teste_estacionariedade(self, timeseries):
        
        '''
        This method tests the stationarity of a series with Dickey-Fuller test
        :param: timeseries: timeseries array
        :return: print the test statistics
        '''
        
        #Determing rolling statistics
        timeseries = pd.DataFrame(timeseries)
        rolmean = timeseries.rolling(window=12, center=False).mean()
        rolstd = timeseries.rolling(window=12, center=False).std()
            
        #Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        timeseries = timeseries[1:].values
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
        
    
        #Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()
    
    def FE(self, time_series):
        '''
        Extracting the TS features
        :param serie_atual: time_series
        '''  
        
        # Differencing time series to soften the influence of trends
        
        serie_diff = pd.Series(time_series)
        serie_diff = serie_diff - serie_diff.shift()
        serie_diff = serie_diff[1:]

        
        
        features = []
        
        #feature 1:
        auto_correlation = acf(serie_diff, nlags=5)
        for i in auto_correlation:
            features.append(i)
        
        #feature 2:
        partial_autocorrelation = pacf(serie_diff, nlags=5)
        for i in partial_autocorrelation:
            features.append(i)
        
        #feature 3:
        variance = serie_diff.std()
        features.append(variance)
        
        #feature 4:
        skewness_coeff = serie_diff.skew()
        features.append(skewness_coeff)

        #feature 5:
        kurtosis_coeff = serie_diff.kurtosis()
        features.append(kurtosis_coeff)
        
        #feature 6:
        turning_p = self.turningpoints(serie_diff)
        features.append(turning_p)
        
        #feature 7:
        
        #feature 8:
        
        return features
    
    def turningpoints(self, lst):
        dx = np.diff(lst)
        return np.sum(dx[1:] * dx[:-1] < 0)
    
    def compute_distance(self, array_1, array_2):
        '''
        Method to compute the cosine distance between two arrays
        :param arrays1: arrays of initial features
        :param arrays2: arrays of current features
        :return: distancia
        '''  
        
        
        distance = cosine(array_1, array_2)
        
        '''
        Method to compute the Pearson Distance between two arrays
        '''
        #correlation = pearsonr(array_1, array_2)
        #distance = correlation[0]
        
        return distance
    
    
def main():
    '''
    dtst = Datasets()
    dataset = dtst.Leitura_dados(dtst.bases_linear_graduais(2, 30), excel=True)


    #particao = Particionar_series(dataset, [0.0, 0.0, 0.0], 0)
    #dataset = particao.Normalizar(dataset)
    
    fedd = FEDD(0.2, 0.75, 1)
    feature = fedd.FE(dataset[5300:5601], dataset[0:300])
    print(feature)
    
    
    '''
    
if __name__ == "__main__":
    main()

