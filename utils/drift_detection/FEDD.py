# -*- coding: utf-8 -*-
'''
Created on 10 de mar de 2017

@author: gusta
'''

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf


# moving window class
class Janela():
    def __init__(self):
        '''
        Classe para instanciar a janela deslizante
        '''
        self.dados = []
        self.dados_mais = []

    def Ajustar(self, valores):
        '''
        Metodo para ajustar o tamanho da jenela deslizante
        :param valores: valores para serem inseridos na janela
        '''

        self.dados = valores
        self.dados_mais = np.append([0], valores)

    def Add_janela(self, valor):
        '''
        Metodo para inserir na janela deslizante, o valor mais antigo sera excluido
        :param valor: valor de entrada
        '''
        self.dados = self.Fila(self.dados, valor)
        self.dados_mais = self.Fila(self.dados_mais, valor)

    def Fila(self, lista, valor):
        '''
        metodo para adicionar um novo valor a um ndarray
        :param: lista: lista que será acrescida
        :param: valor: valor a ser adicionado
        :return: retorna a lista com o valor acrescido
        '''

        if (len(lista) == 1):
            aux2 = len(lista[0])
            aux = [0] * aux2
            aux[len(lista[0]) - 1] = valor
            aux[:len(aux) - 1] = lista[0][1:]
            lista[0] = aux
            lista[0] = np.asarray(lista[0])
            lista[0] = np.column_stack(lista[0])

            return lista

        else:
            aux2 = len(lista)
            aux = [0] * aux2
            aux[len(lista) - 1] = valor
            aux[:len(aux) - 1] = lista[1:]
            lista = aux
            lista = np.asarray(lista)
            lista = np.column_stack(lista)

            return lista

    def Increment_Add(self, valor):
        '''
        Metodo para inserir mais dados na janela deslizante
        :param valor: valor de entrada
        '''

        if (len(self.dados) > 0):
            aux = np.asarray(self.dados)
            aux = [0] * (len(self.dados) + 1)
            aux[:len(self.dados)] = self.dados
            aux[len(self.dados)] = valor
            self.dados = aux
        else:
            self.dados.append(valor)

    def Zerar_Janela(self):
        self.dados = []


class FEDD():
    def __init__(self, Lambda=0.2, w=0.25, c=0.25):
        '''
        Método para criar um modelo do ECDD
        :param Lambda: float com o valor de lambda
        :param w: float com o nivel de alarme
        :param c float com o nivel de deteccao
        '''
        self.Lambda = Lambda
        self.w = w
        self.c = c
        self.vetor_caracteristicas_inicial = 0
        self.media_zero = 0
        self.desvio_zero = 0
        self.desvio_z = 0
        self.zt = 0
        self.below_warn = 0
        self.warn = 0
        self.nada = "Nada"
        self.alerta = "Alerta"
        self.mudanca = "Mudanca"
        self.sensor_mudanca = True
        self.n = 300

        ##########################

        self.drift_happened = False
        self.drift_in_whole_window = False

        self.alarmes = []
        self.deteccoes = []

    def reset(self):
        self.vetor_caracteristicas_inicial = 0
        self.media_zero = 0
        self.desvio_zero = 0
        self.desvio_z = 0
        self.zt = 0
        self.below_warn = 0
        self.warn = 0
        self.sensor_mudanca = True
        self.drift_happened = False

    def armazenar_conceito(self, vetor_caracteristicas_inicial, MI0, SIGMA0):
        '''
        Este metodo tem por objetivo armazenar um conceito de um erro
        :param caracteristicas_iniciais: vetor de caracteristicas iniciais
        :param MI0: media dos erros
        :param SIGMA0: desvio dos erros
        '''
        self.vetor_caracteristicas_inicial = vetor_caracteristicas_inicial
        self.media_zero = MI0
        self.desvio_zero = SIGMA0

    def atualizar_ewma(self, erro, t):
        '''
        método para atualizar o ewma com o erro atual
        :param erro: double com o erro para ser verificado
        :param t: instante de tempo
        '''

        # calculando a m�dia movel
        if (t == 1):
            self.zt = (1 - self.Lambda) * self.media_zero + self.Lambda * erro
        elif (self.sensor_mudanca == True):
            self.sensor_mudanca = False
            self.zt = (1 - self.Lambda) * self.media_zero + self.Lambda * erro
        else:
            self.zt = (1 - self.Lambda) * self.zt + self.Lambda * erro

        # calculando o desvio da m�dia movel
        parte1 = (self.Lambda / (2 - self.Lambda))
        parte2 = (1 - self.Lambda)
        parte3 = (2 * t)
        parte4 = (1 - (parte2**parte3))
        parte5 = (parte1 * parte4 * self.desvio_zero)
        self.desvio_z = np.sqrt(parte5)

    def monitorar(self):
        '''
        Método para consultar a condicao de deteccao do FEDD
        '''

        # consultando as regras
        if (self.zt > self.media_zero + (self.c * self.desvio_z)):
            self.sensor_mudanca = True
            # self.below_warn = 0
            return self.mudanca

        elif (self.zt > self.media_zero + (self.w * self.desvio_z)):
            # self.below_warn += 1

            # if(self.below_warn == 10):
            #    self.below_warn = 0

            return self.alerta

        else:
            return self.nada

    def teste_estacionariedade(self, timeseries):
        '''
        Este metodo tem por testar a estacionariedade de uma serie com o teste adfuller
        :param: timeseries: serie temporal, array
        :return: print com as estatisticas do teste
        '''

        # Determing rolling statistics
        timeseries = pd.DataFrame(timeseries)
        rolmean = timeseries.rolling(window=12, center=False).mean()
        rolstd = timeseries.rolling(window=12, center=False).std()

        # Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        timeseries = timeseries[1:].values
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic',
                                    'p-value',
                                    '#Lags Used',
                                    'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

        # Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()

    def FE(self, serie_atual):
        '''
        Método para fazer a diferenciacao de uma serie_atual
        :param serie_atual: serie_atual real
        '''

        # serie_df = pd.DataFrame(serie_atual)
        serie_diff = pd.Series(serie_atual)
        serie_diff = serie_diff - serie_diff.shift()
        serie_diff = serie_diff[1:]

        features = []

        # feature 1:
        auto_correlacao = acf(serie_diff, nlags=5)
        for i in auto_correlacao:
            features.append(i)

        # feature 2:
        parcial_atcorr = pacf(serie_diff, nlags=5)
        for i in parcial_atcorr:
            features.append(i)

        # feature 3:
        variancia = serie_diff.std()
        features.append(variancia)

        # feature 4:
        serie_skew = serie_diff.skew()
        features.append(serie_skew)

        # feature 5:
        serie_kurtosis = serie_diff.kurtosis()
        features.append(serie_kurtosis)

        # feature 6:
        turning_p = self.turningpoints(serie_diff)
        features.append(turning_p)

        # feature 7:

        # feature 8:

        return features

    def turningpoints(self, lst):
        dx = np.diff(lst)
        return np.sum(dx[1:] * dx[:-1] < 0)

    def computar_distancia(self, vetor1, vetor2):
        '''
        Método para computar a correlacao de pearson entre dois vetores
        :param vetor1: vetor de caracteristicas inicial
        :param vetor2: vetor de caracteristicas atual
        :return: distancia
        '''

        # correlacao = pearsonr(vetor1, vetor2)
        # distancia = correlacao[0]

        distancia = cosine(vetor1, vetor2)

        return distancia

    def train_wrapper(self, data_train, for_training=True):

        # ajustando com os dados finais do treinamento a janela de predicao
        self.janela_predicao = Janela()
        self.janela_predicao.Ajustar(data_train[len(data_train) - 1:])

        # janela com o atual conceito, tambem utilizada para armazenar os dados
        # de retreinamento
        if for_training:
            self.janela_caracteristicas = Janela()
            self.janela_caracteristicas.Ajustar(data_train)

        final = len(self.janela_caracteristicas.dados)
        qtd = 3
        vetor_caracteristicas_0 = self.FE(
            self.janela_caracteristicas.dados[:final - qtd])

        distancias_vetor = []
        for i in range(1, qtd):
            vetor_caracteristicas = self.FE(
                self.janela_caracteristicas.dados[i:final - qtd + i])
            distancia = self.computar_distancia(
                vetor_caracteristicas_0, vetor_caracteristicas)
            distancias_vetor.append(distancia)

        self.armazenar_conceito(
            vetor_caracteristicas_0,
            np.mean(distancias_vetor),
            np.std(distancias_vetor))

    def test_wrapper(self, stream):
        self.drift_in_whole_window = False

        for i in range(len(stream)):
            if not self.drift_happened:

                # atualizar a janela de caracteristicas do FEDD
                self.janela_caracteristicas.Add_janela(stream[i])

                # realizando a diferenciacao no vetor de caracteristicas atuais
                vetor_caracteristicas_atual = self.FE(
                    self.janela_caracteristicas.dados[0])

                # computar a distancia entre os vetores de caracteristicas
                distancia = self.computar_distancia(
                    self.vetor_caracteristicas_inicial, vetor_caracteristicas_atual)
                # distancias_vetor.append(distancia)

                # atualizando o media_zt e desvio_zt
                self.atualizar_ewma(distancia, i + 1)
                # zt_vetor.append(fedd.zt)

                # monitorando o erro
                string_fedd = self.monitorar()

                if (string_fedd == self.mudanca):
                    self.deteccoes.append(i)

                    self.drift_in_whole_window = True
                    self.drift_happened = True

            else:

                if (i < self.deteccoes[len(self.deteccoes) - 1] + self.n):

                    # adicionando a nova instancia na janela de caracteristicas
                    self.janela_caracteristicas.Add_janela(stream[i])

                else:

                    # atualizar por fedd
                    self.reset()

                    self.train_wrapper(stream[-2:-1], for_training=False)

                    # variavel para voltar para o loop principal
                    self.drift_happened = False

        return self.drift_in_whole_window
