import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from fuzzy import SimuladorPendulo, ParametrosFisicos
import random
import matplotlib.pyplot as plt

class ControladorNeuroFuzzy:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=1000,
            learning_rate_init=0.001,
            batch_size=64,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        self.setup_fuzzy_system()
        self.criar_regras()
    
    def setup_fuzzy_system(self):
        self.angulo = ctrl.Antecedent(np.linspace(-1, 1, 100), 'angulo')
        self.vel_angular = ctrl.Antecedent(np.linspace(-1, 1, 100), 'vel_angular')
        self.posicao_carro = ctrl.Antecedent(np.linspace(-1, 1, 100), 'posicao_carro')
        self.vel_carro = ctrl.Antecedent(np.linspace(-1, 1, 100), 'vel_carro')
        self.forca = ctrl.Consequent(np.linspace(-1, 1, 100), 'forca')
        
        self.angulo['N'] = fuzz.trimf(self.angulo.universe, [-1, -1, 0])
        self.angulo['Z'] = fuzz.trimf(self.angulo.universe, [-0.5, 0, 0.5])
        self.angulo['P'] = fuzz.trimf(self.angulo.universe, [0, 1, 1])
        
        self.vel_angular['N'] = fuzz.trimf(self.vel_angular.universe, [-1, -1, 0])
        self.vel_angular['Z'] = fuzz.trimf(self.vel_angular.universe, [-0.5, 0, 0.5])
        self.vel_angular['P'] = fuzz.trimf(self.vel_angular.universe, [0, 1, 1])
        
        self.posicao_carro['N'] = fuzz.trimf(self.posicao_carro.universe, [-1, -1, 0])
        self.posicao_carro['Z'] = fuzz.trimf(self.posicao_carro.universe, [-0.5, 0, 0.5])
        self.posicao_carro['P'] = fuzz.trimf(self.posicao_carro.universe, [0, 1, 1])
        
        self.vel_carro['N'] = fuzz.trimf(self.vel_carro.universe, [-1, -1, 0])
        self.vel_carro['Z'] = fuzz.trimf(self.vel_carro.universe, [-0.5, 0, 0.5])
        self.vel_carro['P'] = fuzz.trimf(self.vel_carro.universe, [0, 1, 1])
        
        self.forca['FN'] = fuzz.trimf(self.forca.universe, [-1, -1, -0.6])
        self.forca['N'] = fuzz.trimf(self.forca.universe, [-0.8, -0.4, 0])
        self.forca['Z'] = fuzz.trimf(self.forca.universe, [-0.2, 0, 0.2])
        self.forca['P'] = fuzz.trimf(self.forca.universe, [0, 0.4, 0.8])
        self.forca['FP'] = fuzz.trimf(self.forca.universe, [0.6, 1, 1])
        
        regras = self.criar_regras()
        
        self.sistema_controle = ctrl.ControlSystem(regras)
        self.simulador = ctrl.ControlSystemSimulation(self.sistema_controle)
    
    def gerar_dados_treinamento(self, n_samples=5000):
        X = []
        y = []
        
        for _ in range(n_samples):
            angulo = random.uniform(-0.2, 0.2)
            vel_angular = random.uniform(-0.3, 0.3)
            posicao = random.uniform(-0.3, 0.3)
            vel_carro = random.uniform(-0.3, 0.3)
            
            estado = [angulo, vel_angular, posicao, vel_carro]
            saida_fuzzy = self.calcular_saida_fuzzy(estado)
            
            X.append(estado)
            y.append([saida_fuzzy])
            
            if abs(angulo) < 0.05:
                for _ in range(3):
                    estado_var = [
                        angulo + random.gauss(0, 0.005),
                        vel_angular + random.gauss(0, 0.02),
                        posicao + random.gauss(0, 0.02),
                        vel_carro + random.gauss(0, 0.02)
                    ]
                    saida_var = self.calcular_saida_fuzzy(estado_var)
                    X.append(estado_var)
                    y.append([saida_var])
        
        return np.array(X), np.array(y)
    
    def treinar(self):
        try:
            print("Gerando dados de treinamento...")
            X, y = self.gerar_dados_treinamento()
            
            print("Normalizando dados...")
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y)
            
            print("Treinando rede neural...")
            self.mlp.fit(X_scaled, y_scaled)
            
            score = self.mlp.score(X_scaled, y_scaled)
            print(f"Score de treinamento: {score:.4f}")
            
        except Exception as e:
            print(f"Erro durante o treinamento: {str(e)}")
            raise
    
    def calcular_controle(self, estado):
        try:
            estado_norm = [
                np.clip(estado[0], -0.2, 0.2),
                np.clip(estado[1], -0.3, 0.3),
                np.clip(estado[2], -0.3, 0.3),
                np.clip(estado[3], -0.3, 0.3)
            ]
            
            saida_fuzzy = self.calcular_saida_fuzzy(estado_norm)
            estado_scaled = self.scaler_X.transform([estado_norm])
            saida_neural = self.scaler_y.inverse_transform(
                self.mlp.predict(estado_scaled).reshape(-1, 1)
            )[0][0]
            
            erro_pendulo = abs(estado_norm[0])
            alpha = np.exp(-20 * erro_pendulo)
            
            if erro_pendulo > 0.1:
                return saida_fuzzy
            
            saida_combinada = alpha * saida_neural + (1 - alpha) * saida_fuzzy
            return np.clip(saida_combinada, -1, 1)
            
        except Exception as e:
            print(f"Erro no cálculo do controle: {e}")
            return self.calcular_saida_fuzzy(estado)

    def criar_regras(self):
        regras = [
            ctrl.Rule(self.angulo['N'] & self.vel_angular['N'], self.forca['FN']),
            ctrl.Rule(self.angulo['N'] & self.vel_angular['Z'], self.forca['N']),
            ctrl.Rule(self.angulo['N'] & self.vel_angular['P'], self.forca['Z']),
            ctrl.Rule(self.angulo['Z'] & self.vel_angular['N'], self.forca['N']),
            ctrl.Rule(self.angulo['Z'] & self.vel_angular['Z'], self.forca['Z']),
            ctrl.Rule(self.angulo['Z'] & self.vel_angular['P'], self.forca['P']),
            ctrl.Rule(self.angulo['P'] & self.vel_angular['N'], self.forca['Z']),
            ctrl.Rule(self.angulo['P'] & self.vel_angular['Z'], self.forca['P']),
            ctrl.Rule(self.angulo['P'] & self.vel_angular['P'], self.forca['FP']),
            
            ctrl.Rule(self.posicao_carro['N'] & self.vel_carro['N'], self.forca['FP']),
            ctrl.Rule(self.posicao_carro['N'] & self.vel_carro['Z'], self.forca['P']),
            ctrl.Rule(self.posicao_carro['N'] & self.vel_carro['P'], self.forca['Z']),
            ctrl.Rule(self.posicao_carro['Z'] & self.vel_carro['N'], self.forca['P']),
            ctrl.Rule(self.posicao_carro['Z'] & self.vel_carro['Z'], self.forca['Z']),
            ctrl.Rule(self.posicao_carro['Z'] & self.vel_carro['P'], self.forca['N']),
            ctrl.Rule(self.posicao_carro['P'] & self.vel_carro['N'], self.forca['Z']),
            ctrl.Rule(self.posicao_carro['P'] & self.vel_carro['Z'], self.forca['N']),
            ctrl.Rule(self.posicao_carro['P'] & self.vel_carro['P'], self.forca['FN'])
        ]
        return regras

    def calcular_saida_fuzzy(self, estado):
        try:
            self.simulador.input['angulo'] = estado[0]
            self.simulador.input['vel_angular'] = estado[1]
            self.simulador.input['posicao_carro'] = estado[2]
            self.simulador.input['vel_carro'] = estado[3]
            
            self.simulador.compute()
            
            return self.simulador.output['forca']
        except Exception as e:
            return 0.0

def main():
    controlador = ControladorNeuroFuzzy()
    print("Treinando controlador neuro-fuzzy...")
    controlador.treinar()

    simulador = SimuladorPendulo()
    simulador.controlador = controlador
    
    print("Executando simulação...")
    simulador.simular()
    
    metricas = simulador.calcular_metricas()
    print("\nMétricas do Sistema Neuro-Fuzzy:")
    for nome, valor in metricas.items():
        print(f"{nome}: {valor:.4f}")
    
    simulador.plotar_resultados()

if __name__ == "__main__":
    main()