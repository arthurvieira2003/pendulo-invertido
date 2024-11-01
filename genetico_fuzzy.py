import numpy as np
from deap import base, creator, tools, algorithms
import random
from fuzzy import SimuladorPendulo, ParametrosFisicos
import skfuzzy as fuzz
from skfuzzy import control as ctrl

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class ControladorGeneticoFuzzy:
    def __init__(self):
        self.n_params = 51
        self.pop_size = 300
        self.n_generations = 150
        self.setup_genetic_algorithm()
        
    def setup_genetic_algorithm(self):
        self.toolbox = base.Toolbox()
        
        def criar_trio_ordenado():
            centro = random.uniform(-0.8, 0.8)
            spread = random.uniform(0.2, 0.4)
            return [centro - spread, centro, centro + spread]
        
        self.toolbox.register("attr_trio", criar_trio_ordenado)
        
        def criar_individuo():
            return [val for _ in range(17) for val in criar_trio_ordenado()]
        
        self.toolbox.register("individual", tools.initIterate, creator.Individual, criar_individuo)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.avaliar_individuo)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutacao_customizada)
        self.toolbox.register("select", tools.selTournament, tournsize=7)
    
    def mutacao_customizada(self, individual, indpb=0.1):
        for i in range(0, len(individual), 3):
            if random.random() < indpb:
                delta = random.gauss(0, 0.3)
                individual[i] = max(min(individual[i] + delta, 1), -1)
                individual[i+1] = max(min(individual[i+1] + delta, 1), -1)
                individual[i+2] = max(min(individual[i+2] + delta, 1), -1)
                individual[i:i+3] = sorted(individual[i:i+3])
        return individual,
    
    def criar_sistema_fuzzy(self, params):
        params_ordenados = []
        idx = 0
        while idx < len(params):
            trio = sorted(params[idx:idx+3])
            params_ordenados.extend(trio)
            idx += 3
        
        angulo = ctrl.Antecedent(np.linspace(-1, 1, 100), 'angulo')
        vel_angular = ctrl.Antecedent(np.linspace(-1, 1, 100), 'vel_angular')
        posicao_carro = ctrl.Antecedent(np.linspace(-1, 1, 100), 'posicao_carro')
        vel_carro = ctrl.Antecedent(np.linspace(-1, 1, 100), 'vel_carro')
        forca = ctrl.Consequent(np.linspace(-1, 1, 100), 'forca')
        
        idx = 0
        
        angulo['N'] = fuzz.trimf(angulo.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        angulo['Z'] = fuzz.trimf(angulo.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        angulo['P'] = fuzz.trimf(angulo.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        
        vel_angular['N'] = fuzz.trimf(vel_angular.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        vel_angular['Z'] = fuzz.trimf(vel_angular.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        vel_angular['P'] = fuzz.trimf(vel_angular.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        
        posicao_carro['N'] = fuzz.trimf(posicao_carro.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        posicao_carro['Z'] = fuzz.trimf(posicao_carro.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        posicao_carro['P'] = fuzz.trimf(posicao_carro.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        
        vel_carro['N'] = fuzz.trimf(vel_carro.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        vel_carro['Z'] = fuzz.trimf(vel_carro.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        vel_carro['P'] = fuzz.trimf(vel_carro.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        
        forca['FN'] = fuzz.trimf(forca.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        forca['N'] = fuzz.trimf(forca.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        forca['Z'] = fuzz.trimf(forca.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        forca['P'] = fuzz.trimf(forca.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        idx += 3
        forca['FP'] = fuzz.trimf(forca.universe, [params_ordenados[idx], params_ordenados[idx+1], params_ordenados[idx+2]])
        
        regras = self.criar_regras(angulo, vel_angular, posicao_carro, vel_carro, forca)
        
        sistema_controle = ctrl.ControlSystem(regras)
        return ctrl.ControlSystemSimulation(sistema_controle)
    
    def avaliar_individuo(self, individual):
        try:
            sistema_fuzzy = self.criar_sistema_fuzzy(individual)
            simulador = SimuladorPendulo()
            simulador.sistema_fuzzy = sistema_fuzzy
            simulador.simular()
            metricas = simulador.calcular_metricas()
            
            erro_theta_norm = min(metricas['erro_max_theta'] / 0.5, 2.0)
            erro_pos_norm = min(metricas['erro_max_posicao'] / 0.3, 2.0)
            energia_norm = min(metricas['energia_controle'] / 100.0, 2.0)
            tempo_norm = min(metricas['tempo_estabilizacao'] / 5.0, 2.0)
            
            penalidade = 0
            for i in range(0, len(individual), 3):
                if not (individual[i] <= individual[i+1] <= individual[i+2]):
                    penalidade += 5
                if i > 0 and i % 9 == 0:
                    if individual[i] - individual[i-3] < -0.8:
                        penalidade += 2
            
            fitness = (
                erro_theta_norm * 4.0 +
                erro_pos_norm * 2.0 +
                energia_norm * 0.5 +
                tempo_norm * 1.0 +
                penalidade
            )
            
            return (fitness,)
        except Exception as e:
            return (100.0,)
    
    def otimizar(self):
        pop = self.toolbox.population(n=self.pop_size)
        
        hof = tools.HallOfFame(10)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        pop, logbook = algorithms.eaSimple(pop, self.toolbox,
                                         cxpb=0.6,
                                         mutpb=0.3,
                                         ngen=self.n_generations,
                                         stats=stats,
                                         halloffame=hof,
                                         verbose=True)
        
        return hof[0]

    def criar_regras(self, angulo, vel_angular, posicao_carro, vel_carro, forca):
        regras = []
        
        regras.extend([
            ctrl.Rule(angulo['N'] & vel_angular['N'], forca['FN']),
            ctrl.Rule(angulo['N'] & vel_angular['Z'], forca['N']),
            ctrl.Rule(angulo['N'] & vel_angular['P'], forca['Z']),
            ctrl.Rule(angulo['Z'] & vel_angular['N'], forca['N']),
            ctrl.Rule(angulo['Z'] & vel_angular['Z'], forca['Z']),
            ctrl.Rule(angulo['Z'] & vel_angular['P'], forca['P']),
            ctrl.Rule(angulo['P'] & vel_angular['N'], forca['Z']),
            ctrl.Rule(angulo['P'] & vel_angular['Z'], forca['P']),
            ctrl.Rule(angulo['P'] & vel_angular['P'], forca['FP'])
        ])
        
        regras.extend([
            ctrl.Rule(posicao_carro['N'] & vel_carro['N'], forca['FP']),
            ctrl.Rule(posicao_carro['N'] & vel_carro['Z'], forca['P']),
            ctrl.Rule(posicao_carro['N'] & vel_carro['P'], forca['Z']),
            ctrl.Rule(posicao_carro['Z'] & vel_carro['N'], forca['P']),
            ctrl.Rule(posicao_carro['Z'] & vel_carro['Z'], forca['Z']),
            ctrl.Rule(posicao_carro['Z'] & vel_carro['P'], forca['N']),
            ctrl.Rule(posicao_carro['P'] & vel_carro['N'], forca['Z']),
            ctrl.Rule(posicao_carro['P'] & vel_carro['Z'], forca['N']),
            ctrl.Rule(posicao_carro['P'] & vel_carro['P'], forca['FN'])
        ])
        
        return regras

def main():
    otimizador = ControladorGeneticoFuzzy()
    melhor_individuo = otimizador.otimizar()
    
    sistema_otimizado = otimizador.criar_sistema_fuzzy(melhor_individuo)
    
    simulador = SimuladorPendulo()
    simulador.sistema_fuzzy = sistema_otimizado
    simulador.simular()
    simulador.plotar_resultados()
    
    metricas = simulador.calcular_metricas()
    print("\nMétricas do Sistema Otimizado:")
    for nome, valor in metricas.items():
        print(f"{nome}: {valor:.4f}")

if __name__ == "__main__":
    main()