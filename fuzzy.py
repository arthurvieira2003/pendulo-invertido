import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

angulo = ctrl.Antecedent(np.linspace(-1, 1, 100), 'angulo')
vel_angular = ctrl.Antecedent(np.linspace(-1, 1, 100), 'vel_angular')
posicao_carro = ctrl.Antecedent(np.linspace(-1, 1, 100), 'posicao_carro')
vel_carro = ctrl.Antecedent(np.linspace(-1, 1, 100), 'vel_carro')
forca = ctrl.Consequent(np.linspace(-1, 1, 100), 'forca')

angulo['N'] = fuzz.trimf(angulo.universe, [-1, -1, 0])
angulo['Z'] = fuzz.trimf(angulo.universe, [-0.5, 0, 0.5])
angulo['P'] = fuzz.trimf(angulo.universe, [0, 1, 1])

vel_angular['N'] = fuzz.trimf(vel_angular.universe, [-1, -1, 0])
vel_angular['Z'] = fuzz.trimf(vel_angular.universe, [-0.5, 0, 0.5])
vel_angular['P'] = fuzz.trimf(vel_angular.universe, [0, 1, 1])

posicao_carro['N'] = fuzz.trimf(posicao_carro.universe, [-1, -1, 0])
posicao_carro['Z'] = fuzz.trimf(posicao_carro.universe, [-0.5, 0, 0.5])
posicao_carro['P'] = fuzz.trimf(posicao_carro.universe, [0, 1, 1])

vel_carro['N'] = fuzz.trimf(vel_carro.universe, [-1, -1, 0])
vel_carro['Z'] = fuzz.trimf(vel_carro.universe, [-0.5, 0, 0.5])
vel_carro['P'] = fuzz.trimf(vel_carro.universe, [0, 1, 1])

forca['FN'] = fuzz.trimf(forca.universe, [-1, -1, -0.6])
forca['N'] = fuzz.trimf(forca.universe, [-0.8, -0.4, 0])
forca['Z'] = fuzz.trimf(forca.universe, [-0.2, 0, 0.2])
forca['P'] = fuzz.trimf(forca.universe, [0, 0.4, 0.8])
forca['FP'] = fuzz.trimf(forca.universe, [0.6, 1, 1])

regra1 = ctrl.Rule(angulo['N'] & vel_angular['N'], forca['FN'])
regra2 = ctrl.Rule(angulo['N'] & vel_angular['Z'], forca['N'])
regra3 = ctrl.Rule(angulo['N'] & vel_angular['P'], forca['Z'])
regra4 = ctrl.Rule(angulo['Z'] & vel_angular['N'], forca['N'])
regra5 = ctrl.Rule(angulo['Z'] & vel_angular['Z'], forca['Z'])
regra6 = ctrl.Rule(angulo['Z'] & vel_angular['P'], forca['P'])
regra7 = ctrl.Rule(angulo['P'] & vel_angular['N'], forca['Z'])
regra8 = ctrl.Rule(angulo['P'] & vel_angular['Z'], forca['P'])
regra9 = ctrl.Rule(angulo['P'] & vel_angular['P'], forca['FP'])

regra10 = ctrl.Rule(posicao_carro['N'] & vel_carro['N'], forca['FP'])
regra11 = ctrl.Rule(posicao_carro['N'] & vel_carro['Z'], forca['P'])
regra12 = ctrl.Rule(posicao_carro['N'] & vel_carro['P'], forca['Z'])
regra13 = ctrl.Rule(posicao_carro['Z'] & vel_carro['N'], forca['P'])
regra14 = ctrl.Rule(posicao_carro['Z'] & vel_carro['Z'], forca['Z'])
regra15 = ctrl.Rule(posicao_carro['Z'] & vel_carro['P'], forca['N'])
regra16 = ctrl.Rule(posicao_carro['P'] & vel_carro['N'], forca['Z'])
regra17 = ctrl.Rule(posicao_carro['P'] & vel_carro['Z'], forca['N'])
regra18 = ctrl.Rule(posicao_carro['P'] & vel_carro['P'], forca['FN'])

sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5, regra6, regra7, regra8, regra9,
                                     regra10, regra11, regra12, regra13, regra14, regra15, regra16, regra17, regra18])
simulacao = ctrl.ControlSystemSimulation(sistema_controle)

def calcular_controle(ang, vel_ang, pos, vel):
    simulacao.input['angulo'] = ang
    simulacao.input['vel_angular'] = vel_ang
    simulacao.input['posicao_carro'] = pos
    simulacao.input['vel_carro'] = vel
    
    try:
        simulacao.compute()
        return simulacao.output['forca']
    except:
        return 0

class ParametrosFisicos:
    def __init__(self):
        self.g = 9.81
        self.M = 1.0
        self.m = 0.1
        self.l = 1.0
        self.b = 0.1

def equacoes_movimento(estado, t, F, params):
    x, dx, theta, dtheta = estado
    g, M, m, l, b = params.g, params.M, params.m, params.l, params.b
    
    sen = np.sin(theta)
    cos = np.cos(theta)
    
    den = M + m*np.sin(theta)**2
    
    ddx = (F + m*l*dtheta**2*sen - b*dx - m*g*sen*cos)/den
    ddtheta = (g*sen - cos*(F + m*l*dtheta**2*sen - b*dx)/(M + m))/l
    
    return [dx, ddx, dtheta, ddtheta]

class SimuladorPendulo:
    def __init__(self):
        self.params = ParametrosFisicos()
        self.dt = 0.01
        self.t_max = 10.0
        self.historico = {
            't': [], 'x': [], 'theta': [], 
            'dx': [], 'dtheta': [], 'forca': []
        }
        
    def simular(self):
        estado_inicial = [0, 0, 0.1, 0]
        t = np.arange(0, self.t_max, self.dt)
        
        estados = []
        forcas = []
        
        estado_atual = estado_inicial
        for i in range(len(t)-1):
            ang = estado_atual[2]
            vel_ang = estado_atual[3]
            pos = estado_atual[0]
            vel = estado_atual[1]
            
            F = self.calcular_controle(ang/np.pi, vel_ang, pos, vel)
            forcas.append(F)
            
            sol = odeint(equacoes_movimento, estado_atual, [t[i], t[i+1]], 
                        args=(F*10, self.params))
            estado_atual = sol[-1]
            estados.append(estado_atual)
            
            self.historico['t'].append(t[i])
            self.historico['x'].append(estado_atual[0])
            self.historico['theta'].append(estado_atual[2])
            self.historico['dx'].append(estado_atual[1])
            self.historico['dtheta'].append(estado_atual[3])
            self.historico['forca'].append(F)
    
    def calcular_metricas(self):
        theta = np.array(self.historico['theta'])
        x = np.array(self.historico['x'])
        f = np.array(self.historico['forca'])
        
        metricas = {
            'erro_max_theta': np.max(np.abs(theta)),
            'erro_max_posicao': np.max(np.abs(x)),
            'energia_controle': np.sum(f**2) * self.dt,
            'tempo_estabilizacao': 0
        }
        
        for i in range(len(theta)):
            if np.all(np.abs(theta[i:]) < 0.1):
                metricas['tempo_estabilizacao'] = self.historico['t'][i]
                break
        
        return metricas
    
    def plotar_resultados(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1.plot(self.historico['t'], self.historico['theta'])
        ax1.set_title('Ângulo do Pêndulo')
        ax1.set_xlabel('Tempo (s)')
        ax1.set_ylabel('Ângulo (rad)')
        
        ax2.plot(self.historico['t'], self.historico['x'])
        ax2.set_title('Posição do Carro')
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Posição (m)')
        
        ax3.plot(self.historico['t'], self.historico['forca'])
        ax3.set_title('Força Aplicada')
        ax3.set_xlabel('Tempo (s)')
        ax3.set_ylabel('Força (N)')
        
        x = np.array(self.historico['x'])
        theta = np.array(self.historico['theta'])
        pendulo_x = x + self.params.l * np.sin(theta)
        pendulo_y = self.params.l * np.cos(theta)
        
        ax4.plot(x, np.zeros_like(x), 'k-', label='Trilho')
        ax4.plot([x[-1], pendulo_x[-1]], [0, pendulo_y[-1]], 'r-', label='Pêndulo')
        ax4.set_title('Posição Final')
        ax4.set_aspect('equal')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()

    def calcular_controle(self, ang, vel_ang, pos, vel):
        if hasattr(self, 'controlador'):
            estado = [ang/np.pi, vel_ang, pos, vel]
            return self.controlador.calcular_controle(estado)
        else:
            return calcular_controle(ang/np.pi, vel_ang, pos, vel)

def testar_sistema_completo():
    simulador = SimuladorPendulo()
    
    print("Iniciando simulação...")
    tempo_inicio = time.time()
    
    simulador.simular()
    metricas = simulador.calcular_metricas()
    
    tempo_total = time.time() - tempo_inicio
    
    print("\nMétricas de Desempenho:")
    print(f"Erro máximo do ângulo: {metricas['erro_max_theta']:.4f} rad")
    print(f"Erro máximo da posição: {metricas['erro_max_posicao']:.4f} m")
    print(f"Energia total de controle: {metricas['energia_controle']:.4f}")
    print(f"Tempo de estabilização: {metricas['tempo_estabilizacao']:.2f} s")
    print(f"Tempo de simulação: {tempo_total:.2f} s")
    
    simulador.plotar_resultados()

if __name__ == "__main__":
    testar_sistema_completo()