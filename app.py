import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
 
class FuzzyController:
    def __init__(self):
        # Universos de discurso
        self.angle_universe = np.linspace(-np.pi/2, np.pi/2, 100)
        self.angular_vel_universe = np.linspace(-2, 2, 100)
        self.position_universe = np.linspace(-2, 2, 100)
        self.velocity_universe = np.linspace(-2, 2, 100)
        
    def triangular_mf(self, x, params):
        """Função de pertinência triangular"""
        a, b, c = params
        return np.maximum(np.minimum((x - a)/(b - a), (c - x)/(c - b)), 0)
    
    def membership_functions(self, x, universe, type="angle"):
        """Define as funções de pertinência para cada variável"""
        if type == "angle":
            N = self.triangular_mf(universe, [-np.pi/2, -np.pi/4, 0])
            Z = self.triangular_mf(universe, [-np.pi/4, 0, np.pi/4])
            P = self.triangular_mf(universe, [0, np.pi/4, np.pi/2])
        else:  # velocity and position
            N = self.triangular_mf(universe, [-2, -1, 0])
            Z = self.triangular_mf(universe, [-1, 0, 1])
            P = self.triangular_mf(universe, [0, 1, 2])
        
        return {"N": N, "Z": Z, "P": P}
    
    def fuzzify(self, value, universe, type="angle"):
        """Processo de fuzzificação"""
        mfs = self.membership_functions(value, universe, type)
        return {k: float(v[np.abs(universe - value).argmin()])
                for k, v in mfs.items()}
    
    def apply_rules(self, angle_mf, angular_vel_mf, pos_mf, vel_mf):
        """Aplica as regras fuzzy e retorna a força resultante"""
        # Regras para o pêndulo
        pendulum_force = 0
        weights = []
        outputs = []
        
        # Exemplo de algumas regras (simplificado)
        # IF pêndulo está inclinado à esquerda (N) AND movendo para esquerda (N)
        # THEN empurre fortemente para esquerda
        strength = min(angle_mf["N"], angular_vel_mf["N"])
        weights.append(strength)
        outputs.append(-2)  # Força forte para esquerda
        
        # IF pêndulo vertical (Z) AND parado (Z) THEN não empurre
        strength = min(angle_mf["Z"], angular_vel_mf["Z"])
        weights.append(strength)
        outputs.append(0)  # Sem força
        
        # Defuzzificação usando média ponderada
        if sum(weights) != 0:
            pendulum_force = sum(w * o for w, o in zip(weights, outputs)) / sum(weights)
            
        # Similar para o carro...
        car_force = 0
        weights = []
        outputs = []
        
        # Defuzzificação final combinando forças do pêndulo e carro
        final_force = 0.7 * pendulum_force + 0.3 * car_force
        return final_force
 
class PendulumSystem:
    def __init__(self):
        self.M = 1.0  # massa do carro
        self.m = 0.1  # massa do pêndulo
        self.l = 1.0  # comprimento do pêndulo
        self.g = 9.81  # aceleração da gravidade
        self.controller = FuzzyController()
        
    def dynamics(self, state, t, F):
        """Equações dinâmicas do sistema"""
        x, dx, theta, dtheta = state
        
        # Equações do movimento
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Cálculos intermediários
        den = self.M + self.m * sin_theta**2
        
        # Acelerações
        ddx = (F + self.m * sin_theta *
               (self.l * dtheta**2 + self.g * cos_theta)) / den
        
        ddtheta = (-F * cos_theta - self.m * self.l * dtheta**2 *
                   sin_theta * cos_theta -
                   (self.M + self.m) * self.g * sin_theta) / (self.l * den)
        
        return [dx, ddx, dtheta, ddtheta]
    
    def simulate(self, t_span, initial_state):
        """Simula o sistema no intervalo de tempo especificado"""
        t = np.linspace(0, t_span, 1000)
        states = []
        current_state = initial_state
        
        dt = t[1] - t[0]
        for time in t:
            # Obter ação de controle do controlador fuzzy
            angle = current_state[2]
            angular_vel = current_state[3]
            position = current_state[0]
            velocity = current_state[1]
            
            # Fuzzificar entradas
            angle_mf = self.controller.fuzzify(angle,
                                             self.controller.angle_universe,
                                             "angle")
            angular_vel_mf = self.controller.fuzzify(angular_vel,
                                                   self.controller.angular_vel_universe,
                                                   "velocity")
            pos_mf = self.controller.fuzzify(position,
                                           self.controller.position_universe,
                                           "position")
            vel_mf = self.controller.fuzzify(velocity,
                                           self.controller.velocity_universe,
                                           "velocity")
            
            # Calcular força de controle
            F = self.controller.apply_rules(angle_mf, angular_vel_mf,
                                          pos_mf, vel_mf)
            
            # Integrar um passo
            sol = odeint(self.dynamics, current_state, [0, dt], args=(F,))
            current_state = sol[-1]
            states.append(current_state)
            
        return np.array(states), t
 
    def plot_results(self, states, t):
        """Plota os resultados da simulação"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot ângulo do pêndulo
        ax1.plot(t, states[:, 2])
        ax1.set_xlabel('Tempo (s)')
        ax1.set_ylabel('Ângulo (rad)')
        ax1.set_title('Ângulo do Pêndulo')
        ax1.grid(True)
        
        # Plot posição do carro
        ax2.plot(t, states[:, 0])
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Posição (m)')
        ax2.set_title('Posição do Carro')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
 
# Exemplo de uso
pendulum = PendulumSystem()
initial_state = [0, 0, np.pi/6, 0]  # [x, dx, theta, dtheta]
states, t = pendulum.simulate(10, initial_state)
pendulum.plot_results(states, t)