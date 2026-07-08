import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
import oapackage
from kneebow.rotor import Rotor
from kneed import KneeLocator

k = 5.95
v = 0.0014
epson = 0.0667 # The probability of de novo resistance development
gamma = 0.02383 # The rate of normal recovery
gamma_a = 0.086 # The rate of recovery with antibiotics (only works in sensitive strains)
mu = 0.0035 # Rate of mortality due to disease
alpha = 0.5 # Fitness cost
average_expenditure = 0.0071 # Cost of maintenance animals per day per animal
antibiotic_cost = 0.0019 # cost of antibiotic per day per chicken
W = 2.404 # Average weight of a healthy chicken
P = 0.83 #Price per kg of final chicken
WlossR = 0.02
WlossI = 0.027
p = 0.02
day_max = 35
I0_relative = 0.058
density_min = 5
density_max = 25
A = 1996.4 #Average size of farm in square meters

densities = np.linspace(density_min, density_max, 1000)
t = np.linspace(0,42,1000)

xplot = np.array([])
yplot = np.array([])

y_revenue = np.array([])
yrev_norm = np.array([])
yamr_norm = np.array([])
y_final = np.array([])
for density in densities:
    # Initial State Values
    S0 = density * A -  (density * A * I0_relative)
    IS0 = density * A * I0_relative
    IR0 = 0
    R0 = 0
    D0 = 0
    N0 = S0 + IS0 + IR0 + R0 + D0


    def SIR_MODEL(y,t, k, v, gamma,gamma_a, mu, alpha, epson):
        S, IS, IR, R, D,N = y
        lambda_S = k * v * IS / A
        lambda_R = k * v * IR * (1 - alpha) / A
        treatment = (D / N0 >= p and day_max >= t)
        dS = -(lambda_S + lambda_R) * S

        if treatment:
            dIS = lambda_S * S - (gamma_a + mu + epson) * IS
            dIR = epson * IS + lambda_R * S - (gamma + mu) * IR
            dR = gamma * IR + gamma_a * IS
        else:
            dIS = lambda_S * S - (gamma + mu) * IS
            dIR = lambda_R * S - (gamma + mu) * IR
            dR = gamma * (IS + IR)
        dD = mu * (IS + IR)
        dN = -mu * (IS + IR)
        return [dS, dIS, dIR, dR, dD, dN]

    result = odeint(SIR_MODEL, y0=[S0, IS0, IR0, R0, D0, N0], t= t, args=(k, v, gamma, gamma_a, mu, alpha, epson))

    S, IS, IR, R, D, N = result.T
    antibiotic_active = (D/N0 > p) & (day_max >= t)
    gross_revenue = N[-1] * P * W
    weight_loss = WlossI * (IS[-1] + IR[-1]) * W + WlossR * W * R[-1]
    maintenance = trapezoid(N, t) * average_expenditure
    antibiotic = trapezoid(N * antibiotic_active, t) * antibiotic_cost

    Revenue = gross_revenue - weight_loss * P - maintenance - antibiotic
    xplot = np.append(xplot, density)
    FR = (trapezoid(IR, t)/ trapezoid(IR + IS, t)) * 100
    yplot = np.append(yplot, FR)
    y_revenue = np.append(y_revenue, Revenue)


yamr_norm = (yplot - yplot.min()) / (yplot.max() - yplot.min())
yrev_norm = (y_revenue - y_revenue.min()) / (y_revenue.max() - y_revenue.min())
y_final = yamr_norm - yrev_norm
index_of = np.where(y_final == y_final.min())
datapoints = np.array([yplot, y_revenue])
pareto_data = np.array([-yplot, y_revenue])
pareto = oapackage.ParetoDoubleLong()

for ii in range(0, pareto_data.shape[1]):
    w = oapackage.doubleVector((pareto_data[0, ii], pareto_data[1, ii]))
    pareto.addvalue(w, ii)

pareto.show(verbose=1)
lst = pareto.allindices()
optimal_datapoints = datapoints[:, lst]

sorted_idx = np.argsort(optimal_datapoints[0, :])
pareto_x = optimal_datapoints[0, sorted_idx]
pareto_y = optimal_datapoints[1, sorted_idx]
rotor = Rotor()
rotor.fit_rotate(np.column_stack([pareto_x, pareto_y]))
kneeb = KneeLocator(optimal_datapoints[0, :], optimal_datapoints[1, :])
print('KneeLocator: ', kneeb)
knee_idx = rotor.get_elbow_index()
print(f"Knee point — FR: {pareto_x[knee_idx]:.2f}%, Revenue: {pareto_y[knee_idx]:.2f}€")
knee_density_idx = np.where(y_revenue == pareto_y[knee_idx])[0]
print("Knee point density: ", xplot[knee_density_idx])
plt.subplot(2, 2, 1)
plt.plot(xplot, yplot)
plt.title('AMR')

plt.xlabel('Density')
plt.ylabel('Resistant strains (%)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(xplot, y_revenue)
plt.title('Revenue')
plt.xlabel('Density')
plt.ylabel('Revenue (€)')
plt.grid(True)

plt.subplot(2,2,3)

plt.plot(datapoints[0, :], datapoints[1, :], ".b", label="Non Pareto")
plt.plot(optimal_datapoints[0, :], optimal_datapoints[1, :], ".r", label="Pareto")
plt.plot(datapoints[0, index_of], datapoints[1, index_of], ".y", label="WSM")
plt.plot(pareto_x[knee_idx], pareto_y[knee_idx], ".g", label="Knee")
plt.title('Pareto plot')
plt.legend()
plt.grid(True)

plt.subplot(2,2,4)
plt.plot(t, S, label="S")
plt.plot(t, IS, label="IS")
plt.plot(t, IR, label="IR")
plt.plot(t, R, label="R")
plt.plot(t, D, label="D")
plt.grid(True)
plt.show()

