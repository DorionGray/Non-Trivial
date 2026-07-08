import numpy as np
from kneed import KneeLocator
from scipy.integrate import odeint, trapezoid
import matplotlib.pyplot as plt
import oapackage

k = 0.000501
v = 0.0014
epson = 0.0667  # Probability of de novo resistance development
gamma = 0.0001  # Rate of natural recovery
gamma_a = 0.086  # Rate of recovery with antibiotics (sensitive only)
mu = 0.0035  # Mortality rate due to disease
alpha = 0.5  # Fitness cost of resistance
average_expenditure = 0.0071  # Maintenance cost per animal per day (€)
antibiotic_cost = 0.0019  # Antibiotic cost per animal per day (€)
W = 2.404  # Average weight of a healthy chicken (kg)
P = 0.83  # Price per kg of final chicken (€)
WlossR = 0.02  # Weight loss fraction – resistant infected
WlossI = 0.112  # Weight loss fraction – sensitive infected
p = 0.02  # Cumulative mortality threshold for treatment
day_max = 35  # Latest day treatment can be triggered
I0_relative = 0.058  # Initial infection prevalence
A = 1996.4  # Farm area

densities = np.linspace(5, 25, 1000)
t = np.linspace(1, 42, 1000)

FR_all = []
Rev_all = []

for density in densities:
    S0 = density * A - (density * A * I0_relative)
    IS0 = density * A * I0_relative
    IR0 = R0 = D0 = 0
    N0 = S0 + IS0


    def SIR_MODEL(y, t, k, v, gamma, gamma_a, mu, alpha, epson):
        S, IS, IR, R, D, N = y
        beta = k * density * v
        beta_R = k * v * density * (1 - alpha)
        lam_S = beta * IS
        lam_R = beta_R * IR
        treat = (D / N0 >= p) and (t <= day_max)
        dS = -(lam_S + lam_R) * S
        if treat:
            dIS = lam_S * S - (gamma_a + mu + epson) * IS
            dIR = epson * IS + lam_R * S - (gamma + mu) * IR
            dR = gamma * IR + gamma_a * IS
        else:
            dIS = lam_S * S - (gamma + mu) * IS
            dIR = lam_R * S - (gamma + mu) * IR
            dR = gamma * (IS + IR)
        dD = mu * (IS + IR)
        dN = -dD
        return [dS, dIS, dIR, dR, dD, dN]


    res = odeint(SIR_MODEL, y0=[S0, IS0, IR0, R0, D0, N0], t=t, args=(k, v, gamma, gamma_a, mu, alpha, epson))
    S, IS, IR, R, D, N = res.T

    antibiotic_active = (D / N0 > p) & (t <= day_max)
    gross_revenue = N[-1] * P * W
    weight_loss = (WlossI * (IS[-1] + IR[-1]) + WlossR * R[-1]) * W
    maintenance = trapezoid(N, t) * average_expenditure
    antibiotic = trapezoid(N * antibiotic_active, t) * antibiotic_cost
    Revenue = gross_revenue - weight_loss * P - maintenance - antibiotic

    denom = IR[-1] + IS[-1]
    FR = (trapezoid(IR, t)/ trapezoid(IR + IS, t)) * 100

    FR_all.append(FR)
    Rev_all.append(Revenue)

densities = np.array(densities)
FR_all = np.array(FR_all)
Rev_all = np.array(Rev_all)

# Keep last ODE solution for the dynamics plot
S_last, IS_last, IR_last, R_last, D_last, N_last = res.T

datapoints = np.array([FR_all, Rev_all])
pareto_data = np.array([-FR_all, Rev_all])
pareto = oapackage.ParetoDoubleLong()

for ii in range(0, pareto_data.shape[1]):
    w = oapackage.doubleVector((pareto_data[0, ii], pareto_data[1, ii]))
    pareto.addvalue(w, ii)

pareto.show(verbose=1)
lst = pareto.allindices()
optimal_datapoints = datapoints[:, lst]

kl = KneeLocator(datapoints[0, :], datapoints[1, :], curve="convex", direction="increasing")
print(kl.knee, kl.knee_y)
plt.subplot(2,2, 1)
plt.plot(densities, FR_all, color='#c0392b', linewidth=1.8)
plt.xlabel('Stocking Density (animals/m²)')
plt.ylabel('Resistant strains (%)')
plt.grid(True, alpha=0.3)


plt.subplot(2,2,2)
plt.plot(densities, Rev_all)
plt.xlabel('Stocking Density (animals/m²)')
plt.ylabel('Revenue (€ thousands)')
plt.grid(True, alpha=0.3)

plt.subplot(2,2, 3)

plt.plot(datapoints[0, :], datapoints[1, :], ".b", label="Non Pareto")
plt.plot(optimal_datapoints[0, :], optimal_datapoints[1, :], ".r", label="Pareto")
plt.plot(kl.knee, kl.knee_y, ".y", label="Knee")
plt.legend()



plt.subplot(2, 2, 4)
plt.plot(t, S_last, label='Susceptible')
plt.plot(t, IS_last, label='Infected (sensitive)')
plt.plot(t, IR_last, label='Infected (resistant)')
plt.plot(t, R_last, label='Recovered')
plt.plot(t, D_last, label='Dead')
plt.xlabel('Time (days)')
plt.ylabel('Number of birds')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.show()