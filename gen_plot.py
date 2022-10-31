import numpy as np
import matplotlib.pyplot as plt

data = {
    "e1":[0.061308, 0.2352, 0.196815],
    "e2":[0.927436, 0.442445, 0.664359],
    "e3":[1.042007, 0.327713, 2.791492]
}

print(data)

xAxis = np.array([2, 4, 16])

yAxis1 = np.array([data["e1"][0]] * 3) / np.array(data["e1"])
yAxis2 = np.array([data["e2"][0]] * 3) / np.array(data["e2"])
yAxis3 = np.array([data["e3"][0]] * 3) / np.array(data["e3"])

print(xAxis)
print(yAxis1)
print(yAxis2)
print(yAxis3)

fig, ax = plt.subplots(dpi=300)

ax.grid()
ax.plot(xAxis, yAxis1, label="Точность 3.0e-5")
ax.plot(xAxis, yAxis2, label="Точность 5.0e-6")
ax.plot(xAxis, yAxis3, label="Точность 1.5e-6")

ax.set(xlabel="Количество процессов", ylabel="Ускорение")

ax.legend()

fig.savefig("text/plot.png")
