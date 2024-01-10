import numpy as np
import matplotlib.pyplot as plt


def plot_pattern(model, ni, nf, colored=0):
    x = np.array([[j for i in range(model.K)] for j in range(ni, nf+1)]) \
        .flatten()
    y = model.x[ni:nf+1, :model.K].flatten()
    plt.figure(figsize=(6.4, 6.4))
    plt.title(
        "L={}, K={}, n0={}, x0={}, v0={}, dt={}, dx={}"
            .format(model.L, model.K, model.n_0, model.x_0, model.v_0, model.dt, model.dx)
    )
    plt.xlabel("Time")
    plt.ylabel("Location of Vehicles")
    plt.scatter(x, y, s=1)
    x = np.array([j for j in range(ni, nf+1)])
    y = model.x[ni:nf+1, colored]
    plt.scatter(x, y, s=1, c="red")
    plt.show()

def plot_pattern_sort(model, ni, nf):
    x = np.array([[j for i in range(model.K)] for j in range(ni, nf+1)]) \
        .flatten()
    y = model.x[ni:nf+1, :model.K].flatten()
    plt.figure(figsize=(10, 10))
    plt.title(
        "L={}, K={}, n0={}, x0={}, v0={}, dt={}, dx={}"
            .format(model.L, model.K, model.n_0, model.x_0, model.v_0, model.dt, model.dx)
    )
    plt.xlabel("Time")
    plt.ylabel("Location of Vehicles")
    plt.scatter(x, y, s=1)
    plt.show()

def plot_flow(model, ni, nf):
	x = np.array([i for i in range(nf - ni + 1)])
	y = np.array([model.flow(0, i) for i in range(ni, nf + 1)])
	plt.figure(figsize=(6.4, 6.4))
	plt.title(
		"L={}, K={}, n0={}, x0={}, v0={}, dt={}, dx={}"
			.format(model.L, model.K, model.n_0, model.x_0, model.v_0, model.dt, model.dx)
	)
	plt.xlabel("Time")
	plt.ylabel("Flow")
	plt.plot(x, y)
	plt.show()


def plot_fundamental(density, flow):
	plt.figure(figsize=(10, 10))
	plt.xlim((0, 1.0))
	plt.ylim((0, 1.0))
	plt.scatter(density, flow, s=3)
	plt.show()


# 自明解の流量
def flow_stable(
		density: np.float64,
		x_0: np.float64,
		v_0: np.float64,
		dt: np.float64,
		dx: np.float64,
		xmin: np.float64,
	):
	if density == 0.0:
		return 0.0
	left = dx * (
		np.log(1 + np.exp((1.0/density - x_0)/dx))
		- np.log(1 + np.exp(-x_0/dx))
		- np.log(1 + np.exp((1.0/density - x_0 - v_0*dt)/dx))
		+ np.log(1 + np.exp(-(x_0 + v_0*dt)/dx))
	)
	right = 1.0/density - xmin
	return density * min(left, right)
