import sys
import matplotlib.pyplot as plt
import random
import us2s_OV
import numpy as np

def main():
	n_0   = 3    # monitoring period
	x_0   = 1    # 最短車間距離
	v_0   = 4    # 車両の最高速度

	density = np.zeros(shape=(101 * 100))
	flow = np.zeros(shape=(101 * 100))
 
	# 密度を変えて、複数の初期条件からflowを計算
	id = 0  # 各シミュレーションに割り振る番号
	for K in range(0, us2s_OV.K_MAX + 1):
		for _ in range(1): # 各密度について100回ずつ計算
			sys.stdout.write("\rK = {}, ({}/{})".format(K, _, 100))
			sys.stdout.flush()
			# 車両の初期位置をランダムに生成
			x_init = sorted(random.sample([i for i in range(us2s_OV.L)], K))
			model = us2s_OV.us2s_OV(K, n_0, x_0, v_0, x_init)
			model.simulate()
			density[id] = model.density()
			flow[id] = model.flow(800, 1000)
			id += 1
	
	plt.figure(figsize=(6.4, 6.4))
	plt.grid()
	plt.scatter(
		x=density,
		y=flow,
		s=2,
		marker='o',
	)
	plt.xlim(0.0, 1.0)
	plt.ylim(0.0, 1.0)
	plt.savefig("./img/test.png")

if __name__ == "__main__":
    main()