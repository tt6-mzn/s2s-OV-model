import sys
import matplotlib.pyplot as plt
import random
import us2s_OV_CA
import numpy as np

M = 1  # 各密度についてシミュレーションする回数

def main():
	L = 100  # レーンの長さ
	n_0 = 3  # monitoring period
	v_0 = 4  # 車両の最高速度
	n_max = 1001  # シミュレーションするステップ数

	density = np.zeros(shape=((L+1) * M))
	flow    = np.zeros(shape=((L+1) * M))
 
	# 密度を変えて、複数の初期条件からflowを計算
	id = 0  # 各シミュレーションに割り振る番号
	for K in range(0, L + 1):
		for _ in range(M):
			sys.stdout.write("\rK = {:3d}, ({:4d}/{:4d})".format(K, _+1, M))
			sys.stdout.flush()
			# 車両の初期位置をランダムに生成
			x_init = sorted(random.sample([i for i in range(L)], K))
			model = us2s_OV_CA.us2s_OV_CA(L, K, n_0, v_0, x_init, n_max)
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