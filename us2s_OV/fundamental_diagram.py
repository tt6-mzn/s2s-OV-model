import sys
import matplotlib.pyplot as plt
import random
import us2s_OV_CA
import numpy as np

random.seed(0)

def main():
	L = 100  # レーンの長さ
	n_0_list = [3, 3, 2, 4]  # monitoring period
	v_0_list = [2, 4, 3, 3]  # 車両の最高速度
	n_max = 1001  # シミュレーションするステップ数

	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(2, 2, 1)
	ax2 = fig.add_subplot(2, 2, 2)
	ax3 = fig.add_subplot(2, 2, 3)
	ax4 = fig.add_subplot(2, 2, 4)
	axs = [ax1, ax2, ax3, ax4]
	for v_0, n_0, ax in zip(v_0_list, n_0_list, axs):
		ax.set_xlim((0, 1.0))
		ax.set_ylim((0, 1.0))
		ax.set_xticks([0.1 * i for i in range(11)])
		ax.set_yticks([0.1 * i for i in range(11)])
		ax.grid()
		ax.set_title("v_0={}, n_0={}".format(v_0, n_0))
		ax.set_xlabel("Density")
		ax.set_ylabel("Flow")

	M = [10 if i <= 60 else 5 for i in range(L+1)]  # 各密度についてシミュレーションする回数

	for v_0, n_0, axi in zip(v_0_list, n_0_list, axs):
		density = []
		flow    = []
	
		# 密度を変えて、複数の初期条件からflowを計算
		for K in range(0, L + 1):
			if K <= 50:
				# 均等に車両を配置した場合を試す
				x_init = sorted([int((i * L) / K) for i in range(K)])
				model = us2s_OV_CA.us2s_OV_CA(L, K, n_0, v_0, x_init, n_max)
				model.simulate()
				density.append(model.density())
				flow.append(model.flow(800, 1000))
			if  K < 40:
				# そこそこ均等に車両を配置した場合を試す
				for _ in range(500):
					sys.stdout.write("\rK = {:3d}, ({:4d}/{:4d})".format(K, _+1, 500))
					sys.stdout.flush()
					x_init = sorted([int((i * L) / K) + random.sample([0, 1], 1)[0] for i in range(K)])
					model = us2s_OV_CA.us2s_OV_CA(L, K, n_0, v_0, x_init, n_max)
					model.simulate()
					density.append(model.density())
					flow.append(model.flow(800, 1000))
				# あらめに均等に車両を配置した場合を試す
				for _ in range(500):
					sys.stdout.write("\rK = {:3d}, ({:4d}/{:4d})".format(K, _+1, 500))
					sys.stdout.flush()
					x_init = sorted([int((i * L) / K) + random.sample([0, 1, 2], 1)[0] for i in range(K)])
					model = us2s_OV_CA.us2s_OV_CA(L, K, n_0, v_0, x_init, n_max)
					model.simulate()
					density.append(model.density())
					flow.append(model.flow(800, 1000))
			# ランダムに試す
			for _ in range(M[K]):
				sys.stdout.write("\rK = {:3d}, ({:4d}/{:4d})".format(K, _+1, M[K]))
				sys.stdout.flush()
				# 車両の初期位置をランダムに生成
				x_init = sorted(random.sample([i for i in range(L)], K))
				model = us2s_OV_CA.us2s_OV_CA(L, K, n_0, v_0, x_init, n_max)
				model.simulate()
				density.append(model.density())
				flow.append(model.flow(800, 1000))
		
		with open("./test/v0={}, n0={}.txt".format(v_0, n_0), 'w', encoding='utf-8') as f:
			for de, fl in zip(density, flow):
				f.write("{}, {}\n".format(de, fl))
		
		axi.scatter(density, flow, s=3)
		print()
	
	plt.savefig("./test/fundamental diagram.png")

if __name__ == "__main__":
    main()