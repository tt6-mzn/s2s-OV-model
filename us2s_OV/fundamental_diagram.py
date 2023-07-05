import sys
import matplotlib.pyplot as plt
import random
import us2s_OV

def main():
	L     = 100  # レーンの長さ
	n_0   = 3  # monitoring period
	x_0   = 1  # 最短車間距離
	v_0   = 4  # 車両の最高速度
	dt    = 1  # 時間差分

	density = []
	flow = []
 
	# 密度を変えて、複数の初期条件からflowを計算
	for K in range(0, 101):
		sys.stdout.write("\r({}/{})".format(K, 100))
		sys.stdout.flush()
		for _ in range(1): # 各密度について100回ずつ計算
			# 車両の初期位置をランダムに生成
			x_init = sorted(random.sample([i for i in range(L)], K))
			model = us2s_OV.us2s_OV(L, K, n_0, x_0, v_0, dt, x_init)
			model.simulate(1100)
			density.append(model.density())
			flow.append(model.flow(800, 1000))
	
	# for d, f in zip(density, flow):
	# 	print(d, f)
	
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