from typing import List
import numpy as np

"""
us2s_OVモデルを表現するクラス
L    : レーンの長さ
K    : 車両の数
n_0  : monitoring period(この時間の間の車間距離を使って次の状態を決定する)
x_0  : 最短車間距離
v_0  : 車両の最高速度
dt   : 時間間隔
x    : 車両の初期位置
N_MAX: シミュレーションするステップ数
"""

L     = 100   # レーンの長さ
K_MAX = L     # 車両の最大数
N_MAX = 1001  # シミュレーションするステップ数
dt    = 1     # 時間間隔

class us2s_OV:
    def __init__(self, K: int, n_0: int, x_0: int, v_0: int, x_init: List[int]) -> None:
        self.K   = K  # 車両の数
        self.n_0 = n_0  # monitoring period
        self.x_0 = x_0  # 最短車間距離
        self.v_0 = v_0  # 車両の最高速度
        self.n   = n_0  # 現在のステップ数
        self.x   = np.full(
            shape=(N_MAX + 1, K_MAX),
            fill_value=-1,
            dtype=np.int64
        )  # 各時刻における車両の位置

        # 車両の初期位置
        self.x[:n_0+1, :self.K] = np.array(x_init)[None, :]
    
	# 時刻nにおける車両kと車両k+1の車間距離を計算する関数
    def _delta_x(self, k: int, n: int) -> int:
        ret: int = self.x[n, (k + 1) % self.K] - self.x[n, k]
        if ret >= 0:
            return ret 
        return L + ret 
    
    # ステップを一つ進める
    def _next(self) -> None:
        for k in range(self.K):
            delta_eff: int = min([self._delta_x(k, self.n - n_) for n_ in range(0, self.n_0 + 1)]) - self.x_0
            self.x[self.n + 1, k] = (
                self.x[self.n, k]
                + max(0, delta_eff)
                - max(0, delta_eff - self.v_0 * dt)
            ) % L
        self.n += 1
    
    # nmaxまでシミュレーションを行う
    def simulate(self) -> None:
        while self.n < N_MAX:
            self._next()
    
    def __repr__(self) -> str:
        return "\n".join(["{:4d}: ".format(i) + "".join("X" if j in self.x[i] else "-" for j in range(L))
                            for i in range(self.n + 1)])
    
    # n_1ステップからn_2ステップまでの流量を計算
    def flow(self, n_1: int, n_2: int):
        tmp = np.sum((self.x[n_1+1:n_2+2, :self.K] - self.x[n_1:n_2+1, :self.K] + L) % L)
        tmp /= dt
        tmp /= (n_2 - n_1 + 1) * L
        return tmp

    def density(self):
        return self.K / L


if __name__ == "__main__":
    # us2s_OVモデルのインスタンス化
    model = us2s_OV(
        K      = 30,                      # 車両の数
        n_0    = 2,                       # monitoring period
        x_0    = 1,                       # 最短車間距離
        v_0    = 1,                       # 車両の最高速度
        x_init = [i for i in range(30)],  # 車両の初期位置
    )

	# シミュレーション
    model.simulate()

    print(model)
    print("{:.10}".format(model.flow(0, 1000)))
