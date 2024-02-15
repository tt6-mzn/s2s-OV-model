import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt

"""
us2s_OVモデルを表現するクラス
L     : レーンの長さ
K     : 車両の数
n_0   : monitoring period(この時間の間の車間距離を使って次の状態を決定する)
v_0   : 車両の最高移動距離
x_init: 車両の初期位置
n_max : シミュレーションするステップ数
x_0   : 最短車間距離
dt    : 時間間隔
"""
class us2s_OV_CA:
    def __init__(
        self,
        L     : int,        # レーンの長さ
        K     : int,        # 車両の台数
        n_0   : int,        # monitoring period
        v_0   : int,        # 車両の最高速度
        x_init: List[int],  # 車両の初期位置(!! ソート済みであること必須 !!)
        n_max : int,        # シミュレーションするステップ数
        x_0   : int = 1,    # 最短車間距離
        dt    : int = 1     # 時間差分
    ) -> None:
        self.L     = L  # レーンの長さ
        self.K_max = L  # 車両の最大数
        self.K     = K  # 車両の数
        self.n_0   = n_0  # monitoring period
        self.x_0   = x_0  # 最短車間距離
        self.v_0   = v_0  # 車両の最高速度
        self.dt    = dt   # 時間間隔
        self.n_max = n_max  # シミュレーションするステップ数
        self.n     = n_0  # 現在のステップ数
        # 各時刻における車両の位置 x[n][k] := x_k^n
        self.x     = np.full(
            shape=(self.n_max + 1, self.K_max),
            fill_value=-1,
            dtype=np.int64
        )

        # 車両の初期位置
        if self.K == 0: return  # 車両が0台のとき
        self.x[:n_0+1, :self.K] = np.array(x_init)[None, :]
        self.delta_x = np.full(
            shape=(self.n_max + 1, self.K_max),
            fill_value=-1,
            dtype=np.int64
        )
        # 前方の車両との車間距離
        if self.K == 1:
            self.delta_x[:self.n_0+1, 0] = self.L
        else:
            self.delta_x[:self.n_0+1,  self.K-1] = (self.x[:self.n_0+1, 0       ] - self.x[:self.n_0+1,  self.K-1] + self.L) % self.L
            self.delta_x[:self.n_0+1, :self.K-1] = (self.x[:self.n_0+1, 1:self.K] - self.x[:self.n_0+1, :self.K-1] + self.L) % self.L
    
    # ステップを一つ進める
    def _next(self) -> None:
        # 各車両における\delta_eff x_kをを計算する
        delta_eff = np.min(self.delta_x[self.n-self.n_0:self.n+1, :self.K], axis=0) - self.x_0
        self.x[self.n + 1, :self.K] = (
            self.x[self.n, :self.K]
            + np.where(delta_eff <= self.v_0 * self.dt, delta_eff, self.v_0 * self.dt)
            # + np.where(delta_eff < 0, 0, delta_eff)
            # - np.where(delta_eff - self.v_0 < 0, 0, delta_eff - self.v_0 * self.dt)
        ) % self.L
        # 車間距離の更新
        if self.K == 1:
            self.delta_x[self.n+1, 0] = self.L
        else:
            self.delta_x[self.n+1,  self.K-1] = (self.x[self.n+1, 0       ] - self.x[self.n+1,  self.K-1] + self.L) % self.L
            self.delta_x[self.n+1, :self.K-1] = (self.x[self.n+1, 1:self.K] - self.x[self.n+1, :self.K-1] + self.L) % self.L
        self.n += 1
    
    # nmaxまでシミュレーションを行う
    def simulate(self) -> None:
        if self.K == 0: return
        while self.n < self.n_max:
            self._next()
    
    def __repr__(self) -> str:
        return "\n".join(["{:4d}: ".format(i) + "".join("X" if j in self.x[i] else "-" for j in range(self.L))
                            for i in range(self.n + 1)])
    
    # n_1ステップからn_2ステップまでの流量を計算
    def flow(self, n_1: int, n_2: int) -> float:
        if self.K == 0: return 0.0
        return np.sum(
            ((self.x[n_1+1:n_2+2, :self.K] - self.x[n_1:n_2+1, :self.K] + self.L) % self.L)
                / (self.dt * (n_2 - n_1 + 1) * self.L)
        )

    # 密度
    def density(self):
        return self.K / self.L

def plot_pattern(model, ni, nf):
    # niからnfまでの車両の位置をプロット
    x = np.array([[j for i in range(1, model.K)] for j in range(ni, nf+1)]) \
        .flatten()
    y = model.x[ni:nf+1, 1:model.K].flatten()
    plt.figure(figsize=(10, 10))
    plt.title(
        "L={}, K={}, n0={}, x0={}, v0={}, dt={}"
            .format(model.L, model.K, model.n_0, model.x_0, model.v_0, model.dt)
    )
    plt.xlabel("Time")
    plt.ylabel("Location of Vehicles")
    plt.grid()
    plt.xlim((ni-1, nf+1))
    plt.ylim((-1, model.L+1))
    plt.scatter(x, y, s=3)
    # 一台目の車両の位置をプロット
    x = np.array([j for j in range(ni, nf+1)])
    y = model.x[ni:nf+1, 0]
    plt.scatter(x, y, s=3, c="red")
    plt.show()


def plot_fundamental(ax, model, density, flow):
    ax.set_xlabel("Density")
    ax.set_ylabel("Flow")
    ax.set_xlim((0, 1.0))
    ax.set_ylim((0, 1.0))
    ax.grid()
    ax.set_title(
        "n0={}, v0={}"
        .format(model.n_0, model.v_0)
    )
    ax.scatter(density, flow, s=5)


if __name__ == "__main__":
    # us2s_OVモデルのインスタンス化
    model = us2s_OV_CA(
        L=100,
        K      = 1,                      # 車両の数
        n_0    = 2,                       # monitoring period
        v_0=2,
        x_init = sorted(random.sample([i for i in range(100)], 1)),  # 車両の初期位置
        n_max=100,
    )

	# シミュレーション
    model.simulate()

    print(model)
    print("{:.10}".format(model.flow(0, 99)))
