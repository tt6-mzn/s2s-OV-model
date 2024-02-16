import random
from typing import List
import numpy as np
import json
import matplotlib.pyplot as plt

"""
ds2s_OVモデルを表現するクラス
L     : レーンの長さ
K     : 車両の数
n_0   : monitoring period(この時間の間の車間距離を使って次の状態を決定する)
x_0   : 最短車間距離
v_0   : 車両の最高移動距離
dt    : 時間間隔
x_init: 車両の初期位置
n_max : シミュレーションするステップ数
"""


class ds2s_OV:
    def __init__(
        self,
        L: np.float64,  # レーンの長さ
        K: np.int32,  # 車両の台数
        n_0: np.int32,  # monitoring period
        x_0: np.int32,  # 車間距離
        v_0: np.float64,  # 車両の最高速度
        dt: np.float64,  # 時間差分
        dx: np.float64,  # 空間差分
        x_init: np.ndarray[np.float64],  # 車両の初期位置
        n_max: np.int32
    ) -> None:
        # 各種メンバ変数の初期化
        self.L = L  # レーンの長さ
        self.K = K  # 車両の数
        self.n_0 = n_0  # monitoring period
        self.x_0 = x_0  # 最短車間距離
        self.v_0 = v_0  # 車両の最高速度
        self.dt = dt  # 時間間隔
        self.dx = dx  # 空間差分
        self.n_max = n_max  # シミュレーションするステップ数
        self.n = n_0  # 現在のステップ数
        self.x = np.full(
            shape=(self.n_max + 1, self.K),
            fill_value=-1,
            dtype=np.float64
        )  # 各時刻における車両の位置 x[n][k] := x_k^n
        self.delta_x = np.full(
            shape=(self.n_max + 1, self.K),
            fill_value=-1,
            dtype=np.float64
        )  # 前方のｓ車両との車間距離 delta_x[n][k] := x_{k+1}^n - x_k^n
        self.v = np.full(
            shape=(self.n_max + 1, self.K),
            fill_value=-1,
            dtype=np.float64
        )  # 各時刻における車両の速度 v[n][k] := v_k^n

        # 車両の初期位置
        x_init = np.sort(x_init)  # 一応ソートしておく
        if self.K == 0:
            return  # 車両が0台のとき
        self.x[:n_0+1] = x_init[None, :]
        self.v[:n_0+1] = np.zeros(shape=(n_0+1, self.K))

        # 前方の車両との車間距離
        for i in range(n_0+1):
            self._update_delta_x(i)
    
    @classmethod
    def from_json(cls, json: dict, n_max: int):
        return cls(
            L=json["L"],
            K=json["K"],
            n_0=json["n_0"],
            x_0=json["x_0"],
            v_0=json["v_0"],
            dt=json["dt"],
            dx=json["dx"],
            x_init=np.array(json["x_init"]),
            n_max=n_max
        )

    # self.xをもとにself.delta_xを更新する
    def _update_delta_x(self, n) -> None:
        if self.K == 1:
            self.delta_x[n, 0] = self.L
            return
        # 先頭にいる車両のインデックスを取得
        head = np.argmax(self.x[n])
        self.delta_x[n, self.K-1] = self.x[n, 0] - self.x[n, self.K-1]
        self.delta_x[n, :self.K-1] = self.x[n, 1:self.K] - self.x[n, :self.K-1]
        # 先頭の車両に対して周期境界条件を適用
        self.delta_x[n, head] += self.L

    # 移動距離の計算
    def _delta(self) -> np.ndarray[np.float64]:
        delta_eff = self.delta_x[self.n-self.n_0:self.n+1] - self.x_0
        e1 = np.sum(
            np.exp(
                -delta_eff / self.dx
            ) / (self.n_0 + 1),
            axis=0
        )
        e2 = np.sum(
            np.exp(
                -(delta_eff - self.v_0 * self.dt) / self.dx
            ) / (self.n_0 + 1),
            axis=0
        )
        return self.dx * (
            np.log(1.0 + 1.0 / e1)
            - np.log(1.0 + np.exp(-self.x_0 / self.dx))
            - np.log(1.0 + 1.0 / e2)
            + np.log(1.0 + np.exp(-(self.x_0 + self.v_0 * self.dt) / self.dx))
        )

    # 周期境界条件の適用
    def _periodic(self) -> None:
        self.x[self.n+1] = \
            np.where(
                self.x[self.n+1] < self.L,
                self.x[self.n+1],
                self.x[self.n+1] - self.L
        )

    # ステップを一つ進める
    def _next(self) -> None:
        self.v[self.n+1] = self._delta() / self.dt
        self.x[self.n+1] = self.x[self.n] + self.v[self.n+1] * self.dt
        self._periodic()
        self._update_delta_x(self.n+1)
        self.n += 1

    # nmaxまでシミュレーションを行う
    def simulate(self) -> None:
        if self.K == 0:
            return
        while self.n < self.n_max:
            self._next()

    # niステップからnfステップまでの流量を計算
    def flow(self, ni: int, nf: int) -> float:
        if self.K == 0:
            return 0.0
        return np.sum(
            self.v[ni:nf+1] / ((nf - ni + 1) * self.L)
        )

    # 密度
    def density(self):
        return np.float64(self.K) / self.L

    # jsonとして出力
    def get_json(self):
        return {
            "L": self.L,
            "K": self.K,
            "n_0": self.n_0,
            "x_0": self.x_0,
            "v_0": self.v_0,
            "dt": self.dt,
            "dx": self.dx,
            "x_init": self.x[0].tolist(),
            "density": self.density(),
        }
    
    # モデル名
    def model_type(self):
        return "ds2s_OV"


def plot_pattern(model, ni, nf, ax=None):
    # niからnfまでの車両の位置をプロット
    x = np.array([[j for i in range(1, model.K)] for j in range(ni, nf+1)]) \
        .flatten()
    y = model.x[ni:nf+1, 1:model.K].flatten()
    if ax == None:
        plt.figure(figsize=(10, 10))
        plt.title(
            "L={}, K={}, n0={}, x0={:.2}, v0={:.2}, dt={:.2}, dx={:.2}"
                .format(model.L, model.K, model.n_0, model.x_0, model.v_0, model.dt, model.dx)
        )
        plt.xlabel("Time")
        plt.ylabel("Location of Vehicles")
        plt.grid()
        plt.xlim((ni-1, nf+1))
        plt.ylim((-model.x_0, model.L+model.x_0))
        plt.scatter(x, y, s=3)
    else:
        ax.set_title(
            "model={}, L={}, K={}, n0={}, x0={:.2}, v0={:.2}, dt={:.2}, dx={:.2}"
                .format(model.model_type(), model.L, model.K, model.n_0, model.x_0, model.v_0, model.dt, model.dx)
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Location of Vehicles")
        ax.grid()
        ax.set_xlim((ni-1, nf+1))
        ax.set_ylim((-model.x_0, model.L+model.x_0))
        ax.scatter(x, y, s=3)
    
    # 一台目の車両の位置をプロット
    x = np.array([j for j in range(ni, nf+1)])
    y = model.x[ni:nf+1, 0]
    if ax == None:
        plt.scatter(x, y, s=3, c="red")
        plt.show()
    else:
        ax.scatter(x, y, s=3, c="red")


def plot_fundamental(ax, model, density, flow):
    ax.set_xlabel("Density")
    ax.set_ylabel("Flow")
    # ax.set_xlim((0, model.L / model.x_0))
    # ax.set_ylim((0, 1.0))
    ax.grid()
    ax.set_title(
        "n0={}, v0={}"
        .format(model.n_0, model.v_0)
    )
    ax.scatter(density, flow, s=5)


if __name__ == "__main__":
    # us2s_OVモデルのインスタンス化
    L = 10.0
    K = 4
    n_0 = 2
    x_0 = 0.1
    v_0 = 0.2
    dt = 1.0
    dx = 1.0
    x_init = np.array([2.0 * i for i in range(K)])
    n_max = 100

    model = ds2s_OV(
        L=L,
        K=K,
        n_0=n_0,
        x_0=x_0,
        v_0=v_0,
        dt=dt,
        dx=dx,
        x_init=x_init,
        n_max=n_max
    )

    print(model.x[0:5])
    print(model.delta_x[0:5])

    # シミュレーション
    # model.simulate()

    # print(model)
    print("{:.10}".format(model.flow(0, 1)))
