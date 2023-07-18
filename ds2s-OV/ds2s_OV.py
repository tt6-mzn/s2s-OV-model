import random
from typing import List
import numpy as np

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
        L     : np.float64,  # レーンの長さ
        K     : np.int32,  # 車両の台数
        n_0   : np.int32,  # monitoring period
        x_0   : np.int32,  # 車間距離
        v_0   : np.float64,  # 車両の最高速度
        dt    : np.float64,  # 時間差分
        dx    : np.float64,  # 空間差分
        x_init: np.ndarray[np.float64],  # 車両の初期位置
        n_max : np.int32
    ) -> None:
        self.L     = L  # レーンの長さ
        self.K     = K  # 車両の数
        self.n_0   = n_0  # monitoring period
        self.x_0   = x_0  # 最短車間距離
        self.v_0   = v_0  # 車両の最高速度
        self.dt    = dt  # 時間間隔
        self.dx    = dx  # 空間差分
        self.n_max = n_max  # シミュレーションするステップ数
        self.n     = n_0  # 現在のステップ数
        # 各時刻における車両の位置 x[n][k] := x_k^n
        self.x     = np.full(
            shape=(self.n_max + 1, self.K),
            fill_value=-1,
            dtype=np.float64
        )

        # 車両の初期位置
        x_init = np.sort(x_init)
        if self.K == 0: return  # 車両が0台のとき
        self.x[:n_0+1] = x_init[None, :]
        self.delta_x = np.full(
            shape=(self.n_max + 1, self.K),
            fill_value=-1,
            dtype=np.float64
        )

        # 前方の車両との車間距離
        if self.K == 1:
            self.delta_x[:self.n_0+1, 0] = self.L
            return

        self.delta_x[:self.n_0+1,  self.K-1] = \
            self.x[:self.n_0+1, 0       ] - self.x[:self.n_0+1,  self.K-1]
        self.delta_x[:self.n_0+1,  self.K-1] = \
            np.where(
                self.delta_x[:self.n_0+1,  self.K-1] >= 0.0,
                self.delta_x[:self.n_0+1,  self.K-1],
                self.delta_x[:self.n_0+1,  self.K-1] + self.L
            )

        self.delta_x[:self.n_0+1, :self.K-1] = \
            self.x[:self.n_0+1, 1:self.K] - self.x[:self.n_0+1, :self.K-1]
        self.delta_x[:self.n_0+1, :self.K-1] = \
            np.where(
                self.delta_x[:self.n_0+1, :self.K-1] >= 0.0,
                self.delta_x[:self.n_0+1, :self.K-1],
                self.delta_x[:self.n_0+1, :self.K-1] + self.L,
            )
    
    # ステップを一つ進める
    def _next(self) -> None:
        # 各車両における\delta_eff x_kをを計算する
        delta_eff = self.delta_x[self.n-self.n_0:self.n+1] - self.x_0
        e1 = np.sum(
            np.exp(
                -delta_eff / self.dx
            ),
            axis=0
        )
        e1 /= self.n_0 + 1
        e2 = np.sum(
            np.exp(
                -(delta_eff - self.v_0 * self.dt) / self.dx
            ),
            axis=0
        )
        e2 /= self.n_0 + 1

        self.x[self.n+1] = \
            self.x[self.n] \
            + self.dx * (
                np.log(1.0 + 1.0 / e1) \
                - np.log(1.0 + np.exp(-self.x_0 / self.dx)) \
                - np.log(1.0 + 1.0 / e2) \
                + np.log(1.0 + np.exp(-(self.x_0 + self.v_0 * self.dt) / self.dx))
            )
        self.x[self.n+1, :self.K] = \
            np.where(
                self.x[self.n+1, :self.K] < self.L,
                self.x[self.n+1, :self.K],
                self.x[self.n+1, :self.K] - self.L
            )
        
        # 車両の追い抜きに対応
        self.x[self.n+1, :self.K] = np.sort(self.x[self.n+1, :self.K])

        # 車間距離の更新
        if self.K == 1:
            self.delta_x[self.n+1, 0] = self.L
        else:
            self.delta_x[self.n+1,  self.K-1] = \
                self.x[self.n+1, 0       ] - self.x[self.n+1,  self.K-1]
            self.delta_x[self.n+1,  self.K-1] = \
                np.where(
                    self.delta_x[self.n+1,  self.K-1] >= 0.0,
                    self.delta_x[self.n+1,  self.K-1],
                    self.delta_x[self.n+1,  self.K-1] + self.L,
                )
            self.delta_x[self.n+1, :self.K-1] = \
                self.x[self.n+1, 1:self.K] - self.x[self.n+1, :self.K-1]
            self.delta_x[self.n+1, :self.K-1] = \
                np.where(
                    self.delta_x[self.n+1, :self.K-1] >= 0.0,
                    self.delta_x[self.n+1, :self.K-1],
                    self.delta_x[self.n+1, :self.K-1] + self.L,
                )
        self.n += 1
    
    # nmaxまでシミュレーションを行う
    def simulate(self) -> None:
        if self.K == 0: return
        while self.n < self.n_max:
            self._next()
    
    # n_1ステップからn_2ステップまでの流量を計算
    def flow(self, n_1: int, n_2: int) -> float:
        if self.K == 0: return 0.0
        return np.sum(
            self.delta_x[n_1:n_2+1, :self.K] \
                / (self.dt * (n_2 - n_1 + 1) * self.L)
        )

    def density(self):
        return np.float64(self.K) / self.L


if __name__ == "__main__":
    # us2s_OVモデルのインスタンス化
    L = 10.0
    K = 30
    n_0 = 2
    x_0 = 0.1
    v_0 = 0.2
    dt = 1.0
    dx = 1.0
    x_init = np.array([(2 * i) / 10 for i in range(30)])
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

    # シミュレーション
    model.simulate()

    print(model)
    print("{:.10}".format(model.flow(0, 99)))
