from typing import List
import numpy as np

"""
us2s_OVモデルを表現するクラス
L   : レーンの長さ
K   : 車両の数
n_0 : monitoring period(この時間の間の車間距離を使って次の状態を決定する)
x_0 : 最短車間距離
v_0 : 車両の最高速度
dt  : 時間間隔
x   : 車両の初期位置
nmax: シミュレーションするステップ数
"""
class us2s_OV:
    def __init__(self, L: int, K: int, n_0: int, x_0: int, v_0: int, dt: int, x: List[int], nmax: int) -> None:
        self.L: int = L  # レーンの長さ
        self.K: int = K  # 車両の数
        self.n_0: int = n_0  # monitoring period
        self.x_0: int = x_0  # 最短車間距離
        self.v_0: int = v_0  # 車両の最高速度
        self.dt: int = dt  # 時間間隔
        self.nmax: int = nmax  # シミュレーションするステップ数
        self.n: int = n_0  # 現在のステップ数
        self.x: np.ndarray = np.ndarray(shape=(nmax + 1, K), dtype=np.int32)  # 各時刻における車両の位置

        # 車両の初期位置
        for i in range(n_0 + 1):
            self.x[i] = x
    
	# 時刻nにおける車両kと車両k+1の車間距離を計算する関数
    def _delta_x(self, k: int, n: int) -> int:
        ret: int = self.x[n][(k + 1) % self.K] - self.x[n][k]
        if ret >= 0:
            return ret 
        return self.L + ret 
    
    # ステップを一つ進める
    def _next(self) -> None:
        for k in range(self.K):
            delta_eff: int = min([self._delta_x(k, self.n - n_) for n_ in range(0, self.n_0 + 1)]) - self.x_0
            self.x[self.n + 1][k] = (
                self.x[self.n][k]
                + max(0, delta_eff)
                - max(0, delta_eff - self.v_0 * self.dt)
            ) % self.L
        self.n += 1
    
    # nmaxまでシミュレーションを行う
    def simulate(self) -> None:
        while self.n < self.nmax:
            self._next()
    
    def __repr__(self) -> str:
        return "\n".join(["{:4d}: ".format(i) + "".join("X" if j in self.x[i] else "-" for j in range(self.L))
                            for i in range(self.n + 1)])
    
    # n_1ステップからn_2ステップまでの流量を計算
    def flow(self, n_1: int, n_2: int):
        ret = 0.0
        for k in range(self.K):
            for n in range(n_1, n_2 + 1):
                tmp = (self.x[n + 1][k] - self.x[n][k])
                if (tmp < 0): tmp = self.L + tmp
                tmp /= self.dt
                tmp /= (n_2 - n_1 + 1) * self.L
                ret += tmp
        return ret

    def density(self):
        return self.K / self.L


def main():
    # us2s_OVモデルのインスタンシエーション
    model = us2s_OV(
        L    = 100,  # レーンの長さ
        K    = 30,  # 車両の数
        n_0  = 2,  # monitoring period
        x_0  = 1,  # 最短車間距離
        v_0  = 1,  # 車両の最高速度
        dt   = 1,  # 時間差分
        x    = [i for i in range(30)],  # 車両の初期位置
        nmax = 1000  # シミュレーションするステップ数
    )

    model.simulate()

    print(model)
    print(model.flow(0, 999))

if __name__ == "__main__":
    main()
