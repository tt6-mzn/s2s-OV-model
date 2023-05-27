import sys
from typing import List

"""
us2s_OVモデルを表現するクラス
L  : レーンの長さ
K  : 車両の数
n_0: monitoring period(この時間の間の車間距離を使って次の状態を決定する)
x_0: 最短車間距離
v_0: 車両の最高速度
dt : 時間間隔
"""
class us2s_OV:
    def __init__(self, L: int, K: int, n_0: int, x_0: int, v_0: int, dt: int, x: List[int]) -> None:
        self.L: int = L
        self.K: int = K
        self.n_0: int = n_0
        self.x_0: int = x_0
        self.v_0: int = v_0 
        self.dt: int = dt
        self.x: List[List[int]] = [x for _ in range(n_0 + 1)]
        self.n: int = n_0

    def _delta_x(self, k: int, n: int) -> int:
        ret: int = self.x[n][(k + 1) % self.K] - self.x[n][k]
        if ret >= 0:
            return ret 
        return self.L + ret 
    
    def _next(self) -> None:
        self.x.append([-1 for _ in range(self.K)])
        for k in range(self.K):
            self.x[self.n + 1][k] = (
                self.x[self.n][k]
                + max(0, min([self._delta_x(k, self.n - n_) for n_ in range(0, self.n_0 + 1)]) - self.x_0)
                - max(0, min([self._delta_x(k, self.n - n_) for n_ in range(0, self.n_0 + 1)]) - self.x_0 - self.v_0 * self.dt)
            ) % self.L
        self.n += 1
    
    def simulate(self, nmax: int) -> None:
        while self.n < nmax:
            self._next()
    
    def __repr__(self) -> str:
        return "\n".join(["{:4d}: ".format(i) + "".join("X" if j in self.x[i] else "-" for j in range(self.L))
                            for i in range(self.n + 1)])


def main():
    # us2s_OVモデルのインスタンシエーション
    model = us2s_OV(
        L     = 100,  # レーンの長さ
        K     = 30,  # 車両の数
        n_0   = 2,  # monitoring period
        x_0   = 1,  # 最短車間距離
        v_0   = 1,  # 車両の最高速度
        dt    = 1,  # 時間差分
        x     = [i for i in range(30)]  # 車両の初期位置
    )

    print(model._delta_x(29, 0))

    model.simulate(100)

    print(model)

if __name__ == "__main__":
    main()