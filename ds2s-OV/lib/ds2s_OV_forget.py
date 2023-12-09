from . import ds2s_OV_xmin
import numpy as np
from typing import Callable


class ds2s_OV_forget(ds2s_OV_xmin.ds2s_OV_xmin):
    def __init__(
        self,
        L: np.float64,  # レーンの長さ
        K: np.int32,  # 車両の台数
        n_0: np.int32,  # monitoring period
        x_0: np.int32,  # 車間距離
        v_0: np.float64,  # 車両の最高速度
        dt: np.float64,  # 時間差分
        dx: np.float64,  # 空間差分
		xmin: np.float64,  # 最短車間距離
        x_init: np.ndarray[np.float64],  # 車両の初期位置
        forget: np.ndarray[np.float64],  # 忘却率
        n_max: np.int32,
    ) -> None:
        super().__init__(L, K, n_0, x_0, v_0, dt, dx, xmin, x_init, n_max)
        self.forget = forget

    # 移動距離の計算
    def _delta(self) -> np.ndarray[np.float64]:
        delta_eff = self.delta_x[self.n-self.n_0:self.n+1] - self.x_0
        e1 = np.sum(
            self.forget * np.exp(
                -delta_eff / self.dx
            ) / (self.n_0 + 1),
            axis=0
        )
        e2 = np.sum(
            self.forget * np.exp(
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


    def _next(self) -> None:
        self.x[self.n+1] = self.x[self.n] + np.where(
            self._delta() <= self.delta_x[self.n] - self.xmin,
            self._delta(),
            self.delta_x[self.n] - self.xmin
        )
        self._periodic()
        self._update_delta_x(self.n+1)
        self.n += 1
    
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

    model = ds2s_OV_forget(
        L=L,
        K=K,
        n_0=n_0,
        x_0=x_0,
        v_0=v_0,
        dt=dt,
        dx=dx,
        x_init=x_init,
        n_max=n_max,
        callback=lambda n: 1.0 if n == 0 else 0.0
    )

    print(model._delta())

    # シミュレーション
    # model.simulate()

    # print(model)
    print("{:.10}".format(model.flow(0, 1)))