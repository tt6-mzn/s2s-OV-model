from . import ds2s_OV
import numpy as np


class ds2s_OV_xmin(ds2s_OV.ds2s_OV):
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
        n_max: np.int32
    ) -> None:
        super().__init__(L, K, n_0, x_0, v_0, dt, dx, x_init, n_max)
        self.xmin = xmin
	
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
            xmin=json["xmin"],
            x_init=np.array(json["x_init"]),
            n_max=n_max
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
    
    def density(self):
        return self.xmin * self.K / self.L

    def get_json(self):
        return super().get_json() | {"xmin": self.xmin}
