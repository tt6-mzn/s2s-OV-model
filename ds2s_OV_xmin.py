from ds2s_OV import ds2s_OV
import numpy as np


class ds2s_OV_xmin(ds2s_OV):
    def __init__(
        self,
        L: np.float64,  # レーンの長さ
        K: np.int32,  # 車両の台数
        n_0: np.int32,  # monitoring period
        x_0: np.int32,  # 車間距離
        v_0: np.float64,  # 車両の最高速度
        dt: np.float64,  # 時間差分
        dx: np.float64,  # 空間差分
		x_min: np.float64,  # 最短車間距離
        x_init: np.ndarray[np.float64],  # 車両の初期位置
        n_max: np.int32
    ) -> None:
        super().__init__(
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
        self.x_min = x_min
	
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
            x_min=json["x_min"],
            x_init=np.array(json["x_init"]),
            n_max=n_max
        )


    def _next(self) -> None:
        self.v[self.n+1] = np.where(
            self._delta() <= self.delta_x[self.n] - self.x_min,
            self._delta() / self.dt,
            (self.delta_x[self.n] - self.x_min) / self.dt
        )
        self.x[self.n+1] = self.x[self.n] + self.v[self.n+1] * self.dt
        # self.x[self.n+1] = self.x[self.n] + np.where(
        #     self._delta() <= self.delta_x[self.n] - self.x_min,
        #     self._delta(),
        #     self.delta_x[self.n] - self.x_min
        # )
        self._periodic()
        self._update_delta_x(self.n+1)
        self.n += 1
    
    def density(self):
        return self.K / self.L

    def get_json(self):
        return super().get_json() | {"x_min": self.x_min}

    def model_type(self):
        return super().model_type() + "_xmin"

    # 自明解の流量
    @classmethod
    def flow_stable(
            cls,
            density: np.float64,
            x_0: np.float64,
            v_0: np.float64,
            dt: np.float64,
            dx: np.float64,
            xmin: np.float64,
        ):
        if density == 0.0:
            return 0.0
        left = dx * (
            np.log(1 + np.exp((1.0/density - x_0)/dx))
            - np.log(1 + np.exp(-x_0/dx))
            - np.log(1 + np.exp((1.0/density - x_0 - v_0*dt)/dx))
            + np.log(1 + np.exp(-(x_0 + v_0*dt)/dx))
        )
        right = 1.0/density - xmin
        return density * min(left, right)
