from . import ds2s_OV_xmin
import numpy as np


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
        x_min: np.float64,  # 最短車間距離
        forget: np.float64,  # 忘却係数
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
            x_min=x_min,
            x_init=x_init,
            n_max=n_max
        )
        self.forget = np.zeros(shape=(self.n_0 + 1, self.K), dtype=np.float64)
        for n_ in range(self.n_0 + 1):
            self.forget[n_] = forget[self.n_0 - n_]
    

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
            forget=json["forget"],
            x_init=np.array(json["x_init"]),
            n_max=n_max
        )
    
    
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


    def density(self):
        return self.K / self.L


    def get_json(self):
        return super().get_json() | { "forget": self.forget.tolist() }
