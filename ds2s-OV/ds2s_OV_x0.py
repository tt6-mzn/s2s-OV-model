# Path: ds2s-OV/ds2s_OV.py
import ds2s_OV
import numpy as np


class ds2s_OV_x0(ds2s_OV.ds2s_OV):
    def _next(self) -> None:
        self.x[self.n+1] = self.x[self.n] + np.where(
            self._delta() <= self.delta_x[self.n] - self.x_0,
            self._delta(),
            self.delta_x[self.n] - self.x_0
        )
        self._periodic()
        self._update_delta_x(self.n+1)
        self.n += 1

    def density(self):
        return super().density() * self.x_0
