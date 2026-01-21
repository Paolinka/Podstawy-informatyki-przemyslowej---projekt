import numpy as np



class TempStabModel:
    def __init__(
            self,
            y_init, u_init, z_init,
            dt=1,
            ) -> None:
        y=y_init,
        u=u_init,
        z=z_init,
        
        self.c_hat, self.d_hat = self._starting_c_d_calc(y, u, z)
        
        self.c=0,
        self.d=0,
        
        self.dt=dt

    def _starting_c_d_calc(self, y, u, z):
        y_init = y[1:] - y[:-1]

        # macierz regresji
        x_init = np.column_stack([
            u[:-1],
            z[:-1] - y[:-1]
        ])
        
        theta_1, theta_2 = self.mse_loss(x_init, y_init)
        
        c_hat = self.dt / theta_1
        d_hat = theta_2 / theta_1
        
        return c_hat, d_hat
        
    @staticmethod
    def mse_loss(X, Y):
        theta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return theta
    
    def predict_with_horizon(self, y_new, u_new, z_new, h):
        """
        Predykcja h-krokowa:
        y_hat[n] liczona wyłącznie z danych archiwalnych
        """
        N = len(y_new)
        y_hat = np.full(N, np.nan)

        for n in range(h, N):
            y_tmp = y_new[n - h]

            for k in range(h):
                idx = n - h + k
                y_tmp = y_tmp + (self.dt / self.c_hat) * (u_new[idx] + self.d_hat * z_new[idx] - self.d_hat * y_tmp)

            y_hat[n] = y_tmp

        return y_hat
    
    def predict_streaming(self, y_current, u_current, z_current):
        if len(y_current) != len(u_current) != len(z_current):
            raise ValueError("y, u, z must be the same length")
        
        N = len(y_current)
        y_hat = np.full(N, np.nan)
        y_hat[0] = y_current[0]

        for n in range(N - 1):
            y_hat[n + 1] = y_hat[n] + (self.dt / self.c_hat) * (u_current[n] + self.d * z_current[n] - self.d * y_hat[n])

        return y_hat
    
    