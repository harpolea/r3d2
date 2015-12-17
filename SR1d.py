import numpy as np

class SR1d():

    def __init__(self, t_end, w_left, w_right, gamma=5./3.):
        """
        Constructor
        """
        self.t_end = t_end
        self.gamma = gamma

        # compute initial left and right states given
        # the primitive variables
        self.q_left = self.compute_state(w_left)
        self.q_right = self.compute_state(w_right)


    def compute_state(self, w):
        """
        Convert from the basic primitive variables w = (rho, v, eps) to the full
        set q = (rho, v, eps, p, W, h, cs^2)
        """
        rho, v, eps = w

        p = (gamma - 1.) * rho * eps
        W_lorentz = 1. / np.sqrt( 1. - v**2)
        h = 1. + eps + p / rho
        cs2 = gamma * p / (rho * h)

        return np.array([rho, v, eps, p, W_lorentz, h, cs2])


    def find_p_star(self):
        pass

    def get_state(self):
        pass

    def get_wave_speeds(self):
        pass

    def rarefaction(self):
        pass

    def riemann_solver(self):
        pass

    def plot_results(self):
        pass 




if __name__ == "__main__":
    pass
