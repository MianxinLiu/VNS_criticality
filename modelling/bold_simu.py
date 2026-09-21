import brainpy as bp
import brainpy.math as bm

class BOLD(bp.NeuGroup):
    def __init__(self, num_regions,
                 kappa=0.65, gamma=0.41, tau=0.98,
                 alpha=0.32, E0=0.34, V0=0.02,
                 k1=7.0, k2=2.0, k3=2.0,
                 dt=0.1):
        super().__init__(size=num_regions)

        self.num = num_regions
        self.dt = dt

        self.kappa = kappa
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.E0 = E0
        self.V0 = V0
        self.k1, self.k2, self.k3 = k1, k2, k3

        # ✅ 状态必须是 Variable
        self.s = bm.Variable(bm.zeros(num_regions, dtype=bm.float32))
        self.f = bm.Variable(bm.ones(num_regions,  dtype=bm.float32))
        self.v = bm.Variable(bm.ones(num_regions,  dtype=bm.float32))
        self.q = bm.Variable(bm.ones(num_regions,  dtype=bm.float32))

        # ✅ BOLD 也做成 Variable，方便 monitor
        self.BOLD = bm.Variable(bm.zeros(num_regions, dtype=bm.float32))

    def step(self, neural_activity):
        # neural_activity: shape = (num_regions, )

        s = self.s.value
        f = self.f.value
        v = self.v.value
        q = self.q.value

        ds = neural_activity - self.kappa * s - self.gamma * (f - 1.)
        df = s
        dv = (f - v ** (1. / self.alpha)) / self.tau
        dq = (f * (1. - (1. - self.E0) ** (1. / f)) / self.E0
              - q / v ** (1. - 1. / self.alpha)) / self.tau

        s_new = s + ds * self.dt
        f_new = f + df * self.dt
        v_new = v + dv * self.dt
        q_new = q + dq * self.dt

        self.s.value = s_new
        self.f.value = f_new
        self.v.value = v_new
        self.q.value = q_new

        bold_t = self.V0 * (
            self.k1 * (1. - q_new) +
            self.k2 * (1. - q_new / v_new) +
            self.k3 * (1. - v_new)
        )

        # ✅ BOLD 更新到 Variable 里
        self.BOLD.value = bold_t

        return bold_t
