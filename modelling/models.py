import brainpy as bp
import brainpy.math as bm
from bold_simu import BOLD

class WholeBrainNet(bp.Network):
  def __init__(self, Cmat, Dmat, num_regions, wEE = 12, wEI = 13, wIE = 4, wII = 11):
    super(WholeBrainNet, self).__init__()

    self.fhn = bp.dyn.WilsonCowanModel(
      num_regions,
      x_ou_sigma=0.01,
      y_ou_sigma=0.01,
      wEE = wEE, #12
      wEI = wEI, #13
      wIE = wIE, #4
      wII = wII, #11
      E_theta = 4,
      I_theta = 2.8,
      E_a = 0.6,
      I_a = 1.4,
      E_tau = 1,
      I_tau = 0.1,
      method='exp_euler_auto'
    )
    self.syn = bp.dyn.DiffusiveCoupling(
      self.fhn.x,
      self.fhn.x,
      var_to_output=self.fhn.input,
      conn_mat=Cmat,
      delay_steps=Dmat.astype(bm.int_),
      initial_delay_data=bp.init.Uniform(0, 0.05)
    )

class WholeBrainWithBOLD(bp.DynSysGroup):
    def __init__(self, conn_mat, delay_mat, num_regions, wEE, wEI, wIE, wII):
        super().__init__()
        self.net = WholeBrainNet(conn_mat, delay_mat, num_regions, wEE, wEI, wIE, wII)
        self.bold = BOLD(num_regions=num_regions)

    def update(self):
        t = bp.share['t']
        self.net.update(t)
        # E_t = self.net.fhn.x  # 或者你网络里相应的活动变量
        E_t = self.net.fhn.x.value if hasattr(self.net.fhn.x, 'value') else self.net.fhn.x
        self.bold.step(E_t)
