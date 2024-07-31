import numpy as np
import bilby 
import matplotlib.pyplot as plot
import scipy
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plot
import tov
from tov import tov
from tov import eos
from tov import tov_utils
from tov import tov_utils_new
import tov.constants as u
from collections import namedtuple
import functools
from functools import partial
from tov import *
import pickle
import pandas as pd
import pickle

# Load data
df12 = np.loadtxt(r"C:\Users\Abhimkul\Downloads\kde_12_O4.txt")
df14 = np.loadtxt(r"C:\Users\Abhimkul\Downloads\kde_14_O4.txt")
df16 = np.loadtxt(r"C:\Users\Abhimkul\Downloads\kde_16_O4.txt")
df18 = np.loadtxt(r"C:\Users\Abhimkul\Downloads\kde_18_O4.txt")

kde_12 = df12[:, 1]
kde_14 = df14[:, 1]
kde_16 = df16[:, 1]
kde_18 = df18[:, 1]

Rs_12 = np.linspace(10.68344335 , 13.07864533, 199)
Rs_14 = np.linspace(10.22067485 , 13.20434239, 200)
Rs_16 = np.linspace(10.36470159 , 13.5683593, 200)
Rs_18 = np.linspace(10.22067485 , 13.20434239, 200)
            
f12 =scipy.interpolate.interp1d(Rs_12, kde_12, fill_value="extrapolate" , kind="cubic")
f14 = scipy.interpolate.interp1d(Rs_14, kde_14, fill_value="extrapolate", kind = "cubic")
f16 = scipy.interpolate.interp1d(Rs_16, kde_16, fill_value="extrapolate", kind = "cubic")
f18 = scipy.interpolate.interp1d(Rs_18, kde_18, fill_value="extrapolate", kind = "cubic")

#prior
my_prior = dict(
    GA1=bilby.core.prior.Uniform(1.71, 4.5, name='GA1', latex_label = r"$\Gamma_1$"),
    GA2=bilby.core.prior.Uniform(1.01, 8, name='GA2', latex_label = r"$\Gamma_2$"),
    GA3=bilby.core.prior.Uniform(1.01, 8, name='GA3', latex_label = r"$\Gamma_3$"),
    log_p1=bilby.core.prior.Uniform(34.3, 34.9, name='log_p1')
)

# Likelihood 
class My_likelihood(bilby.Likelihood):
    

    def __init__(self):
        super().__init__(parameters={"GA1": None, "GA2": None, "GA3": None, "log_p1": None})
        
    def log_likelihood(self):
        # Extract parameters from self.parameters
        GA1 = self.parameters["GA1"]
        GA2 = self.parameters["GA2"]
        GA3 = self.parameters["GA3"]
        log_p1 = self.parameters["log_p1"]

        
        # Solving TOV for the prior EoS 
        try: 
            Params = namedtuple('EoSReadCoreParams', ['logp1', 'G1', 'G2', 'G3'])
            EoS_ = eos.EoS_Read(params=Params(logp1=log_p1, G1=GA1, G2=GA2, G3=GA3))
            solve = tov_utils.TOV_Solver(eos_object=EoS_)

            #using the Pc to calculate R1.X
            A = tov_utils_new.TOV_Solver(eos_object = EoS_)
            Pc_12, a, b = A.get_Pc_M(M = 1.2) 
            Pc_14, a1, b1 = A.get_Pc_M(M = 1.4)
            Pc_16, a2, b2 = A.get_Pc_M(M = 1.6)
            Pc_18, a2, b2 = A.get_Pc_M(M = 1.8)

            #TOV Solve
            M_12, R_12 = solve.tsolve(pct=Pc_12[0]/1000, scode="odeint")
            M_14, R_14 = solve.tsolve(pct=Pc_14[0]/1000, scode="odeint")
            M_16, R_16 = solve.tsolve(pct=Pc_16[0]/1000, scode="odeint")
            M_18, R_18 = solve.tsolve(pct=Pc_18[0]/1000, scode="odeint")
                
            log_likelihood_12 = (np.log(f12(R_12)))
            log_likelihood_14 = (np.log(f14(R_14)))
            log_likelihood_16 = (np.log(f16(R_16)))
            log_likelihood_18 = (np.log(f18(R_18)))
            
            total_log_likelihood = log_likelihood_12 + log_likelihood_14 + log_likelihood_16 + log_likelihood_18

            return total_log_likelihood
        
        except  (ValueError,ZeroDivisionError):
            return -np.inf   ##Probability will be zero at that error (EoS Parameters)

#Final Likelihood
likelihood1 = My_likelihood()

# Run the sampler
result = bilby.run_sampler(
    likelihood=likelihood1,
    priors=my_prior,
    sampler="dynesty",
    nlive = 1000,
    label='Bayesian inference on EoS Parameters',
    outdir='outdir',
    npool = 5,
    verbose = True)


