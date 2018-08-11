from pyrism import I2EM

iza = 30
vza = 35
raa = 50

b = I2EM(iza, vza, raa, normalize=False, nbar=0.0, angle_unit='DEG', frequency=1.26, eps=20 + 40j,
         corrlength=5, sigma=0.5, n=10, corrfunc='exponential')
