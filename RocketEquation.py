import numpy as np
import matplotlib.pyplot as plt

# Constants of nature
g = 9.8

# Rocket mass parameters
M = 1.0
mdot = 0.05
vex = 400.0

y = lambda _t: (vex - 0.5 * g * _t) * _t + np.log((M - mdot * _t) / M) * (vex * M / mdot - vex * _t)
vy = lambda _t: vex * np.log(M / (M - mdot * _t)) - g * _t
ay = lambda _t: mdot * vex / (M - mdot * _t) - g

y2 = lambda _t, _yi, _vi: _yi + _vi * _t - 0.5 * g * _t * _t
vy2 = lambda _t, _vyi: _vyi - g * _t

t1 = np.linspace(0.0, 10.0, 1000)
t3 = np.linspace(0.0, 41.0, 2000)
t2 = t3 + 10.0
_y1 = y(t1)
_vy = vy(t1)
_ay = ay(t1)

_y2 = y2(t3, _y1[-1], _vy[-1])

print(f'Final height: {_y1[-1]:.3f} m')
print(f'Final velocity: {_vy[-1]:.3f} m/s')
print(f'Final acceleration: {_ay[-1]:.3f} m/s^2')

plt.plot(t1, _y1)
plt.plot(t1, _vy)
plt.plot(t1, _ay)
plt.plot(t2, _y2)
plt.plot(t2, vy2(t3, _vy[-1]))
plt.plot(t2, -g * np.ones(2000))
plt.legend([
    'y1(t)',
    'vy(t)',
    'ay(t)',
    'y2(t)',
    'vy2(t)',
    'ay2(t)'
])
plt.show()