

class method2():

    def __init__(self, a = 400, b = 900, c = 0):
        self.a = 100;
        self.b = 200;
        self.c = 300;

    def fit(self, x,y):
        return 22, 33







m = method2().fit(233,333)

print(m[0], m[1])

import matplotlib.pyplot as plt
import numpy as np



#
# # Demonstrate some more complex labels.
# ax = plt.subplot(2, 1, 2)
# plt.plot(x, x**2, label="multi\nline")
# half_pi = np.linspace(0, np.pi / 2)
# plt.plot(np.sin(half_pi), np.cos(half_pi), label=r"$\frac{1}{2}\pi$")
# plt.plot(x, 2**(x**2), label="$2^{x^2}$")
# plt.legend(shadow=True, fancybox=True)

plt.show()