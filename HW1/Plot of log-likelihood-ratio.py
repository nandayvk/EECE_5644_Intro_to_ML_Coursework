import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-15, 15)
y = np.ln(2) + abs(x-1)/2 - abs(x)
plt.title('Plot of log-likelihood-ratio function')
plt.xlabel('values of x')
plt.ylabel('values of l(x)')
plt.plot(x, y, 'b--')
plt.show()
