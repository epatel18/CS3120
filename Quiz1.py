import numpy as np
import matplotlib.pyplot as plt

# Returns evenly spaced numbers over a specified interval
# https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
Xaxis = np.linspace(0, 10)
Yaxis = np.linspace(0, 10)

# Returns the coordinate matrices from coordinate vectors
# https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html?highlight=meshgrid#numpy.meshgrid
X, Y = np.meshgrid(Xaxis, Yaxis)

# Defines a variable Z = (X-5)^2 + (Y-5)^2
Z = np.power(X-5, 2) + np.power(Y-5, 2)

# Taking this from the sample code given to us
plt.contourf(Xaxis, Yaxis, Z, 20, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([5], [5], 'o', ms=12, markeredgewidth=3, color='orange')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel(r'$X$', fontsize=16)
plt.ylabel(r'$Y$', fontsize=16)

# Displaying the title to the figure as Eesha_Patel_Quiz1Contour
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.title.html
plt.title('Eesha_Patel_Quiz1Contour')

plt.show()