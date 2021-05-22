import numpy as np
import matplotlib.pyplot as plt

#generate data samples
x_data = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
y_data = 2*x_data+50+5*np.random.random(10)

#plott the landscapre of the loss function
bb = np.arange(0, 100, 1) 
ww = np.arange(-5, 5, 0.1)  
Z = np.zeros((len(bb), len(ww)))

for i in range(len(bb)):
    for j in range(len(ww)):
        b = bb[i]
        w = ww[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (w*x_data[n]+b - y_data[n]
                                 )**2

        Z[j][i] = Z[j][i]/len(x_data)

#assign inital weights for the gradient descent process
b = 0  
w = 0  
lr = 0.0001  
iteration = 10000

iterations = 0
target_gradient = 0.0001

#store parameters for plotting
b_history = [b]
w_history = [w]

#model by gradient descent
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    iterations += 1
    for n in range(len(x_data)):
        b_grad += w*x_data[n] + b - y_data[n]
        w_grad += (w*x_data[n] + b - y_data[n])*x_data[n]
         #terminate the loop when target gradient is reached
        if abs(b_grad) < target_gradient and abs(w_grad) < target_gradient:
            print('Reached target gradient of less than 0.0001 in {iterations} iterations.')
            break
    b -= lr * b_grad
    w -= lr * w_grad
    b_history.append(b)
    w_history.append(w)

#print necessary information to the terminal
print('Weight:\n', w, '\nBias:\n', b, '\nIteration:\n', iteration)

#construct the graph
plt.xlim(0, 100)
plt.ylim(-5, 5)
plt.xlabel('b = %s' % (b))
plt.ylabel('w = %s' % (w))
plt.contourf(bb, ww, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot(b_history, w_history, 'o-', ms=1.0, lw=1.5, color='black')
plt.plot(b, w, 'x', ms=10, c='red')
plt.show()