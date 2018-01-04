# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:16:39 2017

@author:
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


#准备数据
#linspace 创建0-10之间100个矢量
x_vals = np.linspace(0, 10, 100)
y_vals = 2 * x_vals + np.random.normal(0, 1, 100)


sess = tf.Session()

x_vals_column = np.transpose(np.matrix(x_vals))
one_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, one_column))
b = np.transpose(np.matrix(y_vals))

#转化为张量
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

#使用正规方程法
A_temp = tf.matmul(tf.transpose(A_tensor), A_tensor)
A_temp = tf.matrix_inverse(A_temp)
A_temp = tf.matmul(A_temp, tf.transpose(A_tensor))
solution = tf.matmul(A_temp, b_tensor)


solution_eval = sess.run(solution)

#得到系数，截距
slope = solution_eval[0][0]
intercept = solution_eval[1][0]

print('slope'+str(slope))
print('intercept'+str(intercept))


#画图显示
best_fit = []

for i in x_vals:
    best_fit.append(slope * i + intercept)
plt.plot(x_vals, y_vals, 'o', label ='Data')
plt.plot(x_vals, best_fit, 'r-', label ='Best fit line')
plt.legend(loc = 'upper left')
plt.show()