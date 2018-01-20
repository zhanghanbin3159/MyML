#encoding:utf-8
import numpy as np
import tensorflow as tf

#样本实例
coefficients = np.array([[1.],[-20.],[100]])
print coefficients.shape
#初始化W
w = tf.Variable(0,dtype=tf.float32)
#样本实例的输入变量X
x = tf.placeholder(tf.float32, [3,1])
#花销函数
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
#梯度下降算法
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print session.run(w)

for i in range(1000):
    #通过feed_dict将变量X和coefficients关联
    session.run(train, feed_dict={x:coefficients})
print session.run(w)
