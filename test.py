from mgradscratch import *

#x1 = Value(2.0)
#x2 = Value(0.0)
#w1 = Value(-3.0)
#w2 = Value(1.0)
#b = Value(6.8813735870195432)
#x1w1 = x1 * w1
#x2w2 = x2 * w2
#x1w1_x2w2 = x1w1 + x2w2
#n = x1w1_x2w2 + b
#o = n.tanh()
#o.backward()

x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.8813735870195432)
x1w1 = x1 * w1
x2w2 = x2 * w2
x1w1_x2w2 = x1w1 + x2w2
n = x1w1_x2w2 + b
c = (2*n).exp()
o = (c - 1) / (c + 1)
print(o)
o.backward()

print(x1.grad)