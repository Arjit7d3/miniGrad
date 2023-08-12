import math

class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward =  backward

        return out

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * (other ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,))

        def backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward

        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self, ))

        def backward():
            self.grad = (1 - math.tanh(self.data)**2) * out.grad
        out._backward = backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ))

        def backward():
            self.grad = out.data * out.grad 
        out._backward = backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

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