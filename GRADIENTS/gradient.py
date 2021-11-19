import numpy

class Variable():

    inputs = []
    num_variables = 0

    def __init__(self, name = None, evaluate = None, gradient = None):
        if evaluate == None:
            self.evaluate = lambda values: values[self.name]
        else:
            self.evaluate = evaluate 

        if gradient == None: # thank you to Madison and Tanush for helping me talk through the creation of my returned gradient!
            Variable.num_variables += 1 
            self.current = Variable.num_variables
            output = [0] * Variable.num_variables
            output[self.current-1] = 1
            self.gradient = lambda values: numpy.array(output+[0]*(Variable.num_variables - self.current))
        else:
            self.gradient = gradient
        
        if name != None:
            self.name = name
        
        self.inputs.append(name)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) + other, gradient = lambda values: self.gradient(values))
        return Variable(evaluate = lambda values: self.evaluate(values) + other.evaluate(values), gradient = lambda values: self.gradient(values) + other.gradient(values))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(other.__mul__(-1))

    def __rsub__(self, other):
        return self.__add__(other.__mul__(-1))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) * other, gradient = lambda values: other * self.gradient(values))
        return Variable(evaluate = lambda values: self.evaluate(values) * other.evaluate(values), gradient = lambda values: self.evaluate(values) * other.gradient(values) + other.evaluate(values) * self.gradient(values))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(other.__pow__(-1))

    def __rtruediv__(self, other):
        return self.__mul__(other.__pow__(-1))

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) ** other, gradient = lambda values: other * self.evaluate(values)**(other-1) * self.gradient(values))
        return Variable(evaluate = lambda values: self.evaluate(values) ** other.evaluate(values)) # no gradient here :D
    
    @classmethod
    def exp(cls, other):
        return Variable(
            evaluate = lambda values: numpy.exp(other.evaluate(values)),
            gradient = lambda values: numpy.exp(other.evaluate(values)) * other.gradient(values)
        )
   
    @classmethod
    def log(cls, other):
        return Variable(
            evaluate = lambda values: numpy.log(other.evaluate(values)),
            gradient = lambda values: 1/other.evaluate(values) * other.gradient(values)
        )