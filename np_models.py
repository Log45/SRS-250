import numpy as np

# print(np.empty((2, 3)))
# print(np.zeros((2, 3)))
# print(np.random.randn(2))
# print(np.random.random((2, 3))) out=row, in=col

def Linear(X: np.ndarray, input_features: int, output_features: int, bias: bool=True) -> np.ndarray:
    """"""
    weights = np.empty((output_features, input_features))
    if bias:
        bs = np.empty(output_features)
    else:
        bs = np.random.random(output_features)
    return np.dot(weights, X) + bs

def Sigmoid(X: np.ndarray) -> np.ndarray:
    return 1 / (1+np.exp(-X))

def ReLU(X: np.ndarray):
    return

class LinearRegressionModel():
    def __init__(self) -> None:
        """"""
        self.weights = np.random.randn(1)
        self.bias = np.random.randn(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.dot(self.weights, X) + self.bias
    
class ClassificationModel():
    def __init__(self, input_features: int, output_features: int, hidden_units: int=8) -> None:
        self.in_features=input_features
        self.out_features=output_features
        self.hidden_units=hidden_units

    # def Linear(self, X: np.ndarray, input_features: int, output_features: int, bias: bool=True) -> np.ndarray:
    #     """"""
    #     return np.dot(self.weights, X) + self.bias
    
    # def Sigmoid(self, X: np.ndarray) -> np.ndarray:
    #     return 1 / (1+np.exp(-X))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """"""
        X = Linear(X, self.in_features, self.hidden_units)
        X = ReLU(X)
        return Linear(X, self.hidden_units, self.out_features)
    
    # TODO: Create an update_weights function