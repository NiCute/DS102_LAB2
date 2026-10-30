import numpy as np

class Logistic_Regression_NumPy:
    def __init__(self):
        self.w0 = 0
        self.w1 = None
        self.losses = []
    def Sigmoid(self, z):
        return 1/(1+np.exp(-z))
    def LossFunction(self,y, p):
        m = len(y)
        loss = - (1 / m) * np.sum(y * np.log(p)+(1-y)*np.log(1-p))
        return loss
    def Gradient_Descent_Logistic(self,x,y,lr =0.01,loops = 1000):
        # m la so dong, k la so cot
        m , k = x.shape
        self.w0 = 0
        self.w1 = np.zeros(k)
        self.losses = []
        for i in range(loops):
            #a1 = w0 + w1 * x
            a1 = self.w0 + x @ self.w1
            p = self.Sigmoid(a1)

            loss = self.LossFunction(y, p)
            self.losses.append(loss)

            dw1 = (1/m) * (x.T @ (p-y))
            dw0 = (1/m) * np.sum(p - y)

            self.w1 = self.w1 - lr * dw1 
            self.w0 = self.w0 - lr * dw0
        return self.losses
    def evaluate(self, y, p):
        p = (p >= 0.5).astype(int)
        
        tp = np.sum((y == 1) & (p == 1))
        fp = np.sum((y == 0) & (p == 1))
        fn = np.sum((y == 1) & (p == 0))
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return {"precision": precision, "recall": recall, "f1_score": f1}

    def predict(self, x):
        a1 = self.w0 + x @ self.w1
        p = self.Sigmoid(a1)
        return p
