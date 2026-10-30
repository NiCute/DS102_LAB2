import numpy as np

class Softmax_Regression_Numpy:
    def __init__(self):
        self.w0 = 0
        self.w1 = None
        self.losses = []
    def Softmax(self, z):
        e = np.exp(z)
        for i in range(len(e)):
            e[i] = e[i]/np.sum(e[i])
        return e
    def LossFunction(self, y, p):
        m = len(y)
        loss = -(1/m)*np.sum(y* np.log(p))
        return loss
    def Gradient_Descent_SoftMax(self, x, y, lr = 0.01, loops = 1000):
        m, k = x.shape
        self.w0 = np.zeros(10)
        self.w1 = np.zeros((k, 10))
        y_moi = np.zeros((m, 10))
        for i in range(m):
            y_moi[i][y[i]] = 1
        for i in range(loops):
            a1 = self.w0 + x @ self.w1
            p = self.Softmax(a1)
            loss = self.LossFunction(y_moi, p)
            self.losses.append(loss)
            dw1 = (1/m) * (x.T @ (p - y_moi))
            dw0 = (1/m) * np.sum(p - y_moi, axis = 0)
            self.w1 = self.w1 - lr * dw1
            self.w0 = self.w0 - lr * dw0
        return self.losses
    def evaluate(self, y, p):
        p = np.array([np.argmax(p[i]) for i in range(len(p))])
        precision = 0
        recall = 0
        f1 = 0
        for i in range(10):
            tp = np.sum((y == i) & (p == i))
            fp = np.sum((y != i) & (p == i))
            fn = np.sum((y == i) & (p != i))
            
            pre = tp / (tp + fp)
            rec = tp / (tp + fn)
            precision += pre
            recall += rec
            f1 += 2 * (pre * rec) / (pre + rec)
        return {"precision": precision/10, 'recall': recall/10, "f1_score": f1/10}
    def predict(self, x):
        a1 = self.w0 + x @ self.w1
        p = self.Softmax(a1)
        return p
