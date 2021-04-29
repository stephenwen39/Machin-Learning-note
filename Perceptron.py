import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

'''
函數定義
'''
class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):#這裡直接指定初始值
        self.eta = eta #learning rate學習速度
        self.n_iter = n_iter #loop times迭代次數
        self.random_state = random_state 
        
    def fit(self, X, y):
        
        self.w_ = np.random.RandomState(self.random_state).normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) 
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi)) #更新權重
                self.w_[1:] += update * xi 
                
                self.w_[0] += update #第0號索引元素是偏誤參數，更新偏誤參數
                
                errors += int(update != 0.0) 
                '''
                if update != 0.0:
                  errors += int(update)+1
                else:
                  errors = int(errors)
                '''
                 
            self.errors_.append(errors) #到第6次收斂
        return self
        #fit跑到收斂的時候就完全不會動了，所以繼續loop下去沒有意義

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0] #預測是用點積實作，每個花瓣長度資料跟權重做點積再加上偏差值

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1) #直接以上一個函數做輸入，如果輸入值比零大(含)就標示一
        #所以原則上0就是門檻值，可以直觀的理解成原始值與權重點積以後加上偏差值如果大於門檻值的話就分成第一類，否則分成第二類
        
'''
資料import
'''
s = os.path.join('https://archive.ics.uci.edu', 'ml',
                 'machine-learning-databases', 'iris','iris.data')
print('URL:', s)

df = pd.read_csv(s,
                 header=None,
                 encoding='utf-8')

df.tail()

'''
資料提取
'''
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1) #直接做矩陣替換運算

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values #從第0個資料到第100個資料，提取第一欄跟第三欄

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_06.png', dpi=300)
plt.show()

'''
實際訓練
'''
ppn = Perceptron(0.1, 7, 1)
#後來實作的時候宣告的參數會覆蓋本來的參數（以新的參數為準）
ppn.fit(X, y)
#實作fit，輸入上面的Xy，完成學習
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()

'''
建立視覺話函數
'''
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

'''
視覺化
'''
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')



# plt.savefig('images/02_08.png', dpi=300)
plt.show()
