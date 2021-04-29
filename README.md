# Machin-Learning-note

# 此機器學習筆記內容除中文註解之外皆非我自創，而是來自Python machine learning 3edition 一書

接下來的範例皆以"Anderson's Iris data set“為資料輸入

## Perceptron classifier

為一簡單的神經元線性分類器，下圖為其分類效果
<p align="center"><img width="40%" src="Perceptron.png" /></p>

## Adaline classifier

為Perceptron classifier的改進

使用批次梯度下降(batch gradient descent)為收斂方式
<p align="center"><img width="40%" src="adaline.png" /></p>


在Adaline classifier的基礎上採用隨機梯度下降(stochastic gradient descent)

可見收斂效率快速增長

<p align="center"><img width="40%" src="adalineSGD.png" /></p>

## LogisticRegression

在展現決策邊界之後，採用LogisticRegression分類器的多元分類功能

決策邊界圖
<p align="center"><img width="40%" src="decision_regions.png" /></p>

LogisticRegression視覺化
<p align="center"><img width="40%" src="LogisticRegression1.png" /></p>

正規化的影響
<p align="center"><img width="40%" src="normal.png" /></p>


## SVM

下方展示SVM在不同資料集與不同參數下的呈現
此圖展示其於鳶尾花資料集的一般分類
<p align="center"><img width="40%" src="svm1.png" /></p>

此圖展現其於亂數資料集的非線性分類
<p align="center"><img width="40%" src="svm2.png" /></p>

下兩圖為不同的gamma對SVM分類效果的影響
<p align="center"><img width="40%" src="svm3.png" /></p>
<p align="center"><img width="40%" src="svm4.png" /></p>
