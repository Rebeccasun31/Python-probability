# Bivariate and Multivariate Normal for Regression

使用Python計算機率並畫出預測的迴歸圖

We have learned “bivariate normal” in Lectures 19-22. The idea of bivariate normal can be readily extended
to “multivariate normal”. One interesting application of multivariate normal (MVN) random variables is to solve
regression tasks. In this problem, you will implement a simple MVN-based predictor that predicts the outputs
of the testing queries based on the training data. Specifically, let Dtrain = {(x1, y1),(x2, y2), · · · ,(xN , yN )}
be the training dataset and let Dtest = {xN+1, · · · , xN+M} be the testing queries. The goal is to predict
{yN+1, · · · , yN+M} that correspond to {xN+1, · · · , xN+M}.

![image](https://user-images.githubusercontent.com/86657062/224364449-4835e837-28b7-41d3-9fcf-2ea61b3c7c37.png)

![image](https://user-images.githubusercontent.com/86657062/224364564-757aa137-0e5b-4c05-ac50-11b6a2cf0f2f.png)

# Result

- σf = 1, σ = 0.1, ℓ = 0.5

![Figure_1](https://user-images.githubusercontent.com/86657062/224365073-14bfc616-9352-4ed7-9991-c7680259669e.png)

- σf = 1, σ = 0.1, ℓ = 0.05

![Figure_2](https://user-images.githubusercontent.com/86657062/224365256-93567454-55c1-4e46-8ef4-0e3092c96124.png)

- σf = 1, σ = 0.1, ℓ = 3.0

![Figure_3](https://user-images.githubusercontent.com/86657062/224365357-5fea2e9c-59ba-44fd-abf9-2f28b1b8874d.png)
