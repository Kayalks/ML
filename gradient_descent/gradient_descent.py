import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

def gradient_descent(x,y):
    m_curr = b_curr=0
    iterations = 1000000
    n=len(x)
    learning_rate = 0.0001
    costprev = 0
    for i in range(iterations):
        y_predicted = m_curr*x+b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum((y-y_predicted))
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, costprev, rel_tol=1e-20):
            break
        costprev = cost
        print("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost,i))

    return m_curr,b_curr

def train_linear_reg(x,y):
    lr = LinearRegression()
    lr.fit(x,y)
    return lr.coef_,lr.intercept_


if __name__ == '__main__':

    #x=np.array([1,2,3,4,5])
    #y=np.array([5,7,9,11,13])
    df = pd.read_csv('test_scores.csv')

    x=np.array(df['math'])
    y=np.array(df['cs'])
    print('GD: m {}, b {}',format(gradient_descent(x,y)))

    print('LR: m {}, b {}', format(train_linear_reg(df[['math']],df.cs)))