import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from data_generator import postfix



# Number of samples
N = 1000

# Noise variance 
sigma = 0.01


# Feature dimension
d = 5


psfx = postfix(N,d,sigma) 
      

X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")

print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))

model = LinearRegression()

print("Fitting linear model...",end="")
model.fit(X_train, y_train)
print(" done")


# Compute RMSE on train and test sets
rmse_train = rmse(y_train,model.predict(X_train))
rmse_test = rmse(y_test,model.predict(X_test))

print("Train RMSE = %f, Test RMSE = %f" % (rmse_train,rmse_test))


print("Model parameters:")
print("\t Intercept: %3.5f" % model.intercept_,end="")
for i,val in enumerate(model.coef_):
    print(", β%d: %3.5f" % (i,val), end="")
print("\n")    






