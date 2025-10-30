import numpy as np
from mpmath.matrices.eigen_symmetric import c_he_tridiag_0
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
d = 40


psfx = postfix(N,d,sigma)

# function lift = liftDataset
def lift(x_initial):
    x_prime = []

    xi = len(x_initial)
    for row in range(xi):
        x_expand = []
        x_0 = x_initial[row]

        l = len(x_0)
        for i in range(l):
            x_expand.append(x_0[i])
            for j in range(i,l):
                x_expand.append(x_0[j]*x_0[i])

        x_prime.append(x_expand)

    return np.array(x_prime)
# test lift function
# x_initials = [[1,2,3,4,5],
#              [2,4,6,8,10]]
# new_x_i =lift(x_initials)
# print(new_x_i)

      

X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")



print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])
frac = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rmse_train_array = []
rmse_test_array = []
training_length_array = []
X = lift(X)

for i, fr in enumerate(frac):
    # ... use both i and frac
    print(f"Training with {fr*100}% of the training data")
    #Split data: test set made up of fraction of N data (ex 0.30*1000 = 300)
    testSize = 0.30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testSize, random_state=42)

    print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))
    print("Training with number of n samples = %d" % int(len(X_train) * fr))

    ### select only a fraction (ex:10%) of the test data (X_train y_train)
    #fr = 0.1 # fraction of test set (ex 10%)
    # Calculate the number of elements to select (ex:10% of 1000)
    num_elements_to_select = int(len(X_train) * fr)

    # Generate random indices
    random_indices = np.random.choice(len(X_train), num_elements_to_select, replace=False)

    #print("Original X array:", X_train)
    #print("Original y array:", y_train)

    # Select the elements using the random indices
    X_train = X_train[random_indices]
    y_train = y_train[random_indices]

    #print(f"Selected {fr} of X data:", X_train)
    #print(f"Selected {fr} of y data:", y_train)

    ###


    model = LinearRegression()

    print("Fitting linear model...",end="")
    model.fit(X_train, y_train)
    print(" done")


    # Compute RMSE on train and test sets
    rmse_train = rmse(y_train,model.predict(X_train))
    rmse_test = rmse(y_test,model.predict(X_test))
    rmse_train_array.append(rmse_train)
    rmse_test_array.append(rmse_test)
    training_length_array.append(len(y_train))

    print("Train RMSE = %f, Test RMSE = %f" % (rmse_train,rmse_test))


    print("Model parameters:")
    print("\t Intercept: %3.5f" % model.intercept_,end="")
    for i,val in enumerate(model.coef_):
        print(", β%d: %3.5f" % (i,val), end="")
    print("\n")
    #print(f'x1',X_test)

#Plot the train and test RMSE as a function of the number of training samples given to model

plt.figure(figsize=(8, 6))
plt.plot(training_length_array, rmse_train_array, marker='o', label='Train RMSE')
#plt.plot(training_length_array, rmse_test_array, marker='s', label='Test RMSE')
plt.xlabel("Training Set Size: number of training samples")
plt.ylabel("RMSE")
plt.title(f"Training and Test RMSE vs Training Set Size (Data from {psfx})")
plt.legend()
plt.grid(True)
plt.show()



