import numpy as np
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
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
      

X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")
X = lift(X)

print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))


# alpha = 0.1
#
# model = Lasso(alpha = alpha)
#
# cv = KFold(
#         n_splits=5,
#         random_state=42,
#         shuffle=True
#         )
#
#
#
# scores = cross_val_score(
#         model, X_train, y_train, cv=cv,scoring="neg_root_mean_squared_error")
#
#
# print("Cross-validation RMSE for α=%f : %f ± %f" % (alpha,-np.mean(scores),np.std(scores)) )

# === New code: scan alpha values and pick the best one ===

alphas = 2.0 ** np.arange(-10, 11)  # from 2^-10 to 2^10
cv = KFold(n_splits=5, random_state=42, shuffle=True)

mean_rmse = []
std_rmse = []

print("Performing 5-fold CV for multiple α values...")

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )

    mean_rmse.append(-np.mean(scores))
    std_rmse.append(np.std(scores))
    print("α = %8.5f -> CV RMSE = %.6f ± %.6f" % (alpha, -np.mean(scores), np.std(scores)))

# Find best alpha (lowest mean RMSE)
best_idx = np.argmin(mean_rmse)
best_alpha = alphas[best_idx]

print("\nBest α found: %f (CV RMSE = %.6f)" % (best_alpha, mean_rmse[best_idx]))

model = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
print("Fitting linear model over entire training set...",end="")
model.fit(X_train, y_train)
print(" done")


# Compute RMSE
rmse_train = rmse(y_train,model.predict(X_train))
rmse_test = rmse(y_test,model.predict(X_test))

print("Train RMSE = %f, Test RMSE = %f" % (rmse_train,rmse_test))



# Plot CV mean RMSE as a function of alpha with error bars
plt.figure(figsize=(8, 5))
plt.errorbar(alphas, mean_rmse, yerr=std_rmse, fmt='-o', capsize=4, label='CV RMSE')
plt.xlabel('Alpha')
plt.ylabel('Mean Cross-Validation RMSE')
plt.title('RMSE vs Alpha')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.show()



print("Model parameters:")
print("\t Intercept: %3.5f" % model.intercept_,end="")
for i,val in enumerate(model.coef_):
    print(", β%d: %3.5f" % (i,val), end="")
print("\n")


