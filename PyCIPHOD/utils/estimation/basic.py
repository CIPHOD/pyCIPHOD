from sklearn.linear_model import LinearRegression as lr

def causal_effect_estimation(data, X, Y, covariates=[], method="linear"):
    if method == "linear":
        list_nodes = [X] + covariates
        X_data = data[list_nodes].values
        Y_data = data[Y].values
        reg = lr().fit(X_data, Y_data)

        return reg.coef_[0]
    else:
        print("estimation method not supported")
        exit()