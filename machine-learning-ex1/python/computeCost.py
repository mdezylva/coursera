def compute_cost(X,y,theta):
    '''
    Compute cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the   parameter for linear regression to fit the data points in X and y
    '''
    
    # Initialise some useful values 
    m = length(y)

    h_theta = X*theta 
