import numpy as np

# product of two probabilities
def prod_p(p1, p2):
    return p1 * p2

# geometric mean of two probabilities
def geo_mean(p1, p2):
    return np.sqrt(p1 * p2)

# Joint likelihood function
def joint_l(p1, p2, y1, y2):
    return p1**y1 * (1 - p1)**(1 - y1) * p2**y2 * (1 - p2)**(1 - y2)

# Likelihood 1
def bce_likelihood(p1, p2, y1, y2):
    return joint_l(p1, p2, y1, y2)


# Function for p_a^t (with lambda)
def any_class_likelihood(p1, p2, y1, y2, lambda_val):
    # Define weights based on y1, y2
    if y1 == y2:
        w1 = 1
        w2 = 1
    else:
        w1 = 1 if y1 == 1 else lambda_val
        w2 = 1 if y2 == 1 else lambda_val
    M = 2  # As we have two probabilities (p1, p2)
    
    # Calculate the product terms
    prod_p = np.prod([p1**w1, p2**w2])
    prod_1_minus_p = np.prod([(1 - p1)**w1, (1 - p2)**w2])
    
    # Compute the terms for y_a = 1 and y_a = 0
    if (y1 == 1 or y2 == 1):  # If y_a = 1 (any y_j is 1)
        term1 = np.power(prod_p, 1 / (w1 + w2))
        term2 = np.power(prod_1_minus_p, 1 / (w1 + w2))
        return term1 / (term1 + term2)
    else:  # If y_a = 0 (no y_j is 1)
        term1 = np.power(prod_1_minus_p, 1 / (w1 + w2))
        term2 = np.power(prod_p, 1 / (w1 + w2))
        return term1 / (term1 + term2)

# Function for p_a^t (with lambda)
def any_class_likelihood_prod(p1, p2, y1, y2, lambda_val):
    # Define weights based on y1, y2
    if y1 == y2:
        w1 = 1
        w2 = 1
    else:
        w1 = 1 if y1 == 1 else lambda_val
        w2 = 1 if y2 == 1 else lambda_val
    M = 2  # As we have two probabilities (p1, p2)
    
    # Calculate the product terms
    prod_p = np.prod([p1**w1, p2**w2])
    prod_1_minus_p = np.prod([(1 - p1)**w1, (1 - p2)**w2])
    
    # Compute the terms for y_a = 1 and y_a = 0
    if (y1 == 1 or y2 == 1):  # If y_a = 1 (any y_j is 1)
        term1 = prod_p
        term2 = prod_1_minus_p
        return term1 / (term1 + term2)
    else:  # If y_a = 0 (no y_j is 1)
        term1 = prod_1_minus_p
        term2 = prod_p
        return term1 / (term1 + term2)