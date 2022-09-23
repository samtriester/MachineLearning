# Homework 0 Code
import numpy as np
import matplotlib.pyplot as plt



def perceptron_learn(data_in,d,N):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    #         d: dimensionality of each data point
    #         N: number of points (It was easier to pass in size of matrix)
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:
    ys=data_in[:,d+1]
    xs=np.delete(data_in,d+1,1)

    w=np.zeros((d+1,))

    iterations=0
    while(True):
        done=True
        iterations+=1

        for i in range(0, N - 1):
            if np.matmul(np.transpose(w),xs[i,:]) * ys[i]<=0:
                done=False

                changer=ys[i]*xs[i,:]
                w=w+changer
                break


        if(done):
            return w, iterations


def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW0
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))
    for i in range(0, num_exp):
        weight_vector=np.random.rand(d+1,1)
        weight_vector[0,0]=0
        one_matrix=np.full((N,d),1)
        one_col=np.full((N,1),1)
        data_matrix=np.random.rand(N,d)
        data_matrix=data_matrix*2
        data_matrix=np.subtract(data_matrix, one_matrix)
        data_matrix=np.append(one_col,data_matrix, axis = 1)
        y_numbers=np.matmul(data_matrix, weight_vector)
        y_numbers=np.sign(y_numbers)
        just_data=data_matrix
        data_matrix=np.append(data_matrix,y_numbers, axis = 1)

    # Initialize the return variables


        w,iterations=perceptron_learn(data_matrix, d, N)
        p = min(abs(np.dot(w,np.transpose(just_data))));
        r = np.linalg.norm(np.linalg.norm(just_data, np.Inf));
        n=np.linalg.norm(w)
        bound=(r*r * n*n/( p*p));
        num_iters[i]= iterations
        bounds_minus_ni[i]=np.log(bound-num_iters[i])
    # Your code here, assign the values to num_iters and bounds_minus_ni:

    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()
