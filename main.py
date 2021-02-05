import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import math
import pandas as pd

np.random.seed(1748)
#-------------------------------------------------------------------------------
#--------------------------------Initializing Data------------------------------
N1, N2, N3 = 120, 90, 90
N = N1 + N2 + N3

sample_mean1 = np.array([0.0, 2.5])
sample_mean2 = np.array([-2.5, -2.0])
sample_mean3 = np.array([2.5, -2.0])

cov_matrix1 = np.array([[3.5, 0.0],[0.0, 1.2]])
cov_matrix2 = np.array([[1.2, -0.8],[-0.8, 1.2]])
cov_matrix3 = np.array([[1.2, 0.8],[0.8, 1.2]])

data_points1 = np.random.multivariate_normal(sample_mean1, cov_matrix1, N1)
data_points2 = np.random.multivariate_normal(sample_mean2, cov_matrix2, N2)
data_points3 = np.random.multivariate_normal(sample_mean3, cov_matrix3, N3)

data_points = np.concatenate((data_points1, data_points2, data_points3))
#-------------------------------------------------------------------------------
#----------------------------------Helper Functions-----------------------------
def class_prios_estimator(sample_size, total_sample_size):
    return sample_size/total_sample_size

def sample_mean_estimator(column1, column2, sample_size):
    component1 = sum(column1 / sample_size)
    component2 = sum(column2 / sample_size)
    return np.array([component1, component2])

def sample_covariance_estimator(sample_mean, sample_data_points = 0, sample_size = 0):
    matrix = np.zeros((2, 2))

    for i in range(len(sample_data_points)):
        component1 = sample_data_points[i][0] - sample_mean[0]
        component2 = sample_data_points[i][1] - sample_mean[1]
        
        vector = np.array([[component1, component2]])
        matrix += np.dot(vector.T, vector)
    
    return matrix / sample_size
#-------------------------------------------------------------------------------
#----------------------------------Sample Mean Estimation-----------------------
sample_mean1_estimation = sample_mean_estimator(data_points1[:,0], data_points1[:,1], N1)
sample_mean2_estimation = sample_mean_estimator(data_points2[:,0], data_points2[:,1], N2)
sample_mean3_estimation = sample_mean_estimator(data_points3[:,0], data_points3[:,1], N3)

print("The mean estimation 1: {} \nThe mean estimation 2: {} \nThe mean estimation 3: {}".format(sample_mean1_estimation, sample_mean2_estimation, sample_mean3_estimation))
#-------------------------------------------------------------------------------
#----------------------------------Sample Covariance Estimation-----------------
sample_covariance1 = sample_covariance_estimator(sample_mean1, data_points1, N1)
sample_covariance2 = sample_covariance_estimator(sample_mean2, data_points2, N2)
sample_covariance3 = sample_covariance_estimator(sample_mean3, data_points3, N3)

print("The covariance estimation 1: \n{} \nThe covariance estimation 2: \n{} \nThe covariance estimation 3: \n{}".format(sample_covariance1, sample_covariance2, sample_covariance3))
#-------------------------------------------------------------------------------
#----------------------------------Class Priors Estimation----------------------
class1_prior = class_prios_estimator(N1, N)
class2_prior = class_prios_estimator(N2, N)
class3_prior = class_prios_estimator(N3, N)

print("Class 1 prior: {} \nClass 2 prior: {} \nClass 3 prior: {}".format(class1_prior, class2_prior, class3_prior))
#-------------------------------------------------------------------------------
#----------------------------------Parameter Functions--------------------------
def W_c(covariance_matrix):
    return - 1/2 * linalg.inv(cov_matrix1)

def w_c(covariance_matrix, sample_mean):
    return np.dot(linalg.inv(covariance_matrix), sample_mean)

def w_c0(covariance_matrix, sample_mean, prior):
    cov_det = linalg.det(covariance_matrix)
    cov_inc = linalg.inv(covariance_matrix)
    return - 1/2 * np.dot(sample_mean.T, np.dot(cov_inc, sample_mean)) - 1/2 * math.log(cov_det) + math.log(prior)

def score_function(covariance_matrix, sample_mean, prior, input_vector):
    first_term = W_c(covariance_matrix)
    second_term = w_c(covariance_matrix, sample_mean).T
    third_term = w_c0(covariance_matrix, sample_mean, prior)
    return np.dot(input_vector.T, np.dot(first_term, input_vector)) + np.dot(second_term, input_vector) + third_term
#-------------------------------------------------------------------------------
#----------------------------------Estimated Points Plotting--------------------
predicted_values = []
true_values = np.concatenate([np.repeat(0,120), np.repeat(1,90), np.repeat(2,90)])

def predict_data(sample_size, vector_arr):

    for i in range(sample_size):
        data1_classification = score_function(cov_matrix1, sample_mean1, class1_prior, vector_arr[i])
        data2_classification = score_function(cov_matrix2, sample_mean2, class2_prior, vector_arr[i])
        data3_classification = score_function(cov_matrix3, sample_mean3, class3_prior, vector_arr[i])

        arr = [data1_classification, data2_classification, data3_classification]

        predicted_values.append(np.argmax(arr))
    
predict_data(len(data_points1), data_points1)
predict_data(len(data_points2), data_points2)
predict_data(len(data_points3), data_points3)
#-------------------------------------------------------------------------------
#----------------------------------Classes Discrimination-----------------------
# This part seperates the classes using a grid made with points.
# NOTE: THIS PART IS NOT OPTIMIZED; IF THIS PART CAUSED ANY PERFOMACE ISSUES DELETE IT OR LOWER THE VALUE OF points_num.

points_num = 100

def pred(vector_arr):
    data1_classification = score_function(cov_matrix1, sample_mean1, class1_prior, vector_arr)
    data2_classification = score_function(cov_matrix2, sample_mean2, class2_prior, vector_arr)
    data3_classification = score_function(cov_matrix3, sample_mean3, class3_prior, vector_arr)

    arr = [data1_classification, data2_classification, data3_classification]
    maxim = max(arr)
    if maxim == data1_classification:
        return '1'
    elif maxim == data2_classification:
        return '2'
    else:
        return '3'

x = np.linspace(-6, 6, points_num)
y = np.linspace(-6, 6, points_num)

xx, yy = np.meshgrid(x, y)

def fill_data(xx, yy):
    data = []
    for i in range(points_num):
        for j in range(points_num):
            data.append(([xx[i][j], yy[i][j]]))
    return data
        
plt.figure(figsize = (10, 10))

data = np.array(fill_data(xx, yy))

for i in range(len(data)):
    b = pred(data[i])
    if b == '1':
        plt.plot(data[i][0],data[i][1], "m.", markersize = 10, alpha = 0.3)
    elif b == '2':
        plt.plot(data[i][0],data[i][1], "y.", markersize = 10, alpha = 0.3)
    else: 
        plt.plot(data[i][0],data[i][1], "c.", markersize = 10, alpha = 0.3)
#-------------------------------------------------------------------------------
#---------------------------------Confusion Matrix------------------------------
confusion_matrix = pd.crosstab([predicted_values], [true_values], rownames = ['y_pred'], colnames = ['y_truth'])

print(confusion_matrix)
#-------------------------------------------------------------------------------
#---------------------------------Plotting Data---------------------------------
plt.plot(data_points[predicted_values != true_values, 0], data_points[predicted_values != true_values, 1], "ko", markersize = 15, fillstyle = "none")

plt.plot(data_points1[:,0], data_points1[:,1], "r.", markersize = 10)
plt.plot(data_points2[:,0], data_points2[:,1], "g.", markersize = 10)
plt.plot(data_points3[:,0], data_points3[:,1], "b.", markersize = 10)

plt.xlabel("x1")
plt.ylabel("x2")

plt.show()