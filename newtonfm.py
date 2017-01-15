"""
Code for training a Factorization Machine using Alternating Newton Method

This is a straight numpy translation of this paper:
Wei-Sheng Chin, Bo-Wen Yuan, Meng-Yuan Yang, and Chih-Jen Lin, 
An Efficient Alternating Newton Method for Learning Factorization Machines, 
Technical Report, 2016.

"""

import pickle
import time

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score


def example():
    train_path = './matlab_code/fourclass_scale.tr'
    test_path = './matlab_code/fourclass_scale.te'

    X_train, X_test, y_train, y_test = load_example_data(train_path, test_path)
    lambda_w, lambda_U, lambda_V, d, epsilon, do_pcond, sub_rate = load_example_params()

    w, U, V = fm_train(y_train, X_train, lambda_w, lambda_U, lambda_V, d, epsilon, do_pcond, sub_rate)

    y_tilde = fm_predict(X_test, w, U, V)
    print('test accuracy: {}'.format(accuracy_score(y_test>0, y_tilde>0)))


def load_example_data(train_path, test_path):

    X, y = load_svmlight_file(train_path)
    X_test, y_test = load_svmlight_file(test_path)
    y = np.expand_dims(y, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    n = np.max((X.shape[1], X_test.shape[1]))
    X = X.tocsr()
    X_test = X_test.tocsr()

    return X, X_test, y, y_test


def load_example_params():

    lambda_w = 0.0625
    lambda_U = 0.0625
    lambda_V = 0.0625
    d = 4
    epsilon = 0.01
    do_pcond = True
    sub_rate = 0.1
    return lambda_w, lambda_U, lambda_V, d, epsilon, do_pcond, sub_rate


def fm_predict(X, w, U, V):
    y_tilde = X.dot(w.T) + 0.5 *np.sum(np.multiply((X.dot(U.T)).T, (X.dot(V.T)).T), axis=0, keepdims=True).T
    return y_tilde


def fm_train(y, X, lambda_w, lambda_U, lambda_V, d, epsilon, do_pcond, sub_rate):
    """
    Inputs:
      y: training labels, an l-dimensional binary vector. Each element should be either +1 or -1.
      X: training instances. X is an l-by-n matrix if you have l training instances in an n-dimensional feature space.
      lambda_w: the regularization coefficient of linear term.
      lambda_U, lambda_V: the regularization coefficients of the two interaction matrices.
      d: dimension of the latent space.
      epsilon: stopping tolerance in (0,1). Use a larger value if the training time is too long.
      do_pcond: a flag. Use 1/0 to enable/disable the diagonal preconditioner.
      sub_rate: sampling rate in (0,1] to select instances for the sub-sampled Hessian matrix.
    Outputs:
      w: linear coefficients. An n-dimensional vector.
      U, V: the interaction (d-by-n) matrices.
    """
    t0 = time.time()

    max_iter = 1000
    l, n = X.shape
    w = np.zeros((1, n))
    np.random.seed(0)
    # Random initialize
    U = 2 * (0.1 / np.sqrt(d)) * (np.random.random((d, n)) - 0.5)
    V = 2 * (0.1 / np.sqrt(d)) * (np.random.random((d, n)) - 0.5)
    y_tilde = X.dot(w.T) + 0.5 * np.expand_dims(np.sum(np.multiply((X.dot(U.T)).T, (X.dot(V.T)).T), axis=0), axis=0).T
    expyy = np.exp(np.multiply(y, y_tilde))
    loss = np.sum(np.log1p(1.0 / expyy))
    f = (0.5 * (lambda_w * np.sum(np.multiply(w, w))
               + lambda_U * np.sum(np.multiply(U, U))
               + lambda_V * np.sum(np.multiply(V, V)))
        + loss)

    precomp_indices = np.arange(0, l)

    G_norm_0 = 0
    print('iter        time              obj          |grad|           |gradw| (#nt,#cg)           |gradU| (#nt,#cg)           |gradV| (#nt,#cg)')
    for k in range(1, max_iter + 1):
        w, y_tilde, expyy, f, loss, nt_iters_w, G_norm_w, cg_iters_w = update_block(y, X, w, 2*np.ones((1,l)), y_tilde, expyy, f, loss, lambda_w, do_pcond, sub_rate, precomp_indices)
        U, y_tilde, expyy, f, loss, nt_iters_U, G_norm_U, cg_iters_U = update_block(y, X, U, (X.dot(V.T)).T, y_tilde, expyy, f, loss, lambda_U, do_pcond, sub_rate, precomp_indices)
        V, y_tilde, expyy, f, loss, nt_iters_V, G_norm_V, cg_iters_V = update_block(y, X, V, (X.dot(U.T)).T, y_tilde, expyy, f, loss, lambda_V, do_pcond, sub_rate, precomp_indices)
        G_norm = np.linalg.norm([G_norm_w, G_norm_U, G_norm_V])
        if k == 1:
            G_norm_0 = G_norm
        if G_norm <= epsilon * G_norm_0:
            break
        t1 = time.time()
        toc = t1 - t0
        print('%4d  %11.3f  %14.6f  %14.6f    %14.6f (%3d,%3d)    %14.6f (%3d,%3d)    %14.6f (%3d,%3d)' % (k, toc, f, G_norm, G_norm_w, nt_iters_w, cg_iters_w, G_norm_U, nt_iters_U, cg_iters_U, G_norm_V, nt_iters_V, cg_iters_V))
        if k == max_iter:
            print('Warning: reach max training iteration. Terminate training process.')


    return (w, U, V)


def update_block(y, X, U, Q, y_tilde, expyy, f, loss, lambda_, do_pcond, sub_rate, precomp_indices):
    epsilon = 0.8
    nu = 0.1
    max_nt_iter = 100
    min_step_size = 1e-20
    l = X.shape[0]
    G0_norm = 0
    total_cg_iters = 0
    nt_iters = 0
    all_done = False
    for k in range(1, max_nt_iter + 1):
        spmat = sp.coo_matrix((np.divide(-y, (1 + expyy)).ravel(), (precomp_indices, precomp_indices))).tocsr()
        G = lambda_ * U + 0.5 * (X.T.dot(spmat.T).dot(Q.T)).T
        G_norm = np.sqrt(np.sum(np.multiply(G, G)))
        if k == 1:
            G0_norm = G_norm
        if G_norm <= epsilon * G0_norm:
            break
        nt_iters = k
        if k == max_nt_iter:
            print('Warning: reach newton iteration bound before gradient norm is shrinked enough.')
        D = sp.coo_matrix((np.divide(np.divide(expyy, (1 + expyy)), (1 + expyy)).ravel(), (precomp_indices, precomp_indices))).tocsr()
        S, cg_iters = pcg(X, Q, G, D, lambda_, do_pcond, sub_rate)
        total_cg_iters = total_cg_iters + cg_iters
        Delta = 0.5 * (np.sum(np.multiply(Q.T, X.dot(S.T)), axis=1, keepdims=True))
        US = np.sum(np.multiply(U, S))
        SS = np.sum(np.multiply(S, S))
        GS = np.sum(np.multiply(G, S))
        theta = 1
        while True:
            if theta < min_step_size:
                print('Warning: step size is too small in line search. Switch to the next block of variables.')
                all_done = True
                break
            y_tilde_new = y_tilde + theta * Delta
            expyy_new = np.exp(np.multiply(y, y_tilde_new))
            loss_new = np.sum(np.log1p(np.divide(1.0, expyy_new)))
            f_diff = 0.5 * lambda_ * (2 * theta * US + theta * theta * SS) + loss_new - loss
            if f_diff <= nu*theta*GS:
                loss = loss_new
                f += f_diff
                U += theta * S
                y_tilde = y_tilde_new
                expyy = expyy_new
                break
            theta *= 0.5

        if all_done:
            break

    return (U, y_tilde, expyy, f, loss, nt_iters, G_norm, total_cg_iters)


def pcg(X, Q, G, D, lambda_, do_pcond, sub_rate):
    zeta = 0.3
    cg_max_iter = 100
    if sub_rate < 1:
        l = X.shape[0]
        whole = np.random.permutation(l)
        selected = np.sort(whole[np.arange(0, np.max((1, np.int(np.floor(sub_rate * l)))))])
        X = X[selected, :]
        Q = Q[:, selected]
        D = sp.coo_matrix((D.data[selected], (np.arange(len(selected)), np.arange(len(selected))))).tocsr()
    l = X.shape[0]
    s_bar = np.zeros(G.shape)
    M = np.ones(G.shape)
    if do_pcond:
        M = np.divide(1, np.sqrt(lambda_ + (1/sub_rate) * 0.25 * (D.dot(X.power(2)).T.dot(np.multiply(Q, Q).T).T)))
    r = np.multiply(-M, G)
    d = r
    G0G0 = np.sum(np.multiply(r, r))
    gamma = G0G0
    cg_iters = 0
    precomp_indices = np.arange(0, l)
    while gamma > zeta*zeta*G0G0:
        cg_iters += 1
        Dh = np.multiply(M, d)
        z = 0.5 * np.sum(np.multiply(Q.T, X.dot(Dh.T)), axis=1, keepdims=True)
        spmat = sp.coo_matrix((D*z.ravel(), (precomp_indices, precomp_indices))).tocsr()
        Dh = np.multiply(M, lambda_ * Dh + 0.5 * (1/sub_rate) * (X.T.dot(spmat.T).dot(Q.T)).T)
        alpha = gamma / np.sum(np.multiply(d, Dh))
        s_bar += alpha * d
        r -= alpha * Dh
        gamma_new = np.sum(np.multiply(r, r))
        beta = gamma_new / gamma
        d = r + beta * d
        gamma = gamma_new
        if cg_iters >= cg_max_iter:
            print('Warning: reach max CG iteration. CG process is terminaated.')
            break
    S = np.multiply(M, s_bar)

    return S, cg_iters

if __name__ == '__main__':
    example()