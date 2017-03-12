"""
Code for training a Factorization Machine using Alternating Newton Method

This is a straight numpy translation of this paper:
Wei-Sheng Chin, Bo-Wen Yuan, Meng-Yuan Yang, and Chih-Jen Lin, 
An Efficient Alternating Newton Method for Learning Factorization Machines, 
Technical Report, 2016.

"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import expit


class FactorizationMachineClassifier(object):

    def __init__(self,
                 d=4,
                 lambda_w=0.0625,
                 lambda_U=0.0625,
                 lambda_V=0.0625,
                 epsilon=0.01,
                 do_pcond=True,
                 sub_rate=0.1,
                 max_iter=1,
                 max_cg_iter=100,
                 block_epsilon = 0.8,
                 nu=0.1,
                 max_nt_iter=100,
                 min_step_size=1e-20,
                 random_seed=None,
                 zeta=0.3,
                 fit_linear=True,
                 verbose=False
                 ):
        self.d = d
        self.lambda_w = lambda_w
        self.lambda_U = lambda_U
        self.lambda_V = lambda_V
        self.epsilon = epsilon
        self.do_pcond = do_pcond
        self.sub_rate = sub_rate
        self.max_iter = max_iter
        self.max_cg_iter = max_cg_iter
        self.block_epsilon = block_epsilon
        self.nu = nu
        self.max_nt_iter = max_nt_iter
        self.min_step_size = min_step_size
        self.random_seed = random_seed
        self.zeta = zeta
        self.fit_linear = fit_linear
        self.verbose = verbose

        self.warm_start = False
        self.total_iters = None

    @property
    def P(self):
        if hasattr(self, '_P'):
            return self._P
        else:
            raise AttributeError('P has not been calculated yet. '
                                 'Run calc_P() first.')

    def _diagonalize_latent_vectors(self):

        def _M(x):
            # Backwards from blondel paper 
            # because U ~ (num_latent_factors, num_features))
            return 0.5 * ( np.dot(self.U.T, np.dot(self.V, x))
                         + np.dot(self.V.T, np.dot(self.U, x) ) )

        M = LinearOperator((self.U.shape[1], self.V.shape[1]),
                            matvec=_M, dtype=self.U.dtype)
        w, v = eigsh(M, k=self.U.shape[0] * 2) # k = 2*rank
        return w, v

    def calc_P(self):
        w, v = self._diagonalize_latent_vectors()
        self._P = np.multiply(w, v).T

    @staticmethod
    def expclip(x):
        return np.exp(np.clip(x, a_min=-18, a_max=18))

    def fit(self, X, y):
        self.initialize_model(X, y)
        self.fit_partial(X, y)

    def initialize_model(self, X, y):
        self.l, self.n = X.shape
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        if self.fit_linear:
            self.w = np.zeros((1, self.n))
        self.U = 2 * (0.1 / np.sqrt(self.d)) * (np.random.random((self.d, self.n)) - 0.5)
        self.V = 2 * (0.1 / np.sqrt(self.d)) * (np.random.random((self.d, self.n)) - 0.5)

        self.y_tilde = self.predict(X)
        self.expyy = self.expclip(np.multiply(y, self.y_tilde))
        self.loss = np.sum(np.log1p(1.0 / self.expyy))
        self.f = (0.5 * (self.lambda_U * np.sum(np.multiply(self.U, self.U))
                         + self.lambda_V * np.sum(np.multiply(self.V, self.V)))
                  + self.loss)
        if self.fit_linear:
            self.f += 0.5 * self.lambda_w * np.sum(np.multiply(self.w, self.w))
        self.precomp_indices = np.arange(0, self.l)
        self.G_norm_0 = 0

        self.warm_start = True
        self.total_iters = 0

    def fit_partial(self, X, y):
        self.t0 = time.time()
        if self.verbose:
            print('iter        time              obj          |grad|           |gradw| (#nt,#cg)           |gradU| (#nt,#cg)           |gradV| (#nt,#cg)')
        for k in range(1, self.max_iter + 1):
            done = self._fit(X, y)
            if k == self.max_iter:
                print('Warning: reach max training iteration. Terminate training process.')
            if done:
                break

    def _fit(self, X, y):
        if self.fit_linear:
            nt_iters_w, G_norm_w, cg_iters_w = self.update_block(y, X, self.w,2*np.ones((1, self.l)), self.lambda_w)
        else:
            nt_iters_w = -1
            G_norm_w = -1
            cg_iters_w = -1
        nt_iters_U, G_norm_U, cg_iters_U = self.update_block(y, X, self.U, (X.dot(self.V.T)).T, self.lambda_U)
        nt_iters_V, G_norm_V, cg_iters_V = self.update_block(y, X, self.V, (X.dot(self.U.T)).T, self.lambda_V)

        norms = [G_norm_U, G_norm_V]
        if self.fit_linear:
            norms.append(G_norm_w)

        self.G_norm = np.linalg.norm(norms)
        self.total_iters += 1
        if self.total_iters == 1:
            self.G_norm_0 = self.G_norm
        if self.G_norm <= self.epsilon * self.G_norm_0:
            return True
        self.t1 = time.time()
        toc = self.t1 - self.t0
        if self.verbose:
            print('%4d  %11.3f  %14.6f  %14.6f    %14.6f (%3d,%3d)    %14.6f (%3d,%3d)    %14.6f (%3d,%3d)' % (
                self.total_iters, toc, self.f, self.G_norm, G_norm_w, nt_iters_w, cg_iters_w,
                G_norm_U, nt_iters_U, cg_iters_U, G_norm_V, nt_iters_V, cg_iters_V
                )
            )
        return False

    def predict(self, X):
        pred = 0
        if self.fit_linear:
            pred += self.predict_linear(X)
        pred += 0.5 *np.sum(np.multiply((X.dot(self.U.T)).T, (X.dot(self.V.T)).T), axis=0, keepdims=True).T
        return pred

    def predict_linear(self, X):
        return X.dot(self.w.T)

    def predict_proba(self, X):
        return expit(self.predict(X))

    def update_block(self, y, X, U, Q, lambda_):
        G0_norm = 0
        total_cg_iters = 0
        nt_iters = 0
        all_done = False
        for k in range(1, self.max_nt_iter + 1):
            spmat = sp.coo_matrix((np.divide(-y, (1 + self.expyy)).ravel(), (self.precomp_indices, self.precomp_indices))).tocsr()
            G = lambda_ * U + 0.5 * (X.T.dot(spmat.T).dot(Q.T)).T
            G_norm = np.sqrt(np.sum(np.multiply(G, G)))
            if k == 1:
                G0_norm = G_norm
            if G_norm <= self.block_epsilon * G0_norm:
                break
            nt_iters = k
            if k == self.max_nt_iter:
                print('Warning: reach newton iteration bound before gradient norm is shrinked enough.')
            D = sp.coo_matrix((
                    np.divide(np.divide(self.expyy, (1 + self.expyy)), (1 + self.expyy)).ravel(),
                    (self.precomp_indices, self.precomp_indices)
                )).tocsr()
            S, cg_iters = self.pcg(X, Q, G, D, lambda_)
            total_cg_iters = total_cg_iters + cg_iters
            Delta = 0.5 * (np.sum(np.multiply(Q.T, X.dot(S.T)), axis=1, keepdims=True))
            US = np.sum(np.multiply(U, S))
            SS = np.sum(np.multiply(S, S))
            GS = np.sum(np.multiply(G, S))
            theta = 1
            while True:
                if theta < self.min_step_size:
                    print('Warning: step size is too small in line search. Switch to the next block of variables.')
                    all_done = True
                    break
                y_tilde_new = self.y_tilde + theta * Delta
                # I _think_ that we could replace loss_new with
                # loss_new = np.sum(np.logaddexp(0, -np.multiply(y, y_tilde_new)))
                # However, what to do with self.expyy that uses expyy_new?
                # Alternatively, maybe we clip expyy?
                expyy_new = self.expclip(np.multiply(y, y_tilde_new))
                loss_new = np.sum(np.log1p(np.divide(1.0, expyy_new)))

                f_diff = 0.5 * lambda_ * (2 * theta * US + theta * theta * SS) + loss_new - self.loss
                if f_diff <= self.nu*theta*GS:
                    self.loss = loss_new
                    self.f += f_diff
                    U += theta * S
                    self.y_tilde = y_tilde_new
                    self.expyy = expyy_new
                    break
                theta *= 0.5

            if all_done:
                break

        return (nt_iters, G_norm, total_cg_iters)

    def pcg(self, X, Q, G, D, lambda_):
        cg_max_iter = 100
        if self.sub_rate < 1:
            l = X.shape[0]
            whole = np.random.permutation(l)
            selected = np.sort(whole[np.arange(0, np.max((1, np.int(np.floor(self.sub_rate * l)))))])
            X = X[selected, :]
            Q = Q[:, selected]
            D = sp.coo_matrix((
                    D.data[selected],
                    (np.arange(len(selected)), np.arange(len(selected)))
                )).tocsr()
        l = X.shape[0]
        s_bar = np.zeros(G.shape)
        M = np.ones(G.shape)
        if self.do_pcond:
            M = np.divide(1, np.sqrt(lambda_ + (1/self.sub_rate) * 0.25 * (D.dot(X.power(2)).T.dot(np.multiply(Q, Q).T).T)))
        r = np.multiply(-M, G)
        d = r
        G0G0 = np.sum(np.multiply(r, r))
        gamma = G0G0
        cg_iters = 0
        precomp_indices = np.arange(0, l)
        while gamma > self.zeta*self.zeta*G0G0:
            cg_iters += 1
            Dh = np.multiply(M, d)
            z = 0.5 * np.sum(np.multiply(Q.T, X.dot(Dh.T)), axis=1, keepdims=True)
            spmat = sp.coo_matrix((D.dot(z).ravel(), (precomp_indices, precomp_indices))).tocsr()
            Dh = np.multiply(M, lambda_ * Dh + 0.5 * (1/self.sub_rate) * (X.T.dot(spmat.T).dot(Q.T)).T)
            alpha = gamma / np.sum(np.multiply(d, Dh))
            s_bar += alpha * d
            r -= alpha * Dh
            gamma_new = np.sum(np.multiply(r, r))
            beta = gamma_new / gamma
            d = r + beta * d
            gamma = gamma_new
            if cg_iters >= self.max_cg_iter:
                print('Warning: reach max CG iteration. CG process is terminaated.')
                break
        S = np.multiply(M, s_bar)

        return S, cg_iters
