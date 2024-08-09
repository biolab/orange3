import numpy as np
import sys
# import warnings
# warnings.filterwarnings("ignore")
from Orange.classification.utils.fasterrisk.utils import normalize_X, compute_logisticLoss_from_ExpyXB

class logRegModel:
    def __init__(self, X, y, lambda2=1e-8, intercept=True, original_lb=-5, original_ub=5):
        self.X = X
        self.X_normalized, self.X_mean, self.X_norm, self.scaled_feature_indices = normalize_X(self.X)
        self.n, self.p = self.X_normalized.shape
        self.y = y.reshape(-1).astype(float)
        self.yX = y.reshape(-1, 1) * self.X_normalized
        self.yXT = np.zeros((self.p, self.n))
        self.yXT[:] = np.transpose(self.yX)[:]
        self.beta0 = 0
        self.betas = np.zeros((self.p, ))
        self.ExpyXB = np.exp(self.y * self.beta0 + self.yX.dot(self.betas))

        self.intercept = intercept
        self.lambda2 = lambda2
        self.twoLambda2 = 2 * self.lambda2

        self.Lipschitz = 0.25 + self.twoLambda2
        self.lbs = original_lb * np.ones(self.p)
        self.lbs[self.scaled_feature_indices] *= self.X_norm[self.scaled_feature_indices]
        self.ubs = original_ub * np.ones(self.p)
        self.ubs[self.scaled_feature_indices] *= self.X_norm[self.scaled_feature_indices]

        self.total_child_added = 0
    
    def warm_start_from_original_beta0_betas(self, original_beta0, original_betas):
        # betas_initial has dimension (p+1, 1)
        self.original_beta0 = original_beta0
        self.original_betas = original_betas
        self.beta0, self.betas = self.transform_coefficients_to_normalized_space(self.original_beta0, self.original_betas)
        print("warmstart solution in normalized space is {} and {}".format(self.beta0, self.betas))
        self.ExpyXB = np.exp(self.y * self.beta0 + self.yX.dot(self.betas))

    def warm_start_from_beta0_betas(self, beta0, betas):
        self.beta0, self.betas = beta0, betas
        self.ExpyXB = np.exp(self.y * self.beta0 + self.yX.dot(self.betas))

    def warm_start_from_beta0_betas_ExpyXB(self, beta0, betas, ExpyXB):
        self.beta0, self.betas, self.ExpyXB = beta0, betas, ExpyXB

    def get_beta0_betas(self):
        return self.beta0, self.betas

    def get_beta0_betas_ExpyXB(self):
        return self.beta0, self.betas, self.ExpyXB
        
    def get_original_beta0_betas(self):
        return self.transform_coefficients_to_original_space(self.beta0, self.betas)

    def transform_coefficients_to_original_space(self, beta0, betas):
        original_betas = betas.copy()
        original_betas[self.scaled_feature_indices] = original_betas[self.scaled_feature_indices]/self.X_norm[self.scaled_feature_indices]
        original_beta0 = beta0 - np.dot(self.X_mean, original_betas)
        return original_beta0, original_betas

    def transform_coefficients_to_normalized_space(self, original_beta0, original_betas):
        betas = original_betas.copy()
        betas[self.scaled_feature_indices] = betas[self.scaled_feature_indices] * self.X_norm[self.scaled_feature_indices]
        beta0 = original_beta0 + self.X_mean.dot(original_betas)
        return beta0, betas

    def get_grad_at_coord(self, ExpyXB, betas_j, yX_j, j):
        # return -np.dot(1/(1+ExpyXB), self.yX[:, j]) + self.twoLambda2 * betas_j
        # return -np.inner(1/(1+ExpyXB), self.yX[:, j]) + self.twoLambda2 * betas_j
        # return -np.inner(np.reciprocal(1+ExpyXB), self.yX[:, j]) + self.twoLambda2 * betas_j
        return -np.inner(np.reciprocal(1+ExpyXB), yX_j) + self.twoLambda2 * betas_j
        # return -yX_j.dot(np.reciprocal(1+ExpyXB)) + self.twoLambda2 * betas_j

    def update_ExpyXB(self, ExpyXB, yX_j, diff_betas_j):
        ExpyXB *= np.exp(yX_j * diff_betas_j)

    def optimize_1step_at_coord(self, ExpyXB, betas, yX_j, j):
        # in-place modification, heck that ExpyXB and betas are passed by reference
        prev_betas_j = betas[j]
        current_betas_j = prev_betas_j
        grad_at_j = self.get_grad_at_coord(ExpyXB, current_betas_j, yX_j, j)
        step_at_j = grad_at_j / self.Lipschitz
        current_betas_j = prev_betas_j - step_at_j
        # current_betas_j = np.clip(current_betas_j, self.lbs[j], self.ubs[j])
        current_betas_j = max(self.lbs[j], min(self.ubs[j], current_betas_j))
        diff_betas_j = current_betas_j - prev_betas_j
        betas[j] = current_betas_j

        # ExpyXB *= np.exp(yX_j * diff_betas_j)
        self.update_ExpyXB(ExpyXB, yX_j, diff_betas_j)

    def finetune_on_current_support(self, ExpyXB, beta0, betas, total_CD_steps=100):

        support  = np.where(np.abs(betas) > 1e-9)[0]
        grad_on_support = -self.yXT[support].dot(np.reciprocal(1+ExpyXB)) + self.twoLambda2 * betas[support]
        abs_grad_on_support = np.abs(grad_on_support)
        support = support[np.argsort(-abs_grad_on_support)]

        loss_before = compute_logisticLoss_from_ExpyXB(ExpyXB) + self.lambda2 * betas[support].dot(betas[support])
        for steps in range(total_CD_steps): # number of iterations for coordinate descent

            if self.intercept:
                grad_intercept = -np.reciprocal(1+ExpyXB).dot(self.y)
                step_at_intercept = grad_intercept / (self.n * 0.25) # lipschitz constant is 0.25 at the intercept
                beta0 = beta0 - step_at_intercept
                ExpyXB *= np.exp(self.y * (-step_at_intercept))

            for j in support:
                self.optimize_1step_at_coord(ExpyXB, betas, self.yXT[j, :], j) # in-place modification on ExpyXB and betas
            
            if steps % 10 == 0:
                loss_after = compute_logisticLoss_from_ExpyXB(ExpyXB) + self.lambda2 * betas[support].dot(betas[support])
                if abs(loss_before - loss_after)/loss_after < 1e-8:
                    # print("break after {} steps; support size is {}".format(steps, len(support)))
                    break
                loss_before = loss_after
        
        return ExpyXB, beta0, betas
    
    def compute_yXB(self, beta0, betas):
        return self.y*(beta0 + np.dot(self.X_normalized, betas))
 