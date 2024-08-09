import numpy as np
import sys
# import warnings
# warnings.filterwarnings("ignore")

from Orange.classification.utils.fasterrisk.utils import get_support_indices, get_nonsupport_indices, compute_logisticLoss_from_ExpyXB
from Orange.classification.utils.fasterrisk.base_model import logRegModel

class sparseLogRegModel(logRegModel):
    def __init__(self, X, y, lambda2=1e-8, intercept=True, original_lb=-5, original_ub=5):
        super().__init__(X=X, y=y, lambda2=lambda2, intercept=intercept, original_lb=original_lb, original_ub=original_ub)
    
    def getAvailableIndices_for_expansion(self, betas):
        """Get the indices of features that can be added to the support of the current sparse solution

        Parameters
        ----------
        betas : ndarray
            (1D array with `float` type) The current sparse solution

        Returns
        -------
        available_indices : ndarray
            (1D array with `int` type) The indices of features that can be added to the support of the current sparse solution
        """
        available_indices = get_nonsupport_indices(betas)
        return available_indices
   
    def expand_parent_i_support_via_OMP_by_1(self, i, child_size=10):
        """For parent solution i, generate [child_size] child solutions

        Parameters
        ----------
        i : int
            index of the parent solution
        child_size : int, optional
            how many child solutions to generate based on parent solution i, by default 10
        """
        # non_support = get_nonsupport_indices(self.betas_arr_parent[i])
        non_support = self.getAvailableIndices_for_expansion(self.betas_arr_parent[i])
        support = get_support_indices(self.betas_arr_parent[i])

        grad_on_non_support = self.yXT[non_support].dot(np.reciprocal(1+self.ExpyXB_arr_parent[i]))
        abs_grad_on_non_support = np.abs(grad_on_non_support)

        num_new_js = min(child_size, len(non_support))
        new_js = non_support[np.argsort(-abs_grad_on_non_support)][:num_new_js]
        child_start, child_end = i*child_size, i*child_size + num_new_js

        self.ExpyXB_arr_child[child_start:child_end] = self.ExpyXB_arr_parent[i, :] # (num_new_js, n)
        # self.betas_arr_child[child_start:child_end, non_support] = 0
        self.betas_arr_child[child_start:child_end] = 0
        self.betas_arr_child[child_start:child_end, support] = self.betas_arr_parent[i, support]
        self.beta0_arr_child[child_start:child_end] = self.beta0_arr_parent[i]
        
        beta_new_js = np.zeros((num_new_js, )) #(len(new_js), )
        diff_max = 1e3

        step = 0
        while step < 10 and diff_max > 1e-3:
            prev_beta_new_js = beta_new_js.copy()
            grad_on_new_js = -np.sum(self.yXT[new_js] * np.reciprocal(1.+self.ExpyXB_arr_child[child_start:child_end]), axis=1) + self.twoLambda2 * beta_new_js
            step_at_new_js = grad_on_new_js / self.Lipschitz

            beta_new_js = prev_beta_new_js - step_at_new_js
            beta_new_js = np.clip(beta_new_js, self.lbs[new_js], self.ubs[new_js])
            diff_beta_new_js = beta_new_js - prev_beta_new_js

            self.ExpyXB_arr_child[child_start:child_end] *= np.exp(self.yXT[new_js] * diff_beta_new_js.reshape(-1, 1))

            diff_max = max(np.abs(diff_beta_new_js))
            step += 1

        for l in range(num_new_js):
            child_id = child_start + l
            self.betas_arr_child[child_id, new_js[l]] = beta_new_js[l]
            tmp_support_str = str(get_support_indices(self.betas_arr_child[child_id]))
            if tmp_support_str not in self.forbidden_support:
                self.total_child_added += 1 # count how many unique child has been added for a specified support size
                self.forbidden_support.add(tmp_support_str)

                self.ExpyXB_arr_child[child_id], self.beta0_arr_child[child_id], self.betas_arr_child[child_id] = self.finetune_on_current_support(self.ExpyXB_arr_child[child_id], self.beta0_arr_child[child_id], self.betas_arr_child[child_id])
                self.loss_arr_child[child_id] = compute_logisticLoss_from_ExpyXB(self.ExpyXB_arr_child[child_id])

    def beamSearch_multipleSupports_via_OMP_by_1(self, parent_size=10, child_size=10):
        """Each parent solution generates [child_size] child solutions, so there will be [parent_size] * [child_size] number of total child solutions. However, only the top [parent_size] child solutions are retained as parent solutions for the next level i+1.

        Parameters
        ----------
        parent_size : int, optional
            how many top solutions to retain at each level, by default 10
        child_size : int, optional
            how many child solutions to generate based on each parent solution, by default 10
        """
        self.loss_arr_child.fill(1e12)
        self.total_child_added = 0

        for i in range(self.num_parent):
            self.expand_parent_i_support_via_OMP_by_1(i, child_size=child_size)

        child_indices = np.argsort(self.loss_arr_child)[:min(parent_size, self.total_child_added)] # get indices of children which have the smallest losses
        num_child_indices = len(child_indices)
        self.ExpyXB_arr_parent[:num_child_indices], self.beta0_arr_parent[:num_child_indices], self.betas_arr_parent[:num_child_indices] = self.ExpyXB_arr_child[child_indices], self.beta0_arr_child[child_indices], self.betas_arr_child[child_indices]

        self.num_parent = num_child_indices

    def get_sparse_sol_via_OMP(self, k, parent_size=10, child_size=10):
        """Get sparse solution through beam search and orthogonal matching pursuit (OMP), for level i, each parent solution generates [child_size] child solutions, so there will be [parent_size] * [child_size] number of total child solutions. However, only the top [parent_size] child solutions are retained as parent solutions for the next level i+1.

        Parameters
        ----------
        k : int
            number of nonzero coefficients for the final sparse solution
        parent_size : int, optional
            how many top solutions to retain at each level, by default 10
        child_size : int, optional
            how many child solutions to generate based on each parent solution, by default 10
        """
        nonzero_indices_set = set(np.where(np.abs(self.betas) > 1e-9)[0])
        # print("get_sparse_sol_via_OMP, initial support is:", nonzero_indices_set)
        zero_indices_set = set(range(self.p)) - nonzero_indices_set
        num_nonzero = len(nonzero_indices_set)

        if len(zero_indices_set) == 0:
            return

        # if there is no warm start solution, initialize beta0 analytically
        if (self.intercept) and (len(nonzero_indices_set) == 0):
            y_sum = np.sum(self.y)
            num_y_pos_1 = (y_sum + self.n)/2
            num_y_neg_1 = self.n - num_y_pos_1
            self.beta0 = np.log(num_y_pos_1/num_y_neg_1)
            self.ExpyXB *= np.exp(self.y * self.beta0)

        # create beam search parent
        self.ExpyXB_arr_parent = np.zeros((parent_size, self.n))
        self.beta0_arr_parent = np.zeros((parent_size, ))
        self.betas_arr_parent = np.zeros((parent_size, self.p))
        self.ExpyXB_arr_parent[0, :] = self.ExpyXB[:]
        self.beta0_arr_parent[0] = self.beta0
        self.betas_arr_parent[0, :] = self.betas[:]
        self.num_parent = 1

        # create beam search children. parent[i]->child[i*child_size:(i+1)*child_size]
        total_child_size = parent_size * child_size
        self.ExpyXB_arr_child = np.zeros((total_child_size, self.n))
        self.beta0_arr_child = np.zeros((total_child_size, ))
        self.betas_arr_child = np.zeros((total_child_size, self.p))
        self.isMasked_arr_child = np.ones((total_child_size, ), dtype=bool)
        self.loss_arr_child = 1e12 * np.ones((total_child_size, ))
        self.forbidden_support = set()

        while num_nonzero < min(k, self.p):
            num_nonzero += 1
            self.beamSearch_multipleSupports_via_OMP_by_1(parent_size=parent_size, child_size=child_size)

        self.ExpyXB, self.beta0, self.betas = self.ExpyXB_arr_parent[0], self.beta0_arr_parent[0], self.betas_arr_parent[0]

class groupSparseLogRegModel(sparseLogRegModel):
    def __init__(self, X, y, lambda2=1e-8, intercept=True, original_lb=-5, original_ub=5, group_sparsity=10, featureIndex_to_groupIndex=None, groupIndex_to_featureIndices=None):
        super().__init__(X=X, y=y, lambda2=lambda2, intercept=intercept, original_lb=original_lb, original_ub=original_ub)

        self.group_sparsity = group_sparsity
        self.featureIndex_to_groupIndex = featureIndex_to_groupIndex # this is a numpy array
        self.groupIndex_to_featureIndices = groupIndex_to_featureIndices # this is a dictionary of sets
    
    def getAvailableIndices_for_expansion(self, betas):
        """Get the indices of features that can be added to the support of the current sparse solution

        Parameters
        ----------
        betas : ndarray
            (1D array with `float` type) The current sparse solution

        Returns
        -------
        available_indices : ndarray
            (1D array with `int` type) The indices of features that can be added to the support of the current sparse solution
        """
        support = get_support_indices(betas)
        existing_groupIndices = np.unique(self.featureIndex_to_groupIndex[support])
        if len(existing_groupIndices) < self.group_sparsity:
            available_indices = get_nonsupport_indices(betas)
        else:
            available_indices = set()
            for groupIndex in existing_groupIndices:
                available_indices.update(self.groupIndex_to_featureIndices[groupIndex])
            available_indices = available_indices - set(support)
            available_indices = np.array(list(available_indices), dtype=int)

        return available_indices
   