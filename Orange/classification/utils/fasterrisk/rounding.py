import numpy as np
import sys
# import warnings
# warnings.filterwarnings("ignore")

from Orange.classification.utils.fasterrisk.utils import get_support_indices, compute_logisticLoss_from_betas_and_yX, insertIntercept_asFirstColOf_X

class starRaySearchModel:
    def __init__(self, X, y, lb=-5, ub=5, num_ray_search=20, early_stop_tolerance=0.001):
        self.X = insertIntercept_asFirstColOf_X(X)
        self.y = y.reshape(-1)
        self.yX = self.y.reshape(-1, 1) * self.X

        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

        if isinstance(ub, (float, int)):
            self.ub_arr = ub * np.ones((self.p, ))
            self.ub_arr[0] = 100.0 # intercept upper bound
        else:
            self.ub_arr = np.insert(ub, 0, 100.0)   # add intercept upper bound
        
        if isinstance(lb, (float, int)):
            self.lb_arr = lb * np.ones((self.p, ))
            self.lb_arr[0] = -100.0 # intercept lower bound
        else:
            self.lb_arr = np.insert(lb, 0, -100)    # add intercept lower bound

        self.num_ray_search = num_ray_search
        self.early_stop_tolerance = early_stop_tolerance
    
    def get_multipliers_for_line_search(self, betas):
        """Get an array of multipliers to try for line search

        Parameters
        ----------
        betas : ndarray
            (1D array with `float` type) a given solution with shape = (1+p, ) assuming the first entry is the intercept

        Returns
        -------
        multipliers : ndarray
            (1D array with `float` type) an array of candidate multipliers with shape = (num_ray_search, )
        """
        # largest_multiplier = min(self.abs_coef_ub/np.max(np.abs(betas[1:])), self.abs_intercept_ub/abs(betas[0]))
        pos_nonzeroIndices = np.where(betas > 1e-8)[0]
        neg_nonzeroIndices = np.where(betas < -1e-8)[0]
        len_pos_nonzeroIndices = len(pos_nonzeroIndices)
        len_neg_nonzeroIndices = len(neg_nonzeroIndices)

        assert len_pos_nonzeroIndices + len_neg_nonzeroIndices > 0, "betas needs to have at least one nonzero entries!"
        largest_multiplier = 1e8
        if len_pos_nonzeroIndices > 0:
            largest_multiplier = min(largest_multiplier, min(self.ub_arr[pos_nonzeroIndices] / betas[pos_nonzeroIndices]))
        if len_neg_nonzeroIndices > 0:
            largest_multiplier = min(largest_multiplier, min(self.lb_arr[neg_nonzeroIndices] / betas[neg_nonzeroIndices]))
        
        if largest_multiplier > 1:
            multipliers = np.linspace(1, largest_multiplier, self.num_ray_search)
        else:
            multipliers = np.linspace(1, 0.5, self.num_ray_search)
        return multipliers
    
    def star_ray_search_scale_and_round(self, sparseDiversePool_beta0_continuous, sparseDiversePool_betas_continuous):
        """For each continuous solution in the sparse diverse pool, find the best multiplier and integer solution. Return the best integer solutions and the corresponding multipliers in the sparse diverse pool

        Parameters
        ----------
        sparseDiversePool_beta_continuous : ndarray
            (1D array with `float` type) an array of continuous intercept with shape = (m, )
        sparseDiversePool_betas_continuous : ndarray
            (2D array with `float` type) an array of continuous coefficients with shape = (m, p) 

        Returns
        -------
        multipliers : ndarray
            (1D array with `float` type) best multiplier for each continuous solution with shape = (m, )
        best_beta0 : ndarray
            (1D array with `float` type) best integer intercept for each continuous solution with shape = (m, )
        best_betas : ndarray
            (2D array with `float` type) best integer coefficient for each continuous solution with shape = (m, p)
        """
        sparseDiversePool_continuous = np.hstack((sparseDiversePool_beta0_continuous.reshape(-1, 1), sparseDiversePool_betas_continuous))
        
        sparseDiversePool_integer = np.zeros(sparseDiversePool_continuous.shape)
        multipliers = np.zeros((sparseDiversePool_integer.shape[0]))

        for i in range(len(multipliers)):
            multipliers[i], sparseDiversePool_integer[i] = self.line_search_scale_and_round(sparseDiversePool_continuous[i])
        
        return multipliers, sparseDiversePool_integer[:, 0], sparseDiversePool_integer[:, 1:]

    def line_search_scale_and_round(self, betas):
        """For a given solution betas, multiply the solution with different multipliers and round each scaled solution to integers. Return the best integer solution based on the logistic loss.

        Parameters
        ----------
        betas : ndarray
            (1D array with `float` type) a given solution with shape = (1+p, ) assuming the first entry is the intercept

        Returns
        -------
        best_multiplier : float
            best multiplier among all pairs of (multiplier, integer_solution)
        best_betas : ndarray
            (1D array with `float` type) best integer solution among all pairs of (multiplier, integer_solution)
        """
        nonzero_indices = get_support_indices(betas)
        num_nonzero = len(nonzero_indices)

        # X_sub = self.X[:, nonzero_indices]
        yX_sub = self.yX[:, nonzero_indices]
        betas_sub = betas[nonzero_indices]

        multipliers = self.get_multipliers_for_line_search(betas_sub)

        loss_continuous_betas = compute_logisticLoss_from_betas_and_yX(betas_sub, yX_sub)
        
        best_multiplier = 1.0
        best_loss = 1e12
        best_betas_sub = np.zeros((num_nonzero, ))

        for multiplier in multipliers:
            betas_sub_scaled = betas_sub * multiplier
            yX_sub_scaled = yX_sub / multiplier

            betas_sub_scaled = self.auxilliary_rounding(betas_sub_scaled, yX_sub_scaled)

            tmp_loss = compute_logisticLoss_from_betas_and_yX(betas_sub_scaled / multiplier, yX_sub)

            if tmp_loss < best_loss:
                best_loss = tmp_loss
                best_multiplier = multiplier
                best_betas_sub[:] = betas_sub_scaled[:]

            if (tmp_loss - loss_continuous_betas) / loss_continuous_betas < self.early_stop_tolerance:
                break

        best_betas = np.zeros((self.p, ))
        best_betas[nonzero_indices] = best_betas_sub

        return best_multiplier, best_betas

    def get_rounding_distance_and_dimension(self, betas):
        """For each dimension, get distances from the real coefficient to the rounded-up integer and the rounded-down integer

        Parameters
        ----------
        betas : ndarray
            (1D array with `float` type) current continuous (real-valued) solution

        Returns
        -------
        betas_floor : ndarray
            (1D array with `float` type) rounded-down coefficients
        dist_from_start_to_floor: ndarray
            (1D array with `float` type) distance from the real coefficient to the rounded-down integer
        betas_ceil : ndarray 
            (1D array with `float` type) rounded-up coefficients
        dist_from_start_to_ceil: ndarray 
            (1D array with `float` type) distance from the real coefficient to the rounded-up integer
        dimensions_to_round: int[:]
            array of indices where the coefficients are not integers to begin with and upon which we should do rounding
        """
        betas_floor = np.floor(betas)
        # floor_is_zero = np.equal(betas_floor, 0)
        dist_from_start_to_floor = betas_floor - betas

        betas_ceil = np.ceil(betas)
        # ceil_is_zero = np.equal(betas_ceil, 0)
        dist_from_start_to_ceil = betas_ceil - betas

        dimensions_to_round = np.flatnonzero(np.not_equal(betas_floor, betas_ceil)).tolist()

        return betas_floor, dist_from_start_to_floor, betas_ceil, dist_from_start_to_ceil, dimensions_to_round

    def auxilliary_rounding(self, betas, yX):
        """Round the solutions to intgers according to the auxilliary loss proposed in the paper

        Parameters
        ----------
        betas : ndarray
            (1D array with `float` type) current continuous (real-valued) solution
        yX : ndarray
            (2D array with `float` type) yX[i, j] = y[i] * X[i, j]

        Returns
        -------
        integer_beta : ndarray
            (1D array with `float` type) rounded integer solution
        """
        n_local, p_local = yX.shape[0], yX.shape[1]

        betas_floor, dist_from_start_to_floor, betas_ceil, dist_from_start_to_ceil, dimensions_to_round = self.get_rounding_distance_and_dimension(betas)

        # yXB = yX.dot(betas) # shape is (n_local, )

        Gamma = np.zeros((n_local, p_local))
        Gamma[:] = betas_floor
        Gamma = Gamma + 1.0 * (yX <= 0)

        yX_Gamma = yX * Gamma
        yXB_extreme = np.sum(yX_Gamma, axis=1)
        l_factors = np.reciprocal((1 + np.exp(yXB_extreme))) # corresponding to l_i's in the NeurIPS paper

        lyX = l_factors.reshape(-1, 1) * yX
        lyX_norm_square = np.sum(lyX * lyX, axis = 0)

        upperBound_arr = 1e12 * np.ones((2 * p_local))
        lyXB_diff = np.zeros((n_local, )) # at the start, betas are not rounded, so coefficient difference is zero
        current_upperBound = 0 # at the start, upper is also 0 because betas have not been rounded yet

        while len(dimensions_to_round) > 0:
            upperBound_arr.fill(1e12)

            for j in dimensions_to_round:
                upperBound_expectation = current_upperBound - lyX_norm_square[j] * dist_from_start_to_floor[j] * dist_from_start_to_ceil[j]

                lyX_j = lyX[:, j]
                lyXB_diff_floor_j = lyXB_diff + dist_from_start_to_ceil[j] * lyX_j
                upperBound_arr[2*j+1] = np.sum(lyXB_diff_floor_j ** 2) # odd positions stores upper bound for ceiling operation

                if upperBound_arr[2*j+1] > upperBound_expectation:
                    lyXB_diff_ceil_j = lyXB_diff + dist_from_start_to_floor[j] * lyX_j
                    upperBound_arr[2*j] = np.sum(lyXB_diff_ceil_j ** 2) # even positions stores upper bound for flooring operation
            
            best_idx_upperBound_arr = np.argmin(upperBound_arr)
            current_upperBound = upperBound_arr[best_idx_upperBound_arr]

            best_j, is_ceil = best_idx_upperBound_arr // 2, best_idx_upperBound_arr % 2

            if is_ceil:
                betas[best_j] += dist_from_start_to_ceil[best_j]
                lyXB_diff = lyXB_diff + dist_from_start_to_ceil[best_j] * lyX[:, best_j]
            else:
                betas[best_j] += dist_from_start_to_floor[best_j]
                lyXB_diff = lyXB_diff + dist_from_start_to_floor[best_j] * lyX[:, best_j]
            
            dimensions_to_round.remove(best_j)
        
        return betas