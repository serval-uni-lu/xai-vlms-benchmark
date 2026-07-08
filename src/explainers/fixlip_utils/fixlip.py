import time
from typing import Optional

import shapiq
import numpy as np
import scipy as sp

from .sampler import CoalitionSampler


class FIxLIP:
    """
    Approximates interaction values using the weighted Banzhaf power index (or Shapley).
    """
    def __init__(
            self, 
            n_players: Optional[int] = None, 
            n_players_image: Optional[int] = None,
            n_players_text: Optional[int] = None,
            mode: str = "banzhaf",
            p: float = 0.5, 
            max_order: int = 2, 
            random_state = None,
            sparse_regression: bool = False
        ):
        self.mode = mode
        self.sparse_regression = sparse_regression
        self.is_crossmodal = False

        # 1. Validate inputs immediately to satisfy strict type checking
        if n_players_image is not None and n_players_text is not None:
            if mode.lower() == "shapley":
                raise ValueError("approximate_crossmodal() is not available for mode 'Shapley'")
            self.is_crossmodal = True
            n_players = n_players_image + n_players_text
            self.n_players_image = n_players_image
            self.n_players_text = n_players_text
        elif n_players is None:
            # Catch the None case before any mathematical operations are performed
            raise ValueError(
                "Pass either `n_players` for basic usage or "
                "pass `n_players_image` and `n_players_text` for crossmodal usage."
            )
        
        # At this point, the compiler guarantees `n_players` is an `int`.
        # Assign it to self immediately so we can use self.n_players reliably.
        self.n_players: int = n_players
        self.p = p
        self.max_order = max_order
        self.random_state = random_state

        # 2. Compute sampling weights safely
        if mode.lower() == "banzhaf":
            # Sample using uniform weights
            sampling_weights = np.array([
                sp.special.binom(self.n_players, k) * (p ** k) * ((1 - p) ** (self.n_players - k))
                for k in range(self.n_players + 1)
            ])
            enforce_empty_full = False
        elif mode.lower() == "shapley":
            sampling_weights = np.zeros(self.n_players + 1)
            # KernelSHAP sampling weights
            for coalition_size in range(1, self.n_players):
                sampling_weights[coalition_size] = 1 / (coalition_size * (self.n_players - coalition_size))
            enforce_empty_full = True
        else:
            raise ValueError("`mode` should be either 'Banzhaf' or 'Shapley'.")
        
        # 3. Initialize specific samplers
        if self.is_crossmodal:
            # The type checker knows these are ints because is_crossmodal is True
            self.sampler_image = CoalitionSampler(
                n_players=self.n_players_image, 
                sampling_weights=np.array([
                    sp.special.binom(self.n_players_image, k) * (p ** k) * ((1 - p) ** (self.n_players_image - k))
                    for k in range(self.n_players_image + 1)
                ]), 
                enforce_empty_full=enforce_empty_full,
                pairing_trick=False, 
                random_state=random_state
            )
            self.sampler_text = CoalitionSampler(
                n_players=self.n_players_text, 
                sampling_weights=np.array([
                    sp.special.binom(self.n_players_text, k) * (p ** k) * ((1 - p) ** (self.n_players_text - k))
                    for k in range(self.n_players_text + 1)
                ]), 
                enforce_empty_full=enforce_empty_full,
                pairing_trick=False, 
                random_state=random_state
            )
        
        # 4. Initialize base sampler
        self.sampler = CoalitionSampler(
            n_players=self.n_players,
            sampling_weights=sampling_weights,
            enforce_empty_full=enforce_empty_full,
            pairing_trick=False, 
            random_state=random_state
        )

    def approximate(
            self, 
            game, 
            budget, 
            interaction_lookup=None, 
            time_game=False, 
            approximation_type="original",
            **kwargs
        ):
        if interaction_lookup is not None and approximation_type != "original":
            raise ValueError("`interaction_lookup` is only used for `approximation_type='original'`.")
        # sample coalitions
        self.sampler.sample(budget)
        # evaluate coalition values (un-normalized game call)
        if time_game:
            self.time_game_start = time.time()
        coalition_values = game.value_function(self.sampler.coalitions_matrix)
        if time_game:
            self.time_game_end = time.time()
        coalition_values = coalition_values - game.normalization_value

        if approximation_type == "original":
            if self.mode.lower() == "banzhaf":
                # set kernel weights for weighted banzhaf
                kernel_weights = np.array([self.p ** k * ((1 - self.p) ** (self.n_players - k))\
                                            for k in range(self.n_players + 1)])
            elif self.mode.lower() == "shapley":
                kernel_weights = np.zeros(self.n_players + 1)
                normalization_constant = 0
                for coalition_size in range(1, self.n_players):
                    kernel_weights[coalition_size] = 1 / sp.special.binom(self.n_players - 2, coalition_size - 1)
                    normalization_constant += kernel_weights[coalition_size] * sp.special.binom(self.n_players, coalition_size)
                # Normalize kernel weights to probability distribution
                kernel_weights /= normalization_constant
                big_M = 10e6
                kernel_weights[0] = big_M
                kernel_weights[-1] = big_M
            else:
                raise ValueError(f"Mode {self.mode.lower()} is not implemented !")
            
            regression_weights = get_regression_weights(self.sampler, kernel_weights)
            # aggregate coalition values
            interaction_values = self.aggregate(
                coalition_matrix=self.sampler.coalitions_matrix, 
                regression_weights=regression_weights,
                coalition_values=coalition_values,
                interaction_lookup=interaction_lookup
            )
        elif approximation_type == "proxyshap":
            # cf. https://github.com/mmschlk/shapiq/blob/ec73ba9746c367f4407603d32a4d587c7e4548f5/src/shapiq/approximator/proxy/proxyshap.py#L239-L285
            from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
            from xgboost import XGBRegressor
            defaults = {
                "n_estimators": 2000,
                "learning_rate": 0.05,
                "max_depth": 3,
                "reg_lambda": 5,
                "random_state": self.random_state
            }
            defaults.update(kwargs)
            proxy_model = XGBRegressor(**defaults)
            proxy_model.fit(self.sampler.coalitions_matrix, coalition_values)
            explainer = InterventionalTreeExplainer(
                proxy_model,
                data=np.zeros((1, self.n_players)),  # reference data for boolean tree
                class_index=None,
                index="FBII" if self.mode.lower() == "banzhaf" else "FSII",
                max_order=self.max_order,
                bool_tree=True,
            )
            values = explainer.explain_function(np.ones((1, self.n_players)))
            interaction_values = shapiq.InteractionValues(
                values=values.interactions,
                index="FBII" if self.mode.lower() == "banzhaf" else "FSII",
                max_order=self.max_order,
                n_players=self.n_players,
                min_order=0,
                estimated=2 ** self.n_players > budget,
                estimation_budget=budget,
                baseline_value=float(game.normalization_value),
            )
            interaction_values[()] = float(game.normalization_value)  # Ensure empty coalition value is correct
            # Ensure that all values are present and pad with zeros if necessary.
            interaction_values = populate_sparse_iv_with_zeros(interaction_values)
        
        else:
            raise ValueError(f"Approximation type {approximation_type} is not implemented !")

        return interaction_values


    def approximate_crossmodal(
        self, 
        game, 
        budget=None, 
        budget_image=None, 
        budget_text=None, 
        interaction_lookup=None, 
        time_game=False, 
        approximation_type="original",
        **kwargs
    ):
        if not self.is_crossmodal:
            raise ValueError("Crossmodal approximation is not initialized."+\
                             "Pass `n_players_image` and `n_players_text` to FIxLIP().")
        if interaction_lookup is not None and approximation_type != "original":
            raise ValueError("`interaction_lookup` is only used for `approximation_type='original'`.")
        # split budget based on n_players_text and n_players_image
        if budget is not None:
            if budget < 4:
                raise ValueError("`budget` should be at least 4.")
            budget_image, budget_text = self.split_budget(budget)
        elif budget_image is None or budget_text is None:
            raise ValueError("Pass either `budget` or `budget_image` and `budget_text`.")
        else:
            budget = budget_image * budget_text
        # sample coalitions from both modalities
        self.sampler_image.sample(budget_image)
        self.sampler_text.sample(budget_text)
        # evaluate coalition values efficiently with _crossmodal (un-normalized game call)
        if time_game:
            self.time_game_start = time.time()
        coalition_values_crossmodal = game.value_function_crossmodal(
            coalitions_image=self.sampler_image.coalitions_matrix,
            coalitions_text=self.sampler_text.coalitions_matrix
        )
        if time_game:
            self.time_game_end = time.time()
        coalition_values_crossmodal = coalition_values_crossmodal - game.normalization_value
        # reshape inputs to aggregate()
        coalition_values = coalition_values_crossmodal.reshape(-1)
        coalitions_matrix = np.concatenate([
            np.repeat(self.sampler_image.coalitions_matrix, budget_text, axis=0), 
            np.tile(self.sampler_text.coalitions_matrix, (budget_image, 1))
        ], axis=1)
        if approximation_type == "original":
            # set kernel weights for image and text using banzhaf
            kernel_weights_image = np.array([self.p ** k * ((1 - self.p) ** (self.n_players_image - k)) \
                                                for k in range(self.n_players_image + 1)])
            kernel_weights_text = np.array([self.p ** k * ((1 - self.p) ** (self.n_players_text - k)) \
                                                for k in range(self.n_players_text + 1)])
            image_regression_weights = get_regression_weights(self.sampler_image, kernel_weights_image)
            text_regression_weights = get_regression_weights(self.sampler_text, kernel_weights_text)
            regression_weights = np.outer(
                image_regression_weights,
                text_regression_weights
            ).reshape(-1)
            # aggregate coalition values with aggregate()
            interaction_values = self.aggregate(
                coalition_matrix=coalitions_matrix, 
                regression_weights=regression_weights,
                coalition_values=coalition_values,
                interaction_lookup=interaction_lookup
            )
        elif approximation_type == "proxyshap":
            # cf. https://github.com/mmschlk/shapiq/blob/ec73ba9746c367f4407603d32a4d587c7e4548f5/src/shapiq/approximator/proxy/proxyshap.py#L239-L285
            from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
            from xgboost import XGBRegressor
            defaults = {
                "n_estimators": 2000,
                "learning_rate": 0.05,
                "max_depth": 3,
                "reg_lambda": 5,
                "random_state": self.random_state
            }
            defaults.update(kwargs)
            proxy_model = XGBRegressor(**defaults)
            proxy_model.fit(coalitions_matrix, coalition_values)
            explainer = InterventionalTreeExplainer(
                proxy_model,
                data=np.zeros((1, self.n_players)),  # reference data for boolean tree
                class_index=None,
                index="FBII" if self.mode.lower() == "banzhaf" else "FSII",
                max_order=self.max_order,
                bool_tree=True,
            )
            values = explainer.explain_function(np.ones((1, self.n_players)))
            interaction_values = shapiq.InteractionValues(
                values=values.interactions,
                index="FBII" if self.mode.lower() == "banzhaf" else "FSII",
                max_order=self.max_order,
                n_players=self.n_players,
                min_order=0,
                estimated=2**self.n_players > budget,
                estimation_budget=budget,
                baseline_value=float(game.normalization_value),
            )
            interaction_values[()] = float(game.normalization_value)  # Ensure empty coalition value is correct
            # Ensure that all values are present and pad with zeros if necessary.
            interaction_values = populate_sparse_iv_with_zeros(interaction_values)
        
        else:
            raise ValueError(f"Approximation type {approximation_type} is not implemented !")

        return interaction_values


    def aggregate(
        self,
        coalition_matrix,
        regression_weights,
        coalition_values,
        interaction_lookup: dict | None = None
    ) -> shapiq.InteractionValues:
        """Aggregates the coalition values using the weighted Banzhaf power index."""
        n_coalitions, n_players = np.shape(coalition_matrix)
        # populate interactions to use for regression
        if interaction_lookup is None:  # first check if interaction_lookup is passed
            interaction_lookup = shapiq.utils.generate_interaction_lookup(set(range(n_players)), min_order=0, max_order=self.max_order)
        n_interactions = len(interaction_lookup)
        # set response, subtract baseline for better approximation, it will be added later
        regression_response = coalition_values.copy()
        # create regression matrix
        regression_matrix = np.zeros((n_coalitions, n_interactions))
        for i, interaction in enumerate(interaction_lookup.keys()):
            regression_matrix[:, i] = coalition_matrix[:, interaction].prod(axis=1)
        # solve regression
        values = solve_regression(regression_matrix, regression_response, regression_weights, sparse_regression=self.sparse_regression)
        # return interaction values
        interaction_values = shapiq.InteractionValues(
            values=values,
            interaction_lookup=interaction_lookup,
            baseline_value=float(values[interaction_lookup[()]]),
            n_players=n_players,
            index="Moebius",
            max_order=self.max_order,
            min_order=0,
            estimated=2 ** n_players > n_coalitions,
            estimation_budget=n_coalitions,
        )
        interaction_values.index = "FSII" if self.mode.lower() == "shapley" else "FWBII"
        return interaction_values
    

    #:# ---------- utility functions ---------- #:#

    def split_budget(self, budget):
        """
        Heuristic to choose a reasonable budget split.
        """
        if self.n_players_text < self.n_players_image:
            budget_text = np.sqrt(budget) * self.n_players_text / self.n_players_image
            budget_text = int(np.ceil(np.max([4, budget_text])))
            budget_text = int(np.min([2 ** self.n_players_text, budget_text]))
            budget_image = int(budget / budget_text)
        else:
            budget_image = np.sqrt(budget) * self.n_players_image / self.n_players_text
            budget_image = int(np.ceil(np.max([4, budget_image])))
            budget_image = int(np.min([2 ** self.n_players_image, budget_image]))
            budget_text = int(budget / budget_image)    
        return budget_image, budget_text


def solve_regression(X: np.ndarray, y: np.ndarray, kernel_weights: np.ndarray, sparse_regression=False) -> np.ndarray:
    if not sparse_regression:
        try:
            # try solving via solve function
            WX = kernel_weights[:, np.newaxis] * X
            phi = np.linalg.solve(X.T @ WX, WX.T @ y)
        except np.linalg.LinAlgError:
            # solve WLSQ via lstsq function and throw warning
            W_sqrt = np.sqrt(kernel_weights)
            X = W_sqrt[:, np.newaxis] * X
            y = W_sqrt * y
            phi = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        X_sparse = sp.sparse.csr_matrix(X)
        W_sparse = sp.sparse.diags(kernel_weights)
        WX_sparse = W_sparse @ X_sparse  # Pre-multiply: W * X and W * y
        Wy = kernel_weights * y
        result = sp.sparse.linalg.lsqr(WX_sparse, Wy)  # Solve sparse least squares
        phi = result[0]  # coefficients
        return phi
    return phi


def get_regression_weights(sampler, kernel_weights):
    """ Computes the regression weights, requires that sampling weights are proportional to kernel weights.
    Regression weights are equal to kernel weights for coalitions that are not sampled (using the border-trick).
    Otherwise, the regression weights are set to the empirical averages, i.e. # occurrences / # sampled coalitions.
    """
    regression_weights_not_sampled = kernel_weights[np.sum(sampler.coalitions_matrix, axis=1)]
    regression_weights = sampler.empirical_occurrences
    regression_weights[~sampler.is_coalition_sampled] = regression_weights[~sampler.is_coalition_sampled] * \
                                                        regression_weights_not_sampled[~sampler.is_coalition_sampled]
    return regression_weights


def populate_sparse_iv_with_zeros(iv):
    """Fills in the interaction values object with zeros for missing attributions."""
    new_values = []
    new_interaction_lookup = {}
    for index, interaction in enumerate(shapiq.utils.sets.powerset(range(iv.n_players), min_size=iv.min_order, max_size=iv.max_order)):
        new_interaction_lookup[interaction] = index
        if interaction in iv.interaction_lookup:
            new_values.append(iv[interaction])
        else:
            new_values.append(0.0)
    return shapiq.InteractionValues(
        values=np.array(new_values, dtype=float),
        index=iv.index,
        max_order=iv.max_order,
        n_players=iv.n_players,
        min_order=iv.min_order,
        interaction_lookup=new_interaction_lookup,
        estimated=iv.estimated,
        estimation_budget=iv.estimation_budget,
        baseline_value=iv.baseline_value
    )


