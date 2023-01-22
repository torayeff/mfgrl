import numpy as np


def is_pareto_efficient(costs, return_mask=True):
    """
    Inherited from
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


costs = np.array(
    [
        [1000, 20, -1, 5],
        [1300, 32, -1.5, 7],
        [2000, 20, -2, 9],
        [2400, 13, -0.75, 6],
        [5000, 65, -3, 10],
    ]
)

print(is_pareto_efficient(costs=costs))
