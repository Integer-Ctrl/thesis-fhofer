import pandas as pd
import numpy as np


class GreedySeries(pd.Series):
    def corr(self, other, method, min_periods: int | None = None,):
        if '-greedy' in method:
            return self.greedy_corr(other, method.replace('-greedy', ''), min_periods)
        else:
            return super().corr(other, method, min_periods)

    def greedy_corr(self, other, method, min_periods: int | None = None,):

        scores = self.to_list()
        labels = other.to_list()

        unique_labels = sorted(set(labels))  # Unique labels
        min = 0.0  # Minimum value of the current group
        max = 1.0  # Maximum value of the current group
        step = 0.01  # Step size for the greedy algorithm
        boundaries = []  # Boundaries of the groups

        # Step 1: Compute boundaries for each label
        for iteration in range(0, len(unique_labels) - 1):
            max_correlation = -float('inf')
            best_boundary = -float('inf')
            for boundary in np.arange(min, max, step):
                # Normalize the scores based on the boundary
                new_scores = [1 if score < boundary else 0 for score in scores]
                # If an input array is constant; the correlation coefficient is not define
                if len(set(new_scores)) == 1:
                    continue
                normalized_labels = [1 if label in [unique_labels[iteration]] else 0 for label in labels]
                correlation = pd.Series(new_scores).corr(pd.Series(normalized_labels), method=method)
                if correlation > max_correlation:
                    max_correlation = correlation
                    best_boundary = boundary
            min = best_boundary
            boundaries.append(best_boundary)

        # Step 2: Map scores to labels based on boundaries
        mapped_scores = []
        for score in scores:
            for i in range(0, len(boundaries)):  # Loop through boundaries
                if score < boundaries[i]:
                    mapped_scores.append(unique_labels[i])
                    break  # Exit the loop once a match is found
            else:
                # This is executed only if no boundary condition matched
                mapped_scores.append(unique_labels[-1])

        return pd.Series(mapped_scores).corr(pd.Series(labels), method=method)
