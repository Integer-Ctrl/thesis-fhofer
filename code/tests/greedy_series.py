import pandas as pd


class GreedySeries(pd.Series):
    def corr(self, other, method, min_periods: int | None = None,):
        if '-greedy' in method:
            return self.greedy_corr(other, method.replace('-greedy', ''), min_periods)
        else:
            return super().corr(other, method, min_periods)

    def greedy_corr(self, other, method, min_periods: int | None = None,):
        # Step 1: Pair and sort ar1 and ar2 together
        paired = list(zip(self, other))

        # Sort the pairs based on the values in ar1
        paired.sort(key=lambda x: x[0])

        # Unzip the sorted pairs back into ar1 and ar2
        ar1_sorted, ar2_sorted = zip(*paired)

        # Return the sorted arrays as lists
        ar1 = list(ar1_sorted)
        ar2 = list(ar2_sorted)

        # Step 2: Identify unique values in ar2
        unique_ar2 = sorted(set(ar2))  # Get unique values in sorted order

        # APPROACH 1: Custom
        # Step 3: Determine the boarders
        # boarders = []
        # current_group = unique_ar2[0]  # Start with the smallest group
        # max_val = None

        # for value, group in zip(ar1, ar2):
        #     if group >= current_group:  # If we encounter a new group
        #         boarders.append(max_val)  # Add the max value of the current group as a boarder
        #         current_group = group
        #         max_val = value  # Start tracking max for the new group
        #     else:
        #         max_val = value  # Update max value within the group

        # boarders.append(max_val)  # Add the last group's max value as the final boarder

        # # Step 4: Map values of ar1 based on the boarders
        # def map_value_1(value):
        #     for i, boarder in enumerate(boarders):
        #         if value <= boarder:
        #             return unique_ar2[i]
        #     return unique_ar2[-1]  # If greater than the last boarder, assign the largest group

        # ar1 = [map_value_1(value) for value in ar1]

        # APPROACH 2: Greedy
        current_label = unique_ar2[0]
        for index, (value, label) in enumerate(zip(ar1, ar2)):
            if label > current_label:
                current_label = label
                ar1[index] = current_label
            else:
                ar1[index] = current_label

        ar1_series = pd.Series(ar1)
        ar2_series = pd.Series(ar2)

        return ar1_series.corr(ar2_series, method=method)
