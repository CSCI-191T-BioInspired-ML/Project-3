ranges = [10, 5, 3, 2, 1]
# return the counts of how many are less than 10, less than 5, less than 3, less than 2, less than 1
def proximity_percentage(y_actual, y_predicted, dict):
    y = abs(y_actual - y_predicted)
    for i in ranges:
        if y < i:
            dict[i] = dict.get(i, 0) + 1
    return dict

def average_proximity_percentage(totals, iterations):
    avgs = {}
    for i in ranges:
        avgs[i] = totals.get(i, 0) / iterations * 100
    return avgs