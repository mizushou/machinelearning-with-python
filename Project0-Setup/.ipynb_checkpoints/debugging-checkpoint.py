def get_sum_metrics(predictions, metrics=[]):
    for i in range(3):
        metrics.append(lambda x: x + i)

    import pdb; pdb.set_trace()
    sum_metrics = 0
    for metric in metrics:
#         import pdb; pdb.set_trace()
#         print(type(metric))
#         print(type(predictions))
        print(str(predictions))
        sum_metrics += metric(int(predictions))

    return sum_metrics

get_sum_metrics(0.1, [3, 4, 5])