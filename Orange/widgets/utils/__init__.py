def progress_bar_milestones(count, iterations=100):
    return set([int(i*count/float(iterations)) for i in range(iterations)])
