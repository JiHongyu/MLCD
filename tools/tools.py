def memorize(func):
    cache = dict()
    def compute_key(func, args, kwargs):
        pass
    def inner(*args, **kwargs):

        print('function cache work on')
        if func not in cache:
            cache[func] = func(*args, **kwargs)

        return cache[func]

    return inner


