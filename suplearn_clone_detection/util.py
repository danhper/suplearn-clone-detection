from os import path
import functools


def filename_language(filename, available_languages):
    _basename, ext = path.splitext(filename)
    for known_lang in available_languages:
        if known_lang.startswith(ext[1:]):
            return known_lang
    raise ValueError("no language found for {0}".format(filename))


def in_batch(iterable, batch_size):
    iterator = iter(iterable)
    batch = []
    while True:
        try:
            batch.append(next(iterator))
            if len(batch) == batch_size:
                yield batch
                batch = []
        except StopIteration:
            if batch:
                yield batch
            return


def group_by(iterable, key):
    iterator = iter(iterable)
    grouped = {}
    while True:
        try:
            value = next(iterator)
            value_key = key(value)
            grouped.setdefault(value_key, [])
            grouped[value_key].append(value)
        except StopIteration:
            break
    return grouped


def memoize(make_key):
    def wrapped(f):
        memoized = {}

        def wrapper(*args, **kwargs):
            key = make_key(*args, **kwargs)
            if key not in memoized:
                memoized[key] = f(*args)
            return memoized[key]

        return functools.update_wrapper(wrapper, f)

    if not callable(make_key) or not hasattr(make_key, "__name__"):
        raise ValueError("@memoize argument should be a lambda")

    if make_key.__name__ == "<lambda>":
        return wrapped
    else:
        f = make_key
        make_key = lambda *args, **kwargs: (tuple(args), (tuple(kwargs.items())))
        return wrapped(f)
