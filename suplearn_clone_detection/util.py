from os import path
import functools

import h5py


def filename_language(filename, available_languages):
    _basename, ext = path.splitext(filename)
    for known_lang in available_languages:
        if known_lang.startswith(ext[1:]):
            return known_lang
    raise ValueError("no language found for {0}".format(filename))


def in_batch(iterable, batch_size):
    batch = []
    for value in iterable:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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


def hdf5_keys(dataset: h5py.File):
    keys = []
    def visitor(key, value):
        if isinstance(value, h5py.Dataset):
            keys.append(key)
    dataset.visititems(visitor)
    return keys

def hdf5_key_pairs(dataset: h5py.File, lang_left: str, lang_right: str):
    keys = hdf5_keys(dataset)
    def has_correct_ext(filename, expected):
        ext = path.splitext(filename)[1][1:]
        return expected.startswith(ext)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if has_correct_ext(left, lang_left) and has_correct_ext(right, lang_right):
                yield (left, right)
