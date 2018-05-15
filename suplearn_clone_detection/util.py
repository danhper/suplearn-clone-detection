from os import path
from contextlib import contextmanager


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



@contextmanager
def session_scope(session_maker):
    session = session_maker()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
