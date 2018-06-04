import random

from typing import List, Optional, Dict, Tuple
from suplearn_clone_detection import entities


def find_submission_index(submissions: List[entities.Submission],
                          tokens_count: int, rightmost=False) -> int:
    left = 0
    right = len(submissions)
    while left < right:
        middle = (left + right) // 2
        submission = submissions[middle]
        if submission.tokens_count < tokens_count or \
            (submission.tokens_count == tokens_count and rightmost):
            left = middle + 1
        else:
            right = middle
    return left


def select_negative_candidates(
        sorted_submissions: List[entities.Submission],
        positive_sample: entities.Submission,
        distance: int,
        min_candidates: Optional[int] = None) -> List[entities.Submission]:
    tokens_count = positive_sample.tokens_count
    tokens_diff = int(distance * tokens_count)
    left_index = find_submission_index(sorted_submissions, tokens_count - tokens_diff)
    right_index = find_submission_index(sorted_submissions,
                                        tokens_count + tokens_diff, rightmost=True)
    candidates = sorted_submissions[left_index:right_index]

    if not min_candidates:
        return candidates

    while len(candidates) < min_candidates:
        new_candidate = random.choice(sorted_submissions)
        if new_candidate.group_key != positive_sample.group_key and \
            new_candidate not in candidates:
            candidates.append(new_candidate)
    return candidates


def select_negative_sample(
        sorted_submissions: List[entities.Submission],
        positive_sample: entities.Submission,
        distance: int) -> entities.Submission:
    candidates = select_negative_candidates(sorted_submissions, positive_sample, distance)
    while candidates:
        idx = random.randrange(len(candidates))
        negative_sample = candidates[idx]
        if positive_sample.group_key != negative_sample.group_key:
            return negative_sample
        del candidates[idx]


def group_submissions(submissions: List[entities.Submission]) \
        -> Dict[Tuple[str, int, int], entities.Submission]:
    result = {}
    for submission in submissions:
        result.setdefault(submission.group_key, [])
        result[submission.group_key].append(submission)
    return result


def sort_dataset(submissions: List[entities.Submission]) \
        -> Tuple[List[entities.Submission], Dict[int, int]]:
    sorted_submissions = sorted(submissions, key=lambda x: x.tokens_count)
    return sorted_submissions
