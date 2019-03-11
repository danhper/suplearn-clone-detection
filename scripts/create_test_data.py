#!/usr/bin/env python

from os import path
import json
import random
import argparse

parser = argparse.ArgumentParser(prog="create-test-data")
parser.add_argument("-o", "--output", default="to_check_cross_lang.txt")
parser.add_argument("--max-tokens", type=int, default=250)
parser.add_argument("-n", "--projects-count", default=50, type=int)
parser.add_argument("--min-files", default=4, type=int)
parser.add_argument("--max-files", default=20, type=int)
parser.add_argument("--contest-type", choices=["r", "b"])
args = parser.parse_args()


with open("./asts/asts.jsonl") as f:
    all_asts = [json.loads(v) for v in f if v]

with open("./asts/asts.txt") as f:
    all_files = [v.strip() for v in f if v]

short_files = [name
               for name, ast in zip(all_files, all_asts)
               if len(ast) < args.max_tokens]

grouped = {}

for file in short_files:
    if args.contest_type and not file.startswith("src/{0}".format(args.contest_type)):
        continue
    grouped.setdefault(path.dirname(file), [])
    grouped[path.dirname(file)].append(file)

to_check = set()
for key in random.sample(list(grouped), args.projects_count):
    group = grouped[key]
    sample_count = min(random.randint(args.min_files, args.max_files), len(group))
    for file in random.sample(group, sample_count):
        to_check.add(file)

with open(args.output, "w") as f:
    for name in to_check:
        print(name, file=f)
