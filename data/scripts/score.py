# coding: utf8

import sys
import argparse
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List

from anntools import Collection, DisjointSet, Sentence

CORRECT_A = "correct_A"
INCORRECT_A = "incorrect_A"
PARTIAL_A = "partial_A"
SPURIOUS_A = "spurious_A"
MISSING_A = "missing_A"

CORRECT_B = "correct_B"
SPURIOUS_B = "spurious_B"
MISSING_B = "missing_B"

SAME_AS = "same-as"


def report(data, verbose):
    for key, value in data.items():
        print("{}: {}".format(key, len(value)))

    if verbose:
        for key, value in data.items():
            if key.startswith("correct"):
                continue

            print(
                "\n==================={}===================\n".format(
                    key.upper().center(14)
                )
            )
            if isinstance(value, dict):
                print("\n".join("{} --> {}".format(x, y) for x, y in value.items()))
            else:
                print("\n".join(str(x) for x in value))


def subtaskA(gold, submit, verbose=False):
    return match_keyphrases(gold, submit)


def match_keyphrases(gold, submit, skip_incorrect=False):
    correct = {}
    incorrect = {}
    partial = {}
    spurious = []
    missing = []

    for gold_sent, submit_sent in align(gold.sentences, submit.sentences):
        # if gold_sent.text != submit_sent.text:
        #     warnings.warn(
        #         "Wrong sentence: gold='%s' vs submit='%s'"
        #         % (gold_sent.text, submit_sent.text)
        #     )
        #     continue

        if not gold_sent.keyphrases and not gold_sent.relations:
            continue

        gold_sent = gold_sent.clone(shallow=True)
        submit_sent = submit_sent.clone(shallow=True)

        # correct
        for keyphrase in submit_sent.keyphrases[:]:
            match = gold_sent.find_keyphrase(spans=keyphrase.spans)
            if match and match.label == keyphrase.label:
                correct[keyphrase] = match
                gold_sent.keyphrases.remove(match)
                submit_sent.keyphrases.remove(keyphrase)

        # incorrect
        for keyphrase in submit_sent.keyphrases[:]:
            if skip_incorrect:
                break

            match = gold_sent.find_keyphrase(spans=keyphrase.spans)
            if match:
                assert match.label != keyphrase.label
                incorrect[keyphrase] = match
                gold_sent.keyphrases.remove(match)
                submit_sent.keyphrases.remove(keyphrase)

        # partial
        for keyphrase in submit_sent.keyphrases[:]:
            match = find_partial_match(keyphrase, gold_sent.keyphrases)
            if match:
                partial[keyphrase] = match
                gold_sent.keyphrases.remove(match)
                submit_sent.keyphrases.remove(keyphrase)

        # spurious
        spurious.extend(submit_sent.keyphrases)

        # missing
        missing.extend(gold_sent.keyphrases)

    return {
        CORRECT_A: correct,
        INCORRECT_A: incorrect,
        PARTIAL_A: partial,
        SPURIOUS_A: spurious,
        MISSING_A: missing,
    }


def find_partial_match(keyphrase, sentence):
    return next(
        (
            match
            for match in sentence
            if match.label == keyphrase.label and partial_match(keyphrase, match)
        ),
        None,
    )


def partial_match(keyphrase1, keyphrase2):
    match = False
    match |= any(
        start <= x < end for start, end in keyphrase1.spans for x, _ in keyphrase2.spans
    )
    match |= any(
        start <= x < end for start, end in keyphrase2.spans for x, _ in keyphrase1.spans
    )
    return match


def subtaskB(gold, submit, data, verbose=False):
    return match_relations(gold, submit, data)


def normalize(s: str):
    return "".join(c.lower() for c in s if c.isalnum())


def compare_text(s1: str, s2: str):
    return normalize(s1) == normalize(s2)


def align(gold_sentences: List[Sentence], submit_sentences: List[Sentence]):
    gold_sentences: List[Sentence] = list(gold_sentences)
    submit_sentences: List[Sentence] = list(submit_sentences)

    while gold_sentences and submit_sentences:
        gold = gold_sentences[0]
        submit = submit_sentences[0]

        # si las oraciones coinciden, devolver ambas normalmente
        if compare_text(gold.text, submit.text):
            gold_sentences.pop(0)
            submit_sentences.pop(0)
            yield (gold, submit)
            continue

        # las oraciones no coinciden, asumiremos que en submit falta esta oración
        # generamos una oración sin anotar con el mismo text del gold
        submit = Sentence(gold.text)
        gold_sentences.pop(0)
    
        warnings.warn("Match not found for gold sentence: %r" % gold.text)
        yield (gold, submit)

    while gold_sentences:
        # todas estas oraciones faltan por anotar
        gold = gold_sentences.pop(0)
        warnings.warn("Match not found for gold sentence (submission ended): %r" % gold.text)
        yield (gold, Sentence(gold.text))

    while submit_sentences:
        submit = submit_sentences.pop(0)
        warnings.warn("Spurious submission sentence not considered (gold ended): %r" % submit.text)


def match_relations(gold, submit, data, skip_same_as=False, propagate_error=True):
    correct = {}
    spurious = []
    missing = []

    for gold_sent, submit_sent in align(gold.sentences, submit.sentences):
        # if gold_sent.text != submit_sent.text:
        #     warnings.warn(
        #         "Wrong sentence: gold='%s' vs submit='%s'"
        #         % (gold_sent.text, submit_sent.text)
        #     )
        #     continue

        if not gold_sent.keyphrases and not gold_sent.relations:
            continue

        gold_sent = gold_sent.clone(shallow=True)
        gold_sent.remove_dup_relations()

        submit_sent = submit_sent.clone(shallow=True)
        submit_sent.remove_dup_relations()

        equivalence = DisjointSet(*gold_sent.keyphrases)

        # build equivalence classes
        for relation in gold_sent.relations:
            if relation.label != SAME_AS:
                continue

            origin = relation.from_phrase
            destination = relation.to_phrase

            equivalence.merge([origin, destination])

        if skip_same_as:
            for relation in gold_sent.relations[:]:
                if relation.label == SAME_AS:
                    gold_sent.relations.remove(relation)
            for relation in submit_sent.relations[:]:
                if relation.label == SAME_AS:
                    submit_sent.relations.remove(relation)

        if not propagate_error:
            found = {**data[CORRECT_A], **data[PARTIAL_A]}
            for relation in submit_sent.relations[:]:
                if relation.from_phrase not in found or relation.to_phrase not in found:
                    submit_sent.relations.remove(relation)
            for relation in gold_sent.relations[:]:
                if (
                    relation.from_phrase not in found.values()
                    or relation.to_phrase not in found.values()
                ):
                    gold_sent.relations.remove(relation)

        # correct
        for relation in submit_sent.relations[:]:
            origin = relation.from_phrase
            origin = map_keyphrase(origin, data)

            destination = relation.to_phrase
            destination = map_keyphrase(destination, data)

            if origin is None or destination is None:
                continue

            match = gold_sent.find_relation(origin.id, destination.id, relation.label)
            if match is None and relation.label == SAME_AS:
                match = gold_sent.find_relation(
                    destination.id, origin.id, relation.label
                )

            if match is None:
                origin = equivalence[origin].representative.value
                destination = equivalence[destination].representative.value

                match = find_relation(
                    origin,
                    destination,
                    relation.label,
                    gold_sent.relations,
                    equivalence,
                )
                if match is None and relation.label == SAME_AS:
                    match = find_relation(
                        destination,
                        origin,
                        relation.label,
                        gold_sent.relations,
                        equivalence,
                    )

            if match:
                correct[relation] = match
                gold_sent.relations.remove(match)
                submit_sent.relations.remove(relation)

        # spurious
        spurious.extend(submit_sent.relations)

        # missing
        missing.extend(gold_sent.relations)

    return {
        CORRECT_B: correct,
        SPURIOUS_B: spurious,
        MISSING_B: missing,
    }


def map_keyphrase(keyphrase, data):
    try:
        return data[CORRECT_A][keyphrase]
    except KeyError:
        pass
    try:
        return data[PARTIAL_A][keyphrase]
    except KeyError:
        pass
    return None


def compute_metrics(data, skipA=False, skipB=False):
    correct = 0
    partial = 0
    incorrect = 0
    missing = 0
    spurious = 0

    if not skipA:
        correct += len(data[CORRECT_A])
        incorrect += len(data[INCORRECT_A])
        partial += len(data[PARTIAL_A])
        missing += len(data[MISSING_A])
        spurious += len(data[SPURIOUS_A])

    if not skipB:
        correct += len(data[CORRECT_B])
        missing += len(data[MISSING_B])
        spurious += len(data[SPURIOUS_B])

    recall_num = correct + 0.5 * partial
    recall_den = correct + partial + incorrect + missing
    recall = recall_num / recall_den if recall_den > 0 else 0.

    precision_num = correct + 0.5 * partial
    precision_den = correct + partial + incorrect + spurious
    precision = precision_num / precision_den if precision_den > 0 else 0.

    f1_num = 2 * recall * precision
    f1_den = recall + precision

    f1 = f1_num / f1_den if f1_den > 0 else 0.

    return {"recall": recall, "precision": precision, "f1": f1}


def find_relation(origin, destination, label, target_relations, target_equivalence):
    for relation in target_relations:
        if relation.label != label:
            continue
        target_origin = relation.from_phrase
        target_origin = target_equivalence[target_origin].representative.value

        target_destination = relation.to_phrase
        target_destination = target_equivalence[target_destination].representative.value

        if target_origin == origin and target_destination == destination:
            return relation
    return None

SCENARIOS = {
    1: ("scenario1-main", False, False),
    2: ("scenario2-taskA", False, True),
    3: ("scenario3-taskB", True, False),
}

def main(gold: Path, submit: Path, verbose:bool, scenarios: List[int], runs: List[int], prefix:str):
    runs_data = {}

    for run in runs:
        run_data = {}

        if not (submit / f"run{run}").exists():
            print(f"Run {run} not found!")
            continue

        for id in scenarios:
            folder, skipA, skipB = SCENARIOS[id]

            print(f"Scoring scenario {id} on run {run}:\n")
            run_data[folder.split("-")[0]] = main_scenario(gold / folder / "output.txt", submit / f"run{run}" / folder / "output.txt", skipA, skipB, verbose)
            print()

        runs_data[f"run{run}"] = run_data

    print()
    report_main(runs_data, prefix)


def report_main(runs_data, prefix):
    keys = { f"scenario{s}_{metric}":0 for s in [1,2,3] for metric in ["f1", "precision", "recall", "best"] }
    
    for run_id, run_data in runs_data.items():
        for scn_id, scn_data in run_data.items():
            if scn_data["f1"] <= keys[f"{scn_id}_f1"]:
                pass

            for metric in scn_data:
                keys[f"{scn_id}_{metric}"] = scn_data[metric]
            
            keys[f"{scn_id}_best"] = run_id

    for k,v in keys.items():
        print(f"{prefix}{k}:{v}", file=sys.stderr)


def main_scenario(gold_input, submit_input, skip_A, skip_B, verbose):
    gold = Collection()
    gold.load(gold_input)

    submit = Collection()
    submit.load(submit_input)

    data = OrderedDict()

    dataA = subtaskA(gold, submit, verbose)
    data.update(dataA)
    if not skip_A:
        report(dataA, verbose)

    if not skip_B:
        dataB = subtaskB(gold, submit, dataA, verbose)
        data.update(dataB)
        report(dataB, verbose)

    print("-" * 20)

    metrics = compute_metrics(data, skip_A, skip_B)

    for key, value in metrics.items():
        print("{0}: {1:0.4}".format(key, value))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path, help="Location of the reference data.")
    parser.add_argument("--submit", required=True, type=Path, help="Location of the submission.")
    parser.add_argument("--scenarios", type=int, nargs="+", help="Which scenarios to evaluate.", default=[1,2,3])
    parser.add_argument("--runs", type=int, nargs="+", help="Which runs to evaluate.", default=[1,2,3])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prefix", help="Prefix to report scores, e.g., 'training_'...", default="")
    
    args = parser.parse_args()
    
    main(Path(args.gold), Path(args.submit), args.verbose, args.scenarios, args.runs, args.prefix)
