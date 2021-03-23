# eHealth-KD Corpora - Scripts

This folder contains scripts and tools to aid in the loading, manipulation, and writing of BRAT .ann files.

## Loading the training collection

The `anntools.py` contains a set of classes to load and manipulate BRAT files (Python 3.7 or higher is required).

Start by creating an empty collection:

```python
from anntools import Collection
from pathlib import Path

c = Collection()
```

You can load specific files (note that we pass the path of the `.txt` file):

```python
c.load(Path("/path/to/corpus/2021/training/medline.es.1200.txt"))
```

Or you can load all files at once:

```python
for fname in Path("/path/to/corpus/2021/training/").rglob("*.txt"):
    c.load(fname)
```

Now, you can inspect the contents of the collection. It contains a `sentence` list with `Sentence` objects.
These objects in turn contain the `Keyphrase`s and `Relation`s annotated in each sentence along with their metadata:

```python
>>> len(c.sentences)
1500
>>> c.sentences[0]
Sentence(text='La presencia del gen de células falciformes y otro normal se denomina rasgo drepanocítico.', keyphrases=[Keyphrase(text='presencia', label='Action', id=1, attr=[]), Keyphrase(text='gen', label='Concept', id=2, attr=[]), Keyphrase(text='gen de células falciformes', label='Concept', id=3, attr=[]), Keyphrase(text='normal', label='Concept', id=4, attr=[]), Keyphrase(text='rasgo drepanocítico', label='Concept', id=5, attr=[])], relations=[Relation(from='gen', to='normal', label='in-context'), Relation(from='presencia', to='gen de células falciformes', label='subject'), Relation(from='presencia', to='rasgo drepanocítico', label='same-as'), Relation(from='presencia', to='gen', label='subject')])
>>> c.sentences[100].keyphrases[2]
Keyphrase(text='bombear', label='Action', id=612, attr=[Attribute(label='Negated')])
>>> c.sentences[1000].relations[-1]
Relation(from='bebe', to='obtiene', label='causes')
```

You can also create a `Collection` manually (e.g., from the output of your entity recognition system) and produce an annotated file:

```python
c.dump(Path("output.txt"))
```

This will produce the `.txt` file and the corresponding `.ann` file with all relevant annotations, along with normalized IDs.

## Evaluating a single scenario

You can run the evaluation script offline just to check your results. The evaluation script is in the file score.py and the arguments are:

- The gold annotations (in this case, example_2020/development/main/scenario.txt).
- Your system’s annotations (example_2020/baseline/dev/run1/scenario1-main/scenario.txt)

The evaluation script outputs the total number of correct, incorrect, partial, missing and spurious matches for each subtask, and the final score as defined in the Task section.

```shell
$ python3.7 score.py \
    example_2020/development/main/scenario.txt \
    example_2020/baseline/dev/run1/scenario1-main/scenario.txt

correct_A: 813
incorrect_A: 75
partial_A: 65
spurious_A: 674
missing_A: 352
correct_B: 102
spurious_B: 256
missing_B: 1102
--------------------
recall: 0.3776
precision: 0.4773
f1: 0.4217
```
    
> NOTE: The exact numbers you see with the baseline may vary, as the evaluation script and/or the baseline implementation can suffer changes as we discover bugs or mistakes. These numbers are for illustrative purposes only. The actual scores are the ones published in Codalab.

The options --skip-A and --skip-B instruct the script to ignore the performance of the submission on subtask A and subtask B respectively (i.e. they will not directly impact the final score reported).

You can evaluate just scenario 2 with the evaluation script by passing --skip-B:

```shell
$ python3.7 score.py --skip-B \
    example_2020/development/main/scenario.txt \
    example_2020/baseline/dev/run1/scenario2-taskA/scenario.txt

correct_A: 813
incorrect_A: 75
partial_A: 65
spurious_A: 674
missing_A: 352
--------------------
recall: 0.6479
precision: 0.5197
f1: 0.5767
```

You can evaluate just scenario 3 with the evaluation script by passing --skip-A:

```shell
$ python3.7 score.py --skip-A \
    example_2020/development/main/scenario.txt \
    example_2020/baseline/dev/run1/scenario3-taskB/scenario.txt

correct_B: 107
spurious_B: 91
missing_B: 1097
--------------------
recall: 0.08887
precision: 0.5404
f1: 0.1526
```

Additionally, you can pass `--verbose` if you want to see detailed information about which keyphrases and relations were correct, missing, etc.

```shell
$ python3.7 score.py --verbose \
    example_2020/development/main/scenario.txt \
    example_2020/baseline/dev/run1/scenario1-main/scenario.txt

===================  MISSING_A   ===================

Keyphrase(text='enfrentar', label='Action', id=3)
Keyphrase(text='tubos', label='Concept', id=7)
Keyphrase(text='filtran', label='Action', id=10)
Keyphrase(text='limpian', label='Action', id=11)
Keyphrase(text='eliminando', label='Action', id=13)

... LOTS OF OUTPUT

===================  MISSING_B  ===================

Relation(from='producen', to='genes', label='subject')
Relation(from='producen', to='proteínas', label='target')
Relation(from='producen', to='correctamente', label='in-context')
Relation(from='trastorno', to='niño', label='target')
Relation(from='trastorno', to='genético', label='in-context')
Relation(from='producen', to='trastorno', label='causes')
Relation(from='producen', to='trastorno', label='causes')
--------------------
recall: 0.3776
precision: 0.4773
f1: 0.4217
```