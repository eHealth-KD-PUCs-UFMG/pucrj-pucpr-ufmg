ENTITIES = ["O", "Concept", "Action", "Predicate", "Reference"]

RELATIONS = [
    # "O",
    "is-a",
    "part-of",
    "has-property",
    "causes",
    "entails",
    "in-context",
    "in-place",
    "in-time",
    "subject",
    "target",
    "domain",
    "arg",
]

entity_w2id = {w: i for i, w in enumerate(ENTITIES)}
entity_id2w = {i: w for i, w in enumerate(ENTITIES)}
relation_w2id = {w: i for i, w in enumerate(RELATIONS)}
relation_id2w = {i: w for i, w in enumerate(RELATIONS)}