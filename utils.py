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

RELATIONS_INV = [
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
    "is-a_INV",
    "part-of_INV",
    "has-property_INV",
    "causes_INV",
    "entails_INV",
    "in-context_INV",
    "in-place_INV",
    "in-time_INV",
    "subject_INV",
    "target_INV",
    "domain_INV",
    "arg_INV",
]

entity_w2id = {w: i for i, w in enumerate(ENTITIES)}
entity_id2w = {i: w for i, w in enumerate(ENTITIES)}
relation_w2id = {w: i for i, w in enumerate(RELATIONS)}
relation_id2w = {i: w for i, w in enumerate(RELATIONS)}
relation_inv_w2id = {w: i for i, w in enumerate(RELATIONS_INV)}
relation_inv_id2w = {i: w for i, w in enumerate(RELATIONS_INV)}