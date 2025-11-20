CLASSNAMES = [
    "Any", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural"
]


TEMPLATES = {
    "template": (
        lambda c: f"This study shows: {c}.",
        lambda c: f"This study shows: {c} identified.", 
        lambda c: f"This study shows: {c} noted.", 
        lambda c: f"This study shows: {c} seen.", 
        lambda c: f"This study shows: new {c}.",
        lambda c: f"This study shows: known {c}.",
        lambda c: f"This study shows: prominent {c}.",
        lambda c: f"This study shows: likely {c}.",
        lambda c: f"This study shows: possibly {c}.",
        lambda c: f"This study shows: indicating {c}.",
        lambda c: f"This study shows: reflecting {c}.",
        lambda c: f"This study shows: representing {c}.",
        lambda c: f"This study shows: suggesting {c}.",
        lambda c: f"This study shows: indicative of {c}.",
        lambda c: f"This study shows: suggestive of {c}.",
        lambda c: f"This study shows: related to {c}.",
        lambda c: f"This study shows: consistent with {c}.",
        lambda c: f"This study shows: compatible with {c}.",
    ),
}


PROMPTS = {
    "prompt": (
        "intracranial hemorrhage", 
        "intraparenchymal hemorrhage", 
        "intraventricular hemorrhage", 
        "subarachnoid hemorrhage", 
        "subdural hemorrhage"
    ),
}