CLASSNAMES = [
    "Normal", "Stroke", "Glioma", "Meningioma", "Metastasis"
]


TEMPLATES = {
    "template": (
        lambda c: f"This study shows: {c}.",
        lambda c: f"This study shows: {c} identified." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.", 
        lambda c: f"This study shows: {c} noted." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.", 
        lambda c: f"This study shows: {c} seen." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.", 
        lambda c: f"This study shows: new {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: known {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: prominent {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: likely {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: possibly {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: indicating {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: reflecting {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: representing {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: suggesting {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: indicative of {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: suggestive of {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: related to {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: consistent with {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
        lambda c: f"This study shows: compatible with {c}." if c != "no significant abnormalities" else "This study shows: no significant abnormalities.",
    ),
}


PROMPTS = {
    "prompt": ("no significant abnormalities", "acute stroke", "glioma", "meningioma", "metastasis"),
}