CLASSNAMES = [
    "acute ischemic stroke", "aneurysmal subarachnoid hemorrhage", "arachnoid cyst", "arteriovenous malformation", "brain abscess", 
    "brain contusion", "brain herniation", "brain mass effect", "brain metastasis", "brain midline shift", 
    "brain tumor", "brainstem glioma", "catheter", "cavernous malformation cavernoma", "cavum septum pellucidum", 
    "cephaloceles", "cerebral aneurysm", "cerebral atrophy", "cerebral edema", "chiari malformation", 
    "chordoma chondrosarcoma", "chronic ischemic stroke", "colloid cyst", "craniopharyngioma", "craniotomy craniectomy", 
    "dandy walker malformation", "diffuse axonal injury", "dysgenesis corpus callosum", "encephalomalacia gliosis", "epidermoid cyst", 
    "epidural hematoma", "glioma", "head neck tumor", "heterotopia", "high grade glioma", 
    "hydrocephalus ex vacuo", "intracranial hemorrhage", "intracranial hypotension", "intraparenchymal hemorrhage", "intraventricular hemorrhage", 
    "intraventricular tumor", "lacunar stroke", "low grade glioma", "lymphoma", "mega cisterna magna", 
    "meningioma", "moya moya disease", "multiple sclerosis", "nerve tumor", "neuromyelitis optica", 
    "neurosarcoidosis", "obstructive hydrocephalus", "orbital tumor", "pachygyria lissencephaly", "pediatric posterior fossa tumor", 
    "pineal cyst", "pineal tumor", "pituitary tumor", "pneumocephalus", "postsurgical changes", 
    "rathkes cleft cyst", "schizencephaly", "schwannoma", "small vessel ischemic disease", "spinal degenerative changes", 
    "spine syrinx", "spine tumor", "subdural hematoma", "subdural hygroma effusions", "traumatic subarachnoid hemorrhage", 
    "tumor resection cavity", "venous sinus thrombosis", "ventriculomegaly", "viral encephalitis"
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
        "acute ischemic stroke", "aneurysmal subarachnoid hemorrhage", "arachnoid cyst", "arteriovenous malformation", "brain abscess", 
        "brain contusion", "brain herniation", "brain mass effect", "brain metastasis", "brain midline shift", 
        "brain tumor", "brainstem glioma", "catheter", "cavernous malformation cavernoma", "cavum septum pellucidum", 
        "cephaloceles", "cerebral aneurysm", "cerebral atrophy", "cerebral edema", "chiari malformation", 
        "chordoma chondrosarcoma", "chronic ischemic stroke", "colloid cyst", "craniopharyngioma", "craniotomy craniectomy", 
        "dandy walker malformation", "diffuse axonal injury", "dysgenesis corpus callosum", "encephalomalacia gliosis", "epidermoid cyst", 
        "epidural hematoma", "glioma", "head neck tumor", "heterotopia", "high grade glioma", 
        "hydrocephalus ex vacuo", "intracranial hemorrhage", "intracranial hypotension", "intraparenchymal hemorrhage", "intraventricular hemorrhage", 
        "intraventricular tumor", "lacunar stroke", "low grade glioma", "lymphoma", "mega cisterna magna", 
        "meningioma", "moya moya disease", "multiple sclerosis", "nerve tumor", "neuromyelitis optica", 
        "neurosarcoidosis", "obstructive hydrocephalus", "orbital tumor", "pachygyria lissencephaly", "pediatric posterior fossa tumor", 
        "pineal cyst", "pineal tumor", "pituitary tumor", "pneumocephalus", "postsurgical changes", 
        "rathkes cleft cyst", "schizencephaly", "schwannoma", "small vessel ischemic disease", "spinal degenerative changes", 
        "spine syrinx", "spine tumor", "subdural hematoma", "subdural hygroma effusions", "traumatic subarachnoid hemorrhage", 
        "tumor resection cavity", "venous sinus thrombosis", "ventriculomegaly", "viral encephalitis"
    ),
}