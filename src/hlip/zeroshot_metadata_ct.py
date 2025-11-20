CLASSNAMES = [
    "acute ischemic stroke", "acute subdural hematoma", "airway obstruction", "aneurysm clip", "aneurysm coil", 
    "aneurysmal subarachnoid hemorrhage", "arachnoid cyst", "arterial dissection", "basal ganglia calcification", "brain contusion", 
    "brain herniation", "brain lead", "brain mass effect", "burr hole", "carotid cavernous fistula", 
    "catheter", "cavernous malformation cavernoma", "cerebral abscess", "cerebral arteriovenous malformation", "cerebral atrophy", 
    "cerebral edema", "cerebral foreign body", "cerebral subdural empyema", "cerebral venous sinus thrombosis", "chiari malformation", 
    "cholesteatoma", "colloid cyst", "craniofacial injury", "cranioplasty implant", "craniotomy", 
    "dandy walker malformation", "diffuse axonal injury", "displaced skull fracture", "dysgenesis corpus callosum", "embolization material", 
    "encephalomalacia gliosis", "epidural hematoma", "extra axial brain tumor", "head neck enlarged lymph node", "head neck tumor", 
    "intra axial brain tumor", "intracranial aneurysm", "intracranial atherosclerosis", "intracranial epidermoid dermoid cyst", "intracranial hemorrhage", 
    "intracranial hypotension", "intracranial pressure monitor", "intraparenchymal hemorrhage", "intraventricular hemorrhage", "large vessel occlusion", 
    "mastoid effusion", "mega cisterna magna", "midline shift", "nondisplaced skull fracture", "obstructive hydrocephalus", 
    "orbital emphysema", "orbital trauma", "orbital tumor", "otitis media", "parathyroid nodule", 
    "peritonsillar abscess", "pharyngeal abscess", "pineal tumor", "pituitary tumor", "pneumocephalus", 
    "posterior fossa tumors", "postsurgical changes", "resection cavity", "scalp hemorrhage hematoma", "sinusitis", 
    "skull base fracture", "skull bone tumor", "slit ventricle", "small vessel ischemic disease", "spinal degenerative changes", 
    "spine tumor", "subacute chronic subdural hematoma", "subdural hygroma effusions", "thyroid nodule", "transependymal flow", 
    "traumatic subarachnoid hemorrhage", "ventriculomegaly"
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
        "acute ischemic stroke", "acute subdural hematoma", "airway obstruction", "aneurysm clip", "aneurysm coil", 
        "aneurysmal subarachnoid hemorrhage", "arachnoid cyst", "arterial dissection", "basal ganglia calcification", "brain contusion", 
        "brain herniation", "brain lead", "brain mass effect", "burr hole", "carotid cavernous fistula", 
        "catheter", "cavernous malformation cavernoma", "cerebral abscess", "cerebral arteriovenous malformation", "cerebral atrophy", 
        "cerebral edema", "cerebral foreign body", "cerebral subdural empyema", "cerebral venous sinus thrombosis", "chiari malformation", 
        "cholesteatoma", "colloid cyst", "craniofacial injury", "cranioplasty implant", "craniotomy", 
        "dandy walker malformation", "diffuse axonal injury", "displaced skull fracture", "dysgenesis corpus callosum", "embolization material", 
        "encephalomalacia gliosis", "epidural hematoma", "extra axial brain tumor", "head neck enlarged lymph node", "head neck tumor", 
        "intra axial brain tumor", "intracranial aneurysm", "intracranial atherosclerosis", "intracranial epidermoid dermoid cyst", "intracranial hemorrhage", 
        "intracranial hypotension", "intracranial pressure monitor", "intraparenchymal hemorrhage", "intraventricular hemorrhage", "large vessel occlusion", 
        "mastoid effusion", "mega cisterna magna", "midline shift", "nondisplaced skull fracture", "obstructive hydrocephalus", 
        "orbital emphysema", "orbital trauma", "orbital tumor", "otitis media", "parathyroid nodule", 
        "peritonsillar abscess", "pharyngeal abscess", "pineal tumor", "pituitary tumor", "pneumocephalus", 
        "posterior fossa tumors", "postsurgical changes", "resection cavity", "scalp hemorrhage hematoma", "sinusitis", 
        "skull base fracture", "skull bone tumor", "slit ventricle", "small vessel ischemic disease", "spinal degenerative changes", 
        "spine tumor", "subacute chronic subdural hematoma", "subdural hygroma effusions", "thyroid nodule", "transependymal flow", 
        "traumatic subarachnoid hemorrhage", "ventriculomegaly"
    ),
}