from collections import Counter

# Basic categories. Expand anytime.
FURNITURE = {"chair", "couch", "bed", "dining table", "tv"}
DEVICES = {"cell phone", "laptop", "keyboard", "mouse", "remote"}
KITCHEN = {"bottle", "cup", "bowl", "fork", "knife", "spoon"}
PEOPLE = {"person"}
PETS = {"cat", "dog"}

def bucket(label: str) -> str:
    if label in PEOPLE:
        return "people"
    if label in PETS:
        return "pets"
    if label in FURNITURE:
        return "furniture"
    if label in DEVICES:
        return "devices"
    if label in KITCHEN:
        return "kitchen"
    return "other"

def summarize(detections):
    labels = [d["label"] for d in detections if d.get("confidence", 1.0) >= 0.25]
    if not labels:
        return {"text": "I do not see any clear objects right now.", "counts": {}, "buckets": {}}

    c = Counter(labels)
    # Top 6 objects by frequency
    top = [name for name, _ in c.most_common(6)]

    b = Counter(bucket(x) for x in labels)
    text = "I see " + ", ".join(top) + "."
    return {"text": text, "counts": dict(c), "buckets": dict(b)}
