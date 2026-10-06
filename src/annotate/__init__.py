"""Registry of all annotators. Add a new one by appending to a module's ANNOTATORS list."""
from src.annotate import action, cinematography, location, text_evidence, time_of_day

REGISTRY = (location.ANNOTATORS + time_of_day.ANNOTATORS + action.ANNOTATORS
            + cinematography.ANNOTATORS + text_evidence.ANNOTATORS)


def get(names: list[str] | None = None):
    if not names:
        return REGISTRY
    by_name = {a.name: a for a in REGISTRY}
    unknown = set(names) - set(by_name)
    if unknown:
        raise ValueError(f"Unknown annotators {unknown}. Available: {list(by_name)}")
    return [by_name[n] for n in names]
