# Application-level prompt rules. These settings do not teach the model new concepts.
# Turning a category on only permits the prompt through this local rules layer;
# the model can generate it only if its training data and learned capability support it.

CONTENT_RULES = {
    "nudity": False,
    "violence": True,
    "horror": True,
    "gore": False,
}

# These protections are intentionally not user-disableable.
NON_CONFIGURABLE_BLOCKS = (
    "sexual content involving minors",
    "non-consensual sexual imagery",
    "sexual exploitation",
)

def check_prompt(prompt: str) -> tuple[bool, str]:
    text = prompt.lower()
    if any(term in text for term in (
        "sexualized child", "child sexual", "underage nude",
        "non-consensual intimate", "revenge porn",
    )):
        return False, "Prompt blocked by a non-configurable safety rule."

    # Simple keyword checks are only a starter. They are not a reliable classifier.
    keyword_groups = {
        "nudity": ("nude", "nudity", "裸"),
        "violence": ("violent", "blood", "fight", "weapon"),
        "horror": ("horror", "monster", "terrifying"),
        "gore": ("gore", "dismember", "graphic injury"),
    }
    for category, terms in keyword_groups.items():
        if not CONTENT_RULES.get(category, False) and any(t in text for t in terms):
            return False, f"Prompt blocked: {category} is disabled in rules.py."
    return True, "Allowed by the current local rules."
