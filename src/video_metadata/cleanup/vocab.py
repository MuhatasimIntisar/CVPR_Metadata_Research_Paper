"""Word lists and mapping tables for the rule-based cleanup (no models). Changing any table is a new cleanup version."""
from __future__ import annotations

# ---------------------------------------------------------------- appearance (used to tell people apart)
COLOURS = {"black", "white", "grey", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
           "tan", "beige", "navy", "maroon", "teal", "gold", "silver", "salmon", "cream", "olive", "burgundy",
           "turquoise", "magenta", "lavender", "khaki", "plaid", "striped", "patterned", "floral"}
ACCESSORIES = {"glasses", "sunglasses", "hat", "cap", "beanie", "headscarf", "hijab", "turban", "headband", "earrings",
               "necklace", "beard", "bearded", "mustache", "moustache", "bald", "blonde", "blond", "brunette",
               "redhead", "ponytail", "braids", "curly", "headset", "lanyard", "badge", "mask"}
CLOTHING = {"shirt", "t-shirt", "tshirt", "blouse", "top", "blazer", "jacket", "suit", "sweater", "cardigan", "hoodie",
            "dress", "skirt", "tie", "vest", "coat", "uniform", "apron", "jersey", "scarf", "robe", "gown", "cloak",
            "armor", "armour", "polo", "sweatshirt", "pants", "jeans", "tuxedo", "overalls", "costume"}
# weight of a shared appearance word: colours / accessories / hair identify a person, garment nouns only a little
WEIGHTS = {**{w: 1.0 for w in COLOURS | ACCESSORIES}, **{w: 0.5 for w in CLOTHING}}

MALE = {"man", "men", "boy", "gentleman", "male", "guy", "he", "his", "him", "father", "husband", "mr"}
FEMALE = {"woman", "women", "girl", "lady", "female", "she", "her", "mother", "wife", "mrs", "ms"}

# entity description says the person is not actually on screen
NOT_VISIBLE_RX = r"\b(mentioned|referred to|referenced|discussed|caller|on the (phone|line)|off[- ]?screen|not (visible|shown|seen|present))\b"
PLACE_RX = r"\b(location|place|city|town|village|neighbou?rhood|street|park|venue)\b( mentioned| named| referred)?"
ORG_RX = r"\b(photography|company|business|organi[sz]ation|department|council|committee|association|foundation|church|school|club|team|studio|llc|inc)\b"

# moment shows a screen (video call, slides, graphics): person detectors and framing heuristics do not apply
SCREEN_RX = (r"\b(zoom|webex|video ?call|video ?conference|teams meeting|video feeds?|video windows?|gallery view|"
             r"screen ?share|shared screen|split[- ]screen|picture[- ]in[- ]picture|slides?|presentation screen|"
             r"title card|graphic|countdown|digital interface|livestream screen)\b")

# ---------------------------------------------------------------- tags
TAG_GENERIC = {"woman", "women", "man", "men", "person", "people", "persons", "individual", "individuals", "group",
               "two", "three", "four", "several", "hair", "long", "short", "dark", "light", "seated", "sitting",
               "standing", "wearing", "of", "a", "the", "and", "with", "in", "boy", "girl", "lady", "guy"}
TAG_DROP_WORDS = COLOURS | ACCESSORIES | CLOTHING | TAG_GENERIC

# ---------------------------------------------------------------- relationships (closed list in level2.metadata_generator)
# predicate the model used -> allowed predicate; None = co-presence or no relation between two entities (not kept)
PREDICATE_MAP = {
    "conducts": "performs_with", "leads_orchestra": "performs_with", "accompanies": "plays_music_with",
    "plays_with": "plays_music_with", "plays_alongside": "plays_music_with",
    "looks_at": "watches", "observes": "watches", "views": "watches", "faces": None,
    "talks_with": "talks_to", "speaks_to": "talks_to", "speaks_with": "talks_to", "addresses": "talks_to",
    "converses_with": "talks_to", "chats_with": "talks_to", "asks": "talks_to", "questions": "talks_to",
    "responds_to": "talks_to", "answers": "talks_to", "replies_to": "talks_to", "greets": "talks_to",
    "dances_for": "performs_for", "sings_to": "performs_for", "sings_for": "performs_for", "plays_for": "performs_for",
    "performs_to": "performs_for", "entertains": "performs_for",
    "shows": "demonstrates_to", "shows_to": "demonstrates_to", "demonstrates": "demonstrates_to",
    "teaches_to": "teaches", "coaches": "instructs", "guides": "instructs", "directs": "instructs",
    "presents_to": "presents", "introduces_to": "introduces", "carries": "holds", "holds_up": "holds",
    "operates": "uses", "hands_to": "gives_to", "passes_to": "gives_to", "embraces": "hugs", "points_to": "points_at",
    "claps_for": "applauds",
    "sits_with": None, "sits_next_to": None, "sits_across_from": None, "sits_beside": None, "stands_next_to": None,
    "stands_with": None, "stands_beside": None, "next_to": None, "beside": None, "near": None, "with": None,
    "discusses": None, "helps": None, "assists": None, "smiles_at": None, "nods_to": None}

# ---------------------------------------------------------------- actions (open vocabulary; only synonyms merged)
ACTION_MAP = {"talk": "speak", "talking": "speak", "talk_to": "speak", "speak_to": "speak", "chat": "speak",
              "say": "speak", "address": "speak", "listen_to": "listen", "listening": "listen",
              "perform_music": "play_music", "play_instrument": "play_music", "play_musical_instrument": "play_music",
              "play_instruments": "play_music", "perform_concert": "play_music", "perform_orchestra": "play_music",
              "play_violin": "play_music", "play_piano": "play_music", "play_guitar": "play_music",
              "hold_paper": "hold", "hold_papers": "hold", "hold_object": "hold", "hold_microphone": "hold",
              "sit_down": "sit", "seated": "sit", "stand_up": "stand", "clap": "applaud", "wave_hand": "wave",
              "conduct_orchestra": "conduct", "conducting": "conduct"}
