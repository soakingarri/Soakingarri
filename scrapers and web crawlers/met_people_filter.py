"""Decide whether a MET Egyptian Art object is an original depiction of
ancient Egyptian people or gods, made under native rule.

Kept: complete statues, busts, figures, shabtis, coffins, masks, stelae, and
original wall/temple reliefs and paintings showing people or gods (these
mostly survive as fragments, so fragments are allowed for them only).

Excluded: heads and body-part fragments of statues, animals (unless a god),
belongings and inscribed-only objects, MET facsimiles and modern replicas,
modern repainting, and anything dated to foreign rule: Hyksos (Dyn. 15-16),
Libyan (Dyn. 22-24), Persian (Dyn. 27, 31), Ptolemaic, Roman and later.
Date ranges that overlap foreign rule are excluded too. Kushite Dynasty 25
counts as native.
"""
import re


def rx(words):
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.I)


# Always a depiction of a person or god.
DEPICTION = rx([
    "statues?", "statuettes?", "figures?", "figurines?", "busts?",
    "portraits?", "masks?", "sphinx(es)?", "u?shabtis?", "coffins?",
    "sarcophagus", "mumm(y|ies)", "dolls?", "stelae?", "stele",
    "paintings?", "scenes?",
])

# People, roles, gods and kings: a depiction unless the object is just
# something inscribed with the name.
PEOPLE = rx([
    # people and family
    "man", "men", "woman", "women", "lady", "ladies", "nobles",
    "workman", "attendants?", "child", "children", "boy", "girl",
    "couple", "family", "people", "son of", "daughter of", "wife of",
    "mother of", "father of", "brother", "sister", "wife", "husband",
    "daughters?", "sons", "parents",
    # royalty and officials
    "king", "queen", "prince", "princess", "pharaoh", "vizier", "official",
    "scribe", "steward", "overseer", "noble(man)?", "chancellor", "governor",
    "general", "chief",
    # priests, soldiers and workers
    "priests?", "priestess", "prophet", "god's wife", "soldiers?",
    "archers?", "troops", "workers?", "workmen", "craftsmen", "carpenters?",
    "sculptors?", "farmers?", "brewers?", "bakers?", "weavers?", "potters?",
    "butchers?", "fishermen", "hunters?", "porters?", "servants?",
    "offering bearers?", "musicians?", "dancers?", "harpists?", "singers?",
    "mourners?", "captives?", "prisoners?", "nubians?", "syrians?",
    "libyans?", "asiatics?", "foreigners?",
    # gods
    "gods?", "goddess(es)?", "deity", "deities", "hathor", "isis", "osiris",
    "horus", "amun", "amun-re", "re-harakhty", "ptah", "anubis", "thoth",
    "bastet", "sakhmet", "sekhmet", "neit", "bes", "taweret", "khonsu", "mut",
    "maat", "min", "sobek", "hapy", "nephthys", "seth", "harpocrates",
    "imhotep", "atum", "khnum", "wadjet", "nekhbet", "serapis",
    # pharaohs
    "senwosret", "amenemhat", "hatshepsut", "thutmose", "amenhotep",
    "akhenaten", "nefertiti", "tutankhamun", "ramesses", "seti",
    "mentuhotep", "sneferu", "khufu", "khafre", "menkaure", "taharqo",
    "psamtik", "nectanebo", "ptolemy", "cleopatra", "merneptah", "horemheb",
    "ay", "tiye", "ahmose", "sahure", "userkaf", "djoser", "pepi", "teti",
])

# Partial images: heads, body parts and fragments.
FRAGMENT = rx([
    "heads?", "faces?", "torso", "fragments?", "fragmentary", "part of",
    "parts", "upper part", "lower part", "base and feet", "knees?", "arms?",
    "hands?", "feet", "foot", "legs?", "ears?", "eyes?", "nose", "mouth",
    "wig", "shoulders?", "chest",
])

# Hybrid gods: "lion-headed goddess", "falcon-headed god", etc.
HYBRID_GOD = re.compile(r"\b\w+-headed\b", re.I)

GOD_NAMES = rx([
    "gods?", "goddess(es)?", "deity", "deities", "hathor", "isis", "osiris",
    "horus", "amun", "amun-re", "re-harakhty", "ptah", "anubis", "thoth",
    "bastet", "sakhmet", "sekhmet", "neit", "bes", "taweret", "khonsu", "mut",
    "maat", "min", "sobek", "hapy", "nephthys", "seth", "harpocrates",
    "imhotep", "atum", "khnum", "wadjet", "nekhbet", "serapis", "apis",
    "duamutef", "qebehsenuef", "imsety", "khepri", "montu", "meretseger",
])

ANIMALS = rx([
    "hippopotam(us|i)", "baboons?", "leopards?", "lions?", "lioness",
    "cats?", "falcons?", "hawks?", "cobras?", "snakes?", "serpents?",
    "uraeus", "uraei", "ibis(es)?", "crocodiles?", "dogs?", "hounds?",
    "jackals?", "bulls?", "cows?", "calf", "cattle", "rams?", "goats?",
    "gazelles?", "fish", "birds?", "ducks?", "geese", "goose", "frogs?",
    "hedgehogs?", "monkeys?", "scorpions?", "vultures?", "owls?", "horses?",
    "donkeys?", "mice", "mouse", "beetles?", "animals?", "jerboas?",
    "shrews?", "apes?", "ichneumons?", "antelopes?", "oryx", "ibex",
    "hares?", "giraffes?", "pigs?", "boars?", "turtles?", "tortoises?",
    "lizards?", "insects?", "flies", "grasshoppers?",
])

# Things owned by or made for a person, not pictures of them.
BELONGINGS = rx([
    "box(es)?", "mirrors?", "knife", "knives", "linen", "sheets?",
    "girdles?", "canopic", "headrests?", "sandals?", "palettes?", "chairs?",
    "beds?", "tools?", "shrine", "spoons?", "combs?", "pins?",
    "necklaces?", "bracelets?", "armlets?", "pectorals?", "earrings?",
    "base", "beard", "wrist guard", "quiver", "staff", "tambourine",
    "labels?", "tags?", "scarabs?", "seals?", "sealings?", "cones?",
    "tiles?", "rings?", "beads?", "amulets?", "cartouches?", "pot stand",
    "hieratic", "teaching", "copy of", "tablets?", "columns?", "lids?",
    "offering table", "pendants?", "coins?", "texts?", "thrones?",
    "implements?", "bench", "knobs?", "scaraboids?", "titles of",
    "burnishers?", "decrees?", "bottles?", "containers?",
])

# Objects that usually only carry a name or inscription.
INSCRIBED_ONLY = rx([
    "scarabs?", "cartouches?", "seals?", "sealings?", "cones?", "tiles?",
    "amulets?", "inscribed", "inscriptions?", "labels?", "blocks?", "rings?",
    "beads?", "jars?", "vessels?", "vases?", "bowls?", "cups?", "kohl",
    "plaques?", "bricks?", "weights?", "cowroids?", "name",
])

# MET facsimiles are modern copies, all on paper or canvas.
FACSIMILE = re.compile(r"facsimile", re.I)
ON_PAPER = re.compile(r"\b(paper|canvas)\b", re.I)

# Original wall/temple art: reliefs, wall/ceiling paintings, temple blocks.
WALL_ART_TYPE = rx([
    "relief", "talatat", "wall", "ceiling", "painting", "chapel",
    "burial chamber",
])

# Statues are often painted limestone too, so these are never wall art.
STATUE_LIKE = rx([
    "statues?", "statuettes?", "figures?", "figurines?", "busts?", "heads?",
    "masks?", "u?shabtis?", "coffins?", "sarcophagus", "stelae?", "stele",
    "torso",
])

# Foreign-rule dynasties: Hyksos (15-16), Libyan (22-24), Persian (27, 31).
FOREIGN_DYNASTIES = {15, 16, 22, 23, 24, 27, 31}
FOREIGN_PERIOD = re.compile(
    r"roman|ptolema|macedon|persian|islamic|byzantine|coptic|hyksos|"
    r"or later|twentieth century", re.I
)
# Broad periods that contain foreign dynasties; only allowed when a
# dynasty number places the object in a native reign.
MIXED_PERIOD = re.compile(
    r"late period|third intermediate|second intermediate", re.I
)
MODERN_PAINT = re.compile(r"modern", re.I)


def dynasty_range(dynasty):

    nums = [int(n) for n in re.findall(r"\d+", dynasty or "")]

    if not nums:
        return None

    if re.search(r"or later", dynasty or "", re.I):
        return min(nums), 99

    if "SIP" in (dynasty or ""):
        nums.append(17)

    return min(nums), max(nums)


def excluded_reason(medium, period, dynasty):

    if FOREIGN_PERIOD.search(period or ""):
        return "foreign or modern period"

    if MODERN_PAINT.search(medium or ""):
        return "modern repainting"

    span = dynasty_range(dynasty)

    if span:

        if any(span[0] <= d <= span[1] for d in FOREIGN_DYNASTIES):
            return "dynasty range includes foreign rule"

        return None

    if MIXED_PERIOD.search(period or ""):
        return "broad period with foreign rule, no dynasty given"

    return None


def is_facsimile(title, object_name, medium):

    return bool(
        FACSIMILE.search(title)
        or FACSIMILE.search(object_name)
        or ON_PAPER.search(medium)
    )


def is_wall_art(title, object_name):

    if STATUE_LIKE.search(title):
        return False

    return bool(WALL_ART_TYPE.search(f"{title} {object_name}"))


def title_matches(title):

    if FRAGMENT.search(title) or BELONGINGS.search(title):
        return False

    is_god = bool(GOD_NAMES.search(title) or HYBRID_GOD.search(title))

    if ANIMALS.search(title) and not is_god:
        return False

    if DEPICTION.search(title):
        return True

    return bool(PEOPLE.search(title) or is_god) and not INSCRIBED_ONLY.search(title)


def wall_art_matches(title, object_name):

    # Original wall art survives mostly as fragments, so fragments are kept
    # here as long as people or gods are named.
    text = f"{title} | {object_name}"

    if BELONGINGS.search(title) or INSCRIBED_ONLY.search(title):
        return False

    is_god = bool(GOD_NAMES.search(text) or HYBRID_GOD.search(text))

    if ANIMALS.search(title) and not (is_god or PEOPLE.search(text)):
        return False

    return bool(PEOPLE.search(text) or is_god or DEPICTION.search(title))


def depicts_people(title, object_name, medium, period, dynasty):

    title = title or ""
    object_name = object_name or ""
    medium = medium or ""

    if is_facsimile(title, object_name, medium):
        return False

    if excluded_reason(medium, period, dynasty):
        return False

    if is_wall_art(title, object_name):
        return wall_art_matches(title, object_name)

    return title_matches(title)
