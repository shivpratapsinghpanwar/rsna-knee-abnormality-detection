"""Weak-label extraction from multilingual free-text radiology reports.

Only ~1.3% of training studies carry gold labels, but every study has a report.
This module turns reports into per-target weak labels, which become the primary
training signal for the image model. Accuracy here matters more than model
capacity, so the lexicon is validated against the 58 gold-labelled studies.

Text is matched after accent-stripping and lower-casing, so Latin-script
patterns are written unaccented. Cyrillic patterns are matched as-is.
"""

from __future__ import annotations

import re
import unicodedata

from .labels import TARGETS

# --- normalisation ----------------------------------------------------------

_TR_MAP = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "ğ": "g"})


def normalize(text: str) -> str:
    """Lower-case, strip diacritics, collapse whitespace. Cyrillic is preserved."""
    if not isinstance(text, str):
        return ""
    text = text.translate(_TR_MAP).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


# --- shared vocabulary ------------------------------------------------------

# Pathology terms indicating a tear or injury, across the observed languages.
TEAR = (
    r"(tear|torn|rupture|ruptur|disrupt|discontinu|partial thickness|full thickness"
    r"|rotura|ruptura|desgarro|roto|lesion|lesao|rottura|lacerazione"
    r"|riss|einriss|laesion"
    r"|dechirure|fissuration"
    r"|scheur|ruptuur"
    r"|yirtik|rupturu|yirtilma"
    r"|разрыв|повреждени|надрыв)"
)

DEGEN = r"(degener|maceration|mucoid|intrasubstance|degenerativ|degenerac)"

# Negation cues; searched in the text immediately preceding a match.
#
# Every cue is word-bounded. Without \b, short cues match inside ordinary words --
# Spanish "menisco interno y" contains the substring "no ", which silently negated
# the following finding. interno/externo are pervasive in Romance-language reports,
# so an unbounded "no " corrupts a large fraction of the corpus.
NEGATION = (
    r"(\bno\b|\bnot\b|\bwithout\b|\babsence of\b|\bnegative for\b|\bfree of\b"
    r"|\bunremarkable\b|\bintact\b|\bnormal\w*\b|\bno evidence\b"
    r"|\bsin\b|\bausencia\b|\bintegro\b|\bintegra\b|\bconservad\w*\b|\bnormales\b"
    r"|\bkeine?\b|\bohne\b|\bunauffaellig\w*\b|\bintakt\b|\bregelrecht\w*\b"
    r"|\bpas de\b|\bsans\b|\babsence\b"
    r"|\bgeen\b|\bzonder\b|\bintacte\b"
    r"|\byok\b|\byoktur\b|\bizlenmedi\b|\bizlenmemistir\b|\bsaglam\b|\bdogaldir\b"
    r"|\bнет\b|\bбез\b|\bне выявлен\w*\b|\bне определя\w*\b|\bинтакт\w*\b|\bсохранен\w*\b"
    r"|\bδεν\b|\bχωρις\b|\bακεραι\w*\b|\bφυσιολογικ\w*\b)"
)

# How far back to look for a negation cue, and how far to look for a finding term.
NEG_WINDOW = 60
FIND_WINDOW = 90

_OA = (
    r"(osteoarthritis|osteoarthrosis|arthrosis|arthrose|artrosis|artrose|artrosi"
    r"|gonarthrose|gonartrosis|degenerative joint|chondropath|chondromalac"
    r"|cartilage loss|joint space narrowing|osteofit|osteophyt|osteofito"
    r"|artroz|остеоартр|артроз|хондромаляц)"
)

_SIDE = {
    "medial": r"(medial|interno|interne|innen|mediale|медиальн|внутренн)",
    "lateral": r"(lateral|externo|externe|aussen|laterale|латеральн|наружн)",
}

_ACL = (
    r"(\bacl\b|\blca\b|\bvkb\b|\bocb\b|\bpks\b"
    r"|anterior cruciate|cruzado anterior|crociato anteriore"
    r"|vorderes kreuzband|vorderen kreuzband"
    r"|croise anterieur|voorste kruisband|on capraz bag"
    r"|передней крестообразн|передняя крестообразн)"
)

_MCL = (
    r"(\bmcl\b|\blcm\b|\bmkl\b"
    r"|medial collateral|colateral medial|colateral interno|collaterale mediale"
    r"|innenband|mediales kollateralband|collateral interne|mediale collaterale"
    r"|ic yan bag"
    r"|внутренней боков|медиальной коллатеральн)"
)

_MED_MEN = (
    r"(medial meniscus|menisco interno|menisco medial|menisque interne"
    r"|menisque medial|innenmeniskus|medialer meniskus|mediale meniscus"
    r"|ic menisku|medial menisku|menisco mediale"
    r"|медиального мениск|внутреннего мениск)"
)

_LAT_MEN = (
    r"(lateral meniscus|menisco externo|menisco lateral|menisque externe"
    r"|menisque lateral|aussenmeniskus|lateraler meniskus|laterale meniscus"
    r"|dis menisku|lateral menisku|menisco laterale"
    r"|латерального мениск|наружного мениск)"
)

_PF = (
    r"(patellofemoral|femoropatelar|femoropatellar|retropatellar|patelofemoral"
    r"|femoro-patellaire|femoropatellare|пателлофеморальн)"
)

# target -> (anchor regex, required finding regex or None when the anchor is the finding)
ANCHORS: dict[str, tuple[str, str | None]] = {
    "ACL": (_ACL, TEAR),
    "MCL": (_MCL, TEAR),
    "Medial Meniscus": (_MED_MEN, "(" + TEAR + "|" + DEGEN + ")"),
    "Lateral Meniscus": (_LAT_MEN, "(" + TEAR + "|" + DEGEN + ")"),
    "PF OA": (_PF, _OA),
    "Effusion": (
        r"(joint effusion|effusion|derrame|erguss|gelenkerguss|epanchement"
        r"|vocht|effusie|efuzyon|eklem sivisi"
        r"|выпот|синовиальной жидкост)",
        None,
    ),
    "Synovitis": (
        r"(synovitis|sinovitis|synovialitis|synovite|sinovit|синовит)",
        None,
    ),
    "Baker's": (
        r"(baker|popliteal cyst|quiste de baker|quiste poplite|bakerzyste"
        r"|poplitealzyste|kyste de baker|kyste poplite|baker kisti|popliteal kist"
        r"|киста бейкера|подколенной кист)",
        None,
    ),
    "Contusion": (
        r"(contusion|bone bruise|bone marrow edema|bone marrow oedema|marrow edema"
        r"|contusao|edema oseo|edema de medula"
        r"|knochenmarkodem|knochenmarksodem|kontusion"
        r"|oedeme osseux|botcontusie|beenmergoedeem"
        r"|kontuzyon|kemik ilig odem"
        r"|ушиб|отек костного мозга|контузи)",
        None,
    ),
    "Fracture": (
        r"(fracture|fractura|fratura|frattura|fraktur|breuk|kirik|avulsion"
        r"|перелом)",
        None,
    ),
}

_COMPARTMENT_OA = {"Medial OA": "medial", "Lateral OA": "lateral"}


def _negated(text: str, start: int) -> bool:
    """True if a negation cue appears shortly before position `start`."""
    window = text[max(0, start - NEG_WINDOW):start]
    return re.search(NEGATION, window) is not None


def _score_simple(text: str, anchor: str, finding: str | None) -> float:
    """1.0 if an affirmed mention of the target exists, else 0.0."""
    for m in re.finditer(anchor, text):
        if finding is not None:
            window = text[max(0, m.start() - FIND_WINDOW): m.end() + FIND_WINDOW]
            if not re.search(finding, window):
                continue
        if not _negated(text, m.start()):
            return 1.0
    return 0.0


def _score_compartment_oa(text: str, side: str) -> float:
    """OA is compartment-specific: require the OA term and the side in one clause."""
    side_re = _SIDE[side]
    for clause in re.split(r"[.;\n]", text):
        m = re.search(_OA, clause)
        if m and re.search(side_re, clause) and not _negated(clause, m.start()):
            return 1.0
    return 0.0


def label_report(report: str) -> dict[str, float]:
    """Extract weak labels for all 12 targets from one report."""
    text = normalize(report)
    out: dict[str, float] = {}
    for target in TARGETS:
        if target in _COMPARTMENT_OA:
            out[target] = _score_compartment_oa(text, _COMPARTMENT_OA[target])
        else:
            anchor, finding = ANCHORS[target]
            out[target] = _score_simple(text, anchor, finding)
    return out
