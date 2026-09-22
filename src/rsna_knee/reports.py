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
    r"|разрыв|повреждени|надрыв"
    r"|ρηξ\w*|ρηγμα|διατομ\w*)"
)

DEGEN = (
    r"(degener|maceration|mucoid|intrasubstance|degenerativ|degenerac"
    r"|dejenerasyon|εκφυλ\w*)"
)

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

# Degenerative / cartilage findings. Gold PF-OA cases are frequently written as a
# chondral defect rather than the word "arthrosis", so cartilage vocabulary has to
# count. "(?<!sub)" keeps "subchondral" (e.g. subchondral fracture oedema) from
# firing, since that is not itself a degenerative finding.
_OA = (
    r"(osteoarthritis|osteoarthrosis|arthrosis|arthrose|artrosis|artrose|artrosi"
    r"|gonarthrose|gonartrosis|gonartroz|degenerative joint|joint space narrowing"
    r"|chondropath|chondromalac|condropat|condromalac|chondrosis|condrosis"
    r"|(?<!sub)chondral|(?<!sub)condral|cartilage loss|cartilage defect"
    r"|cartilage thinning|chondral ulcer|ulcera condral|perdida de cartilago"
    r"|adelgazamiento del cartilago|desgaste del cartilago"
    r"|knorpelschaden|knorpeldefekt|knorpelverschmaelerung|knorpelglatze"
    r"|kraakbeen|kikirdak kayb|kikirdak incelme"
    r"|osteofit|osteophyt|osteofito|osteofiet"
    r"|artroz|остеоартр|артроз|хондромаляц|хондральн|остеофит|истончение хряща"
    r"|αρθριτιδ|αρθρωσ|χονδροπαθ|χονδρομαλακ|χονδρ|οστεοφυτ|αποψιλωσ)"
)

# Compartment anchors. Specific phrases first, bare side words as a fallback --
# the OA term must co-occur in the same clause, which keeps bare "medial" from
# firing on e.g. "medial meniscus tear".
_SIDE = {
    "medial": (
        r"(medial compartment|compartimento medial|compartimento interno"
        r"|femorotibial medial|femorotibial interno|medial femoral condyle"
        r"|medial tibial plateau|condilo femoral medial|condilo femoral interno"
        r"|platillo tibial medial|innenkompartiment|mediale kompartiment"
        r"|\bmedial\w*|\binterno\b|\binterne\b|\binnen\w*|\bmediale\b"
        r"|εσω|μηριαιου κονδυλου|медиальн\w*|внутренн\w*)"
    ),
    "lateral": (
        r"(lateral compartment|compartimento lateral|compartimento externo"
        r"|femorotibial lateral|femorotibial externo|lateral femoral condyle"
        r"|lateral tibial plateau|condilo femoral lateral|condilo femoral externo"
        r"|platillo tibial lateral|aussenkompartiment|laterale kompartiment"
        r"|\blateral\w*|\bexterno\b|\bexterne\b|\baussen\w*|\blaterale\b"
        r"|εξω|латеральн\w*|наружн\w*)"
    ),
    "pf": (
        r"(patellofemoral|patelofemoral|femoropatelar|femoropatellar"
        r"|femoro-patellaire|femoropatellare|retropatellar|retropatelar"
        r"|\bpatellar\w*|\bpatella\b|\bpatellae\b|\brotulian\w*|\brotula\b"
        r"|troclea|trochlea|trochlear|kneecap|femoropatellar"
        r"|επιγονατιδ|пателлофеморальн\w*|надколенник\w*)"
    ),
}

_ACL = (
    r"(\bacl\b|\blca\b|\bvkb\b|\bocb\b|\bpks\b"
    r"|anterior cruciate|cruzado anterior|crociato anteriore"
    r"|vorderes kreuzband|vorderen kreuzband"
    r"|croise anterieur|voorste kruisband|on capraz bag"
    r"|передней крестообразн|передняя крестообразн"
    r"|προσθι\w* χιαστ\w*)"
)

_MCL = (
    r"(\bmcl\b|\blcm\b|\bmkl\b"
    r"|medial collateral|colateral medial|colateral interno|collaterale mediale"
    r"|innenband|mediales kollateralband|collateral interne|mediale collaterale"
    r"|ic yan bag"
    r"|внутренней боков|медиальной коллатеральн"
    r"|εσω πλαγι\w*)"
)

_MED_MEN = (
    r"(medial meniscus|menisco interno|menisco medial|menisque interne"
    r"|menisque medial|innenmeniskus|medialer meniskus|mediale meniscus"
    r"|ic menisku|medial menisku|menisco mediale"
    r"|медиального мениск|внутреннего мениск"
    r"|εσω μηνισκ\w*)"
)

_LAT_MEN = (
    r"(lateral meniscus|menisco externo|menisco lateral|menisque externe"
    r"|menisque lateral|aussenmeniskus|lateraler meniskus|laterale meniscus"
    r"|dis menisku|lateral menisku|menisco laterale"
    r"|латерального мениск|наружного мениск"
    r"|εξω μηνισκ\w*)"
)

# target -> (anchor regex, required finding regex or None when the anchor is the finding)
# The three OA targets are not listed here; they need compartment attribution and
# are handled by _score_compartment_oa.
ANCHORS: dict[str, tuple[str, str | None]] = {
    "ACL": (_ACL, TEAR),
    "MCL": (_MCL, TEAR),
    "Medial Meniscus": (_MED_MEN, "(" + TEAR + "|" + DEGEN + ")"),
    "Lateral Meniscus": (_LAT_MEN, "(" + TEAR + "|" + DEGEN + ")"),
    "Effusion": (
        r"(joint effusion|effusion|joint fluid|fluid in the joint|articular fluid"
        r"|suprapatellar fluid|joint distension|hydrops|hydrarthrosis"
        r"|derrame|liquido articular|aumento de liquido"
        r"|erguss|gelenkerguss|gelenkfluessigkeit|gelenkflussigkeit"
        r"|epanchement|liquide articulaire"
        r"|vocht|gewrichtsvocht|effusie"
        r"|efuzyon|eklem sivisi|eklem ici sivi|sivi artis|sivi miktari|hidrartroz"
        r"|выпот|жидкост\w* в полости|синовиальн\w* жидкост"
        r"|αρθρικο υγρο|υγρου|συλλογ\w*)",
        None,
    ),
    "Synovitis": (
        r"(synovitis|synovial thickening|synovial proliferation|synovial hypertroph"
        r"|pannus|sinovitis|engrosamiento sinovial|hipertrofia sinovial"
        r"|synovialitis|synovialis hypertroph|synoviale verdickung"
        r"|synovite|epaississement synovial"
        r"|synoviale verdikking|sinovit|sinovyal kalinlasma"
        r"|синовит|утолщение синовиальн\w*"
        r"|υμενιτιδ\w*|υμενα παχυνσ|παχυνση του αρθρικου υμενα)",
        None,
    ),
    "Baker's": (
        r"(baker|popliteal cyst|quiste de baker|quiste poplite|bakerzyste"
        r"|poplitealzyste|kyste de baker|kyste poplite|baker kisti|popliteal kist"
        r"|киста бейкера|подколенной кист"
        r"|ιγνυακη κυστ\w*|κυστη baker|κυστη του baker)",
        None,
    ),
    "Contusion": (
        r"(contusion|bone bruise|bone marrow edema|bone marrow oedema|marrow edema"
        r"|contusao|edema oseo|edema de medula"
        r"|knochenmarkodem|knochenmarksodem|kontusion"
        r"|oedeme osseux|botcontusie|beenmergoedeem"
        r"|kontuzyon|kemik ilig odem"
        r"|ушиб|отек костного мозга|контузи"
        r"|οστεομυελικ\w* οιδημα|οιδημα μυελου|μωλωπ\w*)",
        None,
    ),
    "Fracture": (
        r"(fracture|fractura|fratura|frattura|fraktur|breuk|kirik|avulsion"
        r"|перелом|καταγμα\w*|καταγματ\w*)",
        None,
    ),
}

_COMPARTMENT_OA = {"Medial OA": "medial", "Lateral OA": "lateral", "PF OA": "pf"}


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


# Phrases that name a femorotibial structure explicitly. Used to disambiguate a
# clause such as "cartilage defect at lateral trochlea", where "lateral" qualifies
# a patellofemoral structure and must not be credited to the lateral compartment.
_FT_SPECIFIC = {
    "medial": (
        r"(medial compartment|compartimento medial|compartimento interno"
        r"|femorotibial medial|femorotibial interno|medial femoral condyle"
        r"|medial tibial plateau|condilo femoral medial|condilo femoral interno"
        r"|platillo tibial medial|innenkompartiment|mediale kompartiment)"
    ),
    "lateral": (
        r"(lateral compartment|compartimento lateral|compartimento externo"
        r"|femorotibial lateral|femorotibial externo|lateral femoral condyle"
        r"|lateral tibial plateau|condilo femoral lateral|condilo femoral externo"
        r"|platillo tibial lateral|aussenkompartiment|laterale kompartiment)"
    ),
}


def _score_compartment_oa(text: str, side: str) -> float:
    """OA is compartment-specific: require an OA term and the compartment in one clause."""
    side_re = _SIDE[side]
    pf_re = _SIDE["pf"]
    for clause in re.split(r"[.;\n]", text):
        m = re.search(_OA, clause)
        if not m or _negated(clause, m.start()):
            continue
        if not re.search(side_re, clause):
            continue
        if side in _FT_SPECIFIC and re.search(pf_re, clause):
            # Clause concerns the patellofemoral joint; only credit the
            # femorotibial compartment when it is named outright.
            if not re.search(_FT_SPECIFIC[side], clause):
                continue
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
