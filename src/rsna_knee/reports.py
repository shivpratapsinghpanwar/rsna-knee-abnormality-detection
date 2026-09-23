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
    # \broto\b is essential: unbounded, "roto" fires inside "protocol" and
    # "proton density", which appear in ~19% of reports.
    r"|rotura|ruptura|desgarro|\broto\b|lesion|lesao|rottura|lacerazione"
    r"|riss|einriss|laesion"
    r"|dechirure|fissuration"
    r"|scheur|ruptuur"
    # yirti\w* covers the Turkish inflections: yirtik, yirtigi, yirtilma.
    r"|yirti\w*|rupturu"
    r"|разрыв|повреждени|надрыв"
    r"|ρηξ\w*|ρηγμα|διατομ\w*)"
)

DEGEN = (
    r"(degener|maceration|mucoid|intrasubstance|degenerativ|degenerac"
    r"|dejenerasyon|εκφυλ\w*)"
)

# --- negation (NegEx-style, sentence-scoped, bidirectional) -----------------
#
# Cues are split by the direction they scope, because negation position is
# language-dependent: English/Romance/Greek negate before the finding ("no tear of
# the medial meniscus"), while Turkish and Russian are verb-final and negate after
# it ("medial menisküs yırtığı YOK", "разрыв НЕ ВЫЯВЛЕН"). English also has common
# post-forms ("...tear is not identified"), which a backward-only scan misses
# entirely and scores as positive.
#
# Every cue is word-bounded. Unbounded short cues match inside ordinary words --
# Spanish "menisco interno y" contains "no ", which silently negated whatever
# followed, and interno/externo are pervasive in Romance-language reports.

# Cues that scope FORWARD over the finding that follows them.
PRE_NEGATION = (
    r"(\bno\b|\bnot\b|\bwithout\b|\babsence of\b|\bnegative for\b|\bfree of\b"
    r"|\bno evidence\b"
    r"|\bsin\b|\bausencia\b"
    r"|\bkeine?\b|\bohne\b"
    r"|\bpas de\b|\bsans\b|\babsence\b"
    r"|\bgeen\b|\bzonder\b"
    r"|\bнет\b|\bбез\b|\bне\b"
    r"|\bδεν\b|\bχωρις\b)"
)

# Cues that scope BACKWARD over the finding that precedes them.
POST_NEGATION = (
    r"(\bnot\b|\bintact\b|\bunremarkable\b|\bnormal\w*\b|\bpreserved\b"
    r"|\bintegro\b|\bintegra\b|\bconservad\w*\b|\bnormales\b"
    r"|\bintakt\b|\bunauffaellig\w*\b|\bregelrecht\w*\b"
    r"|\bintacte\b"
    r"|\byok\b|\byoktur\b|\bizlenmedi\b|\bizlenmemistir\b|\bsaglam\b|\bdogaldir\b"
    r"|\bnormaldir\b|\bmevcut degil\b"
    r"|\bне выявлен\w*\b|\bне определя\w*\b|\bинтакт\w*\b|\bсохранен\w*\b"
    r"|\bακεραι\w*\b|\bφυσιολογικ\w*\b)"
)

# Tokens that terminate a negation scope: after one of these, the cue no longer
# applies ("no effusion BUT there is a meniscal tear").
NEG_TERMINATOR = (
    r"(\bbut\b|\bhowever\b|\balthough\b|\bwhereas\b|\bwhile\b|\bwith\b"
    r"|\bshowing\b|\bdemonstrat\w*\b|\bthere is\b|\bpresent\b"
    r"|\bpero\b|\bsin embargo\b|\baunque\b"
    r"|\baber\b|\bjedoch\b"
    r"|\bmaar\b|\bancak\b|\bfakat\b"
    r"|\bно\b|\bоднако\b"
    r"|\bαλλα\b|\bωστοσο\b)"
)

# Scope sizes. The backward scope is tighter because post-cues bind closely to the
# finding they qualify, while pre-cues routinely govern a longer noun phrase.
PRE_WINDOW = 60
POST_WINDOW = 45
# How far from an OA term the compartment word must appear to be attributed to it.
OA_SIDE_WINDOW = 60

# Degenerative / cartilage findings. Gold PF-OA cases are frequently written as a
# chondral defect rather than the word "arthrosis", so cartilage vocabulary has to
# count. "(?<!sub)" keeps "subchondral" (e.g. subchondral fracture oedema) from
# firing, since that is not itself a degenerative finding.
_OA = (
    r"(osteoarthriti\w*|osteoarthrosis|arthrosis|arthrose|artrosis|artrose|artrosi"
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

# --- meniscal grading, hedging and severity ------------------------------------
#
# These three axes turn the labeler's output from binary into graded. The image
# model is scored by ROC AUC, which rewards ranking, so a report that says
# "large effusion" should train a higher target than one saying "trace fluid",
# and "mucoid degeneration" should not train the same target as "radial tear".

# Explicit meniscal tear morphologies. A named tear pattern is a tear regardless
# of how the rest of the sentence hedges.
MENISCUS_TEAR_PATTERN = (
    r"(bucket[- ]?handle|bucket handle|radial tear|longitudinal tear|vertical tear"
    r"|horizontal tear|oblique tear|complex tear|parrot beak|flap tear|root tear"
    r"|posterior root|displaced fragment|meniscal fragment"
    r"|asa de cubo|rotura radial|rotura longitudinal|rotura horizontal"
    r"|korbhenkel|radiarriss|laengsriss|horizontalriss"
    r"|anse de seau|radiaire"
    r"|emmerhandvat"
    r"|kova sapi|radial yirtik|longitudinal yirtik|horizontal yirtik"
    r"|ручка лейки|радиальн\w* разрыв"
    r"|δικην λαβης καδου|ακτινωτ\w* ρηξ\w*)"
)

# Meniscal extrusion is a structural failure and tracks with root tears.
MENISCUS_EXTRUSION = (
    r"(extrusion|extrude|extrusao|extruido|extrusi[oó]n|ekstrude|ekstruzyon"
    r"|subluxation of the meniscus|экструзи\w*|υπεξαρθρημα)"
)

# Stoller grading. Grade 3 signal reaches the articular surface and IS a tear;
# grades 1-2 are intrasubstance degeneration and are not.
GRADE_3 = r"(grade\s*(iii|3)\b|grado\s*(iii|3)\b|grad\s*(iii|3)\b|evre\s*(iii|3)\b|βαθμου\s*(iii|3)\b)"
GRADE_12 = r"(grade\s*(i{1,2}|[12])\b|grado\s*(i{1,2}|[12])\b|grad\s*(i{1,2}|[12])\b|evre\s*(i{1,2}|[12])\b)"

# Signal that reaches the articular surface is a tear; signal that explicitly does
# not reach it is degeneration. Both phrasings are common and decisive.
SURFACE_REACH = (
    r"(extend\w*\s+((in)?to\s+)?the (articular )?surface|reach\w*\s+the (articular )?surface"
    r"|contact\w*\s+the (articular )?surface|communicat\w*\s+with the (articular )?surface"
    r"|surface extension|articular surface extension"
    r"|contacta l[ao] superficie|se extiende a la superficie|alcanza la superficie"
    r"|eklem yuzune uzan\w*|yuzeye ulas\w*"
    r"|достига\w* сустав\w* поверхност"
    r"|εκτεινεται στην αρθρικη επιφανεια)"
)
SURFACE_NO_REACH = (
    r"(\bnot?\b\s+\w*\s*(extend|reach|contact|communicat)\w*\s+((in)?to\s+)?the (articular )?surface"
    r"|without (extension|contact) (in)?to the (articular )?surface"
    r"|no impresiona contactar|no contacta|sin extension a la superficie"
    r"|sin alcanzar la superficie"
    r"|eklem yuzune uzanmamakta|yuzeye ulasmamakta"
    r"|не достига\w*"
    r"|δεν εκτεινεται)"
)

# Abnormal intrasubstance signal, the substrate of meniscal grading. On its own it
# is degeneration; it becomes a tear only once it reaches the articular surface.
INTRASUBSTANCE_SIGNAL = (
    r"(increased signal|hyperintense signal|signal hyperintensity|high signal"
    r"|linear signal|signal intensity|signal change|signal alteration"
    r"|aumento de senal|hiperintensidad|senal intrameniscal"
    r"|signalanhebung|signalsteigerung|signalalteration"
    r"|hypersignal|signaalverhoging"
    r"|sinyal artis\w*|sinyal degisikli\w*|intrasubstans sinyal"
    r"|повышени\w* сигнал|сигнал\w* измен"
    r"|αυξημενο σημα|σηματοδοτησ\w*)"
)

# Hedging. A suspected finding is real evidence but weaker than a stated one.
HEDGE = (
    r"(\bpossible\b|\bpossibly\b|\bprobable\b|\bprobably\b|\bsuspicious for\b"
    r"|\bsuspected\b|\bquestionable\b|\bcannot (be )?exclude\w*\b|\bmay represent\b"
    r"|\bcould represent\b|\bequivocal\b|\bborderline\b|\blikely\b|\bapparent\b"
    r"|\bpodria\b|\bprobable\b|\bsugestivo\b|\bsospecha\b|\bno se puede excluir\b"
    r"|\bimpresiona\b|\bmoeglich\b|\bverdachtig\b|\bfraglich\b|\bvermutlich\b"
    r"|\bmogelijk\b|\bverdacht\b"
    r"|\bsuphel\w*\b|\bolabilir\b|\bdusundur\w*\b|\buyumlu olabilec\w*\b"
    r"|\bвозможно\b|\bвероятно\b|\bподозрительн\w*\b|\bне исключ\w*\b"
    r"|\bπιθαν\w*\b|\bυποπτ\w*\b)"
)

# Severity scale, used for graded targets such as effusion. Ordered strongest
# first so the strongest qualifier present in a sentence wins.
SEVERITY = [
    (1.00, r"(\bmassive\b|\blarge\b|\bgross\b|\bmarked\b|\btense\b|\bsevere\b|\bextensive\b"
           r"|\babundante\b|\bimportante\b|\bsevero\b|\bgrave\b"
           r"|\bausgepraegt\w*\b|\bmassiv\w*\b|\bgross\w*\b"
           r"|\bbelirgin\b|\byaygin\b|\bciddi\b|\bileri\b"
           r"|\bвыраженн\w*\b|\bбольш\w*\b|\bμεγαλ\w*\b|\bαθροο\b)"),
    (0.80, r"(\bmoderate\b|\bmoderado\b|\bmassig\b|\bmaessig\w*\b|\bmatig\b|\borta\b"
           r"|\bумеренн\w*\b|\bμετρι\w*\b|\bικανη\b)"),
    (0.55, r"(\bsmall\b|\bmild\b|\bslight\b|\bsome\b|\bleve\b|\bdiscreta\b|\bligero\b"
           r"|\bgering\w*\b|\bleicht\w*\b|\bgeringe\b|\bbeperkte\b|\blicht\w*\b"
           r"|\bhafif\b|\baz\b|\bнебольш\w*\b|\bнезначительн\w*\b|\bμικρ\w*\b|\bηπι\w*\b)"),
    (0.35, r"(\btrace\b|\bminimal\b|\bminima\b|\bminimo\b|\bscant\b|\btiny\b"
           r"|\bminimal\w*\b|\bspoor\b|\bminimal\b|\bеденичн\w*\b|\bследов\w*\b"
           r"|\bελαχιστ\w*\b)"),
]
DEFAULT_SEVERITY = 0.70   # a finding stated with no qualifier at all

# Graded score assigned when a meniscus shows degeneration but no tear. Non-zero
# because degenerate menisci do carry tear risk, but well below a stated tear.
DEGEN_ONLY = 0.30
HEDGE_FACTOR = 0.65       # multiplier applied to a hedged finding


# target -> (anchor regex, required finding regex or None when the anchor is the finding)
# The three OA targets are not listed here; they need compartment attribution and
# are handled by _score_compartment_oa.
ANCHORS: dict[str, tuple[str, str | None]] = {
    "ACL": (_ACL, TEAR),
    "MCL": (_MCL, TEAR),
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

# Menisci need tear-vs-degeneration separation, so they bypass ANCHORS.
_MENISCUS = {"Medial Meniscus": _MED_MEN, "Lateral Meniscus": _LAT_MEN}

# Targets whose reported severity is graded onto the score. Effusion is the clear
# case: "trace fluid" and "massive effusion" are the same binary label but very
# different images, and the metric rewards ranking them apart.
_SEVERITY_GRADED = {"Effusion", "Synovitis", "Contusion"}


_SENT_SPLIT = re.compile(r"[.;:\n]|(?:\s-\s)")

# Sub-clause boundaries. Compartment attribution must not cross one of these:
# "the lateral compartment shows osteoarthritis WHILE the medial compartment is
# normal" contains both side words near the OA term, and only clause structure
# says which compartment the finding belongs to.
_CLAUSE_SPLIT = re.compile(
    r",|\bbut\b|\bhowever\b|\bwhile\b|\bwhereas\b|\balthough\b"
    r"|\bpero\b|\baunque\b|\baber\b|\bjedoch\b|\bmaar\b|\bancak\b|\bfakat\b"
    r"|\bно\b|\bоднако\b|\bαλλα\b|\bωστοσο\b"
)


def sentences(text: str) -> list[str]:
    """Split a normalised report into sentence-like units.

    Negation is scoped to these units, so a cue in one sentence can never suppress
    a finding in the next -- previously "ACL tear is not seen. Medial meniscus tear
    is present." lost the meniscus label to the ACL sentence's negation.
    """
    return [s for s in _SENT_SPLIT.split(text) if s and not s.isspace()]


def _negation_cues(sentence: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Spans of forward-scoping and backward-scoping negation cues in one sentence."""
    pre = [(m.start(), m.end()) for m in re.finditer(PRE_NEGATION, sentence)]
    post = [(m.start(), m.end()) for m in re.finditer(POST_NEGATION, sentence)]
    return pre, post


def _is_negated(
    sentence: str,
    start: int,
    end: int,
    pre: list[tuple[int, int]],
    post: list[tuple[int, int]],
) -> bool:
    """True if a finding spanning [start, end) falls inside a negation scope."""
    for _, cue_end in pre:
        if cue_end <= start and start - cue_end <= PRE_WINDOW:
            if not re.search(NEG_TERMINATOR, sentence[cue_end:start]):
                return True
    for cue_start, _ in post:
        if cue_start >= end and cue_start - end <= POST_WINDOW:
            if not re.search(NEG_TERMINATOR, sentence[end:cue_start]):
                return True
    return False


def _severity_of(sentence: str) -> float:
    """Strongest severity qualifier present in a sentence, else the default."""
    for weight, pattern in SEVERITY:
        if re.search(pattern, sentence):
            return weight
    return DEFAULT_SEVERITY


def _hedged(sentence: str) -> bool:
    return bool(re.search(HEDGE, sentence))


def _score_simple(
    text: str, anchor: str, finding: str | None, graded: bool = False
) -> float:
    """Score an affirmed mention of the target across all sentences.

    Returns the strongest score found. With ``graded`` the score carries the
    reported severity ("trace" vs "massive"); otherwise an affirmed mention
    scores 1.0. A hedged mention is discounted either way.
    """
    best = 0.0
    for sentence in sentences(text):
        if finding is not None and not re.search(finding, sentence):
            continue
        pre, post = _negation_cues(sentence)
        for m in re.finditer(anchor, sentence):
            if _is_negated(sentence, m.start(), m.end(), pre, post):
                continue
            score = _severity_of(sentence) if graded else 1.0
            if _hedged(sentence):
                score *= HEDGE_FACTOR
            best = max(best, score)
            break
    return best


def _score_meniscus(text: str, anchor: str) -> float:
    """Meniscal tear, kept distinct from meniscal degeneration.

    Intrasubstance degeneration (Stoller grade 1-2, "mucoid", "myxoid") is a
    different entity from a tear and the gold labels treat it as such: reports
    reading "increased signal not contacting the articular surface" are labelled
    negative, while "radial tear of the posterior root" is positive. Grade 3
    signal is by definition a tear, because it reaches the articular surface.
    """
    best = 0.0
    for sentence in sentences(text):
        anchored = [
            m for m in re.finditer(anchor, sentence)
        ]
        if not anchored:
            continue
        # "signal that does NOT contact the articular surface" describes a real
        # degenerative meniscus. The negation belongs to the surface clause, not
        # to the meniscus, so it must be resolved before the anchor is tested --
        # otherwise the nearby "not" suppresses the finding entirely.
        if re.search(SURFACE_NO_REACH, sentence) and (
            re.search(INTRASUBSTANCE_SIGNAL, sentence) or re.search(DEGEN, sentence)
        ):
            best = max(best, DEGEN_ONLY)
            continue

        pre, post = _negation_cues(sentence)
        affirmed = [
            m for m in anchored
            if not _is_negated(sentence, m.start(), m.end(), pre, post)
        ]
        if not affirmed:
            continue

        # Decisive tear evidence, in descending order of certainty.
        tear = bool(
            re.search(MENISCUS_TEAR_PATTERN, sentence)
            or re.search(MENISCUS_EXTRUSION, sentence)
            or re.search(GRADE_3, sentence)
        )
        # An explicit tear word counts unless the sentence simultaneously says the
        # signal stops short of the articular surface, which describes degeneration.
        if not tear and re.search(TEAR, sentence):
            tear = not re.search(SURFACE_NO_REACH, sentence)
        # Signal reaching the surface is a tear even when the word is absent.
        if not tear and re.search(SURFACE_REACH, sentence):
            tear = not re.search(SURFACE_NO_REACH, sentence)

        if tear:
            score = HEDGE_FACTOR if _hedged(sentence) else 1.0
        elif (
            re.search(DEGEN, sentence)
            or re.search(GRADE_12, sentence)
            or re.search(INTRASUBSTANCE_SIGNAL, sentence)
        ):
            score = DEGEN_ONLY
        else:
            continue
        best = max(best, score)
    return best


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
    """OA is compartment-specific: the compartment word must sit near the OA term.

    Every OA match in a sentence is considered, not just the first -- a sentence
    like "the lateral compartment shows osteoarthritis while the medial compartment
    is normal" otherwise credited Medial OA off the trailing, negated phrase.
    """
    side_re = _SIDE[side]
    pf_re = _SIDE["pf"]
    for sentence in sentences(text):
        for clause in _CLAUSE_SPLIT.split(sentence):
            if not clause or clause.isspace():
                continue
            if not re.search(side_re, clause):
                continue
            pre, post = _negation_cues(clause)
            for m in re.finditer(_OA, clause):
                if _is_negated(clause, m.start(), m.end(), pre, post):
                    continue
                if side in _FT_SPECIFIC and re.search(pf_re, clause):
                    # Clause concerns the patellofemoral joint; only credit the
                    # femorotibial compartment when it is named outright.
                    if not re.search(_FT_SPECIFIC[side], clause):
                        continue
                return 1.0
    return 0.0


def label_report(report: str) -> dict[str, float]:
    """Extract graded weak labels in [0, 1] for all 12 targets from one report.

    Scores are deliberately not thresholded: they are training targets for a
    model scored by ROC AUC, so preserving "large effusion" > "trace effusion"
    and "radial tear" > "mucoid degeneration" is worth more than a crisp 0/1.
    """
    text = normalize(report)
    out: dict[str, float] = {}
    for target in TARGETS:
        if target in _COMPARTMENT_OA:
            out[target] = _score_compartment_oa(text, _COMPARTMENT_OA[target])
        elif target in _MENISCUS:
            out[target] = _score_meniscus(text, _MENISCUS[target])
        else:
            anchor, finding = ANCHORS[target]
            out[target] = _score_simple(
                text, anchor, finding, graded=target in _SEVERITY_GRADED
            )
    return out
