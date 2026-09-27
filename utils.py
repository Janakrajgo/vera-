"""
Utility helpers for the Vera bot:
  - Auto-reply detection
  - Intent/transition detection
  - Language preference helpers
  - Context extraction helpers
"""
from __future__ import annotations

import re
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# AUTO-REPLY DETECTION
# ═══════════════════════════════════════════════════════════════════════════

# Known WhatsApp Business canned auto-reply patterns (case-insensitive)
AUTO_REPLY_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"thank\s*you\s+for\s+contacting",
        r"thanks?\s+for\s+(reaching|getting)\s+(out|in\s+touch)",
        r"our\s+team\s+will\s+respond\s+shortly",
        r"we\s+will\s+get\s+back\s+to\s+you",
        r"we['']?ll\s+get\s+back",
        r"your\s+(message|query|request)\s+(has\s+been|is)\s+(received|noted)",
        r"automated\s+(reply|response|message|assistant)",
        r"main\s+ek\s+automated\s+assistant\s+hoon",
        r"aapki\s+jaankari.*shukriya",
        r"hamari\s+team\s+tak\s+pahuncha",
        r"apki\s+madad\s+ke\s+liye\s+shukriya.*lekin\s+main\s+ek",
        r"this\s+is\s+an?\s+auto(mated|matic)?\s*(reply|response|message)",
        r"currently\s+(away|unavailable|out\s+of\s+office)",
        r"will\s+revert\s+(back\s+)?soon",
        r"please\s+leave\s+your\s+(message|query)",
    ]
]


def is_auto_reply(message: str) -> bool:
    """Detect whether a message is a WhatsApp Business canned auto-reply."""
    if not message or len(message.strip()) < 5:
        return False
    text = message.strip()
    for pattern in AUTO_REPLY_PATTERNS:
        if pattern.search(text):
            return True
    return False


def is_repeated_message(message: str, history: list[dict]) -> bool:
    """Check if this exact message has been received 2+ times before."""
    normalized = message.strip().lower()
    count = sum(1 for turn in history
                if turn.get("from") in ("merchant", "customer")
                and turn.get("msg", "").strip().lower() == normalized)
    return count >= 2


# ═══════════════════════════════════════════════════════════════════════════
# INTENT / TRANSITION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

POSITIVE_INTENT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\byes\b", r"\byep\b", r"\byeah\b", r"\bsure\b",
        r"\bgo\s+ahead\b", r"\bproceed\b", r"\bdo\s+it\b",
        r"\blet['']?s\s+do\s+(it|this)\b", r"\bok\b",
        r"\bhaan\b", r"\bchalao\b", r"\bchalo\b",
        r"\bkar\s+(do|dein|dijiye)\b", r"\bbhejo\b",
        r"\bsend\b", r"\bstart\b", r"\bbegin\b",
        r"\bi\s+want\s+to\s+join\b", r"\bjudna\b", r"\bjoin\b",
        r"\bplease\s+(do|send|start|proceed)\b",
    ]
]

NEGATIVE_INTENT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bnot\s+interested\b", r"\bno\s+thanks?\b",
        r"\bstop\b", r"\bunsubscribe\b",
        r"\bdon['']?t\s+(message|contact|send|bother)\b",
        r"\bnahi\b.*\b(chahiye|interest)\b",
        r"\band\s+karo\b", r"\bbas\s+karo\b",
        r"\bplease\s+stop\b", r"\bremove\s+me\b",
    ]
]

OFF_TOPIC_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bgst\s+(filing|return)\b", r"\bincome\s+tax\b",
        r"\bloan\b", r"\binsurance\b", r"\blegal\s+advice\b",
        r"\bCA\s+(help|contact)\b",
    ]
]


def detect_intent(message: str) -> str:
    """
    Classify the merchant's reply intent.
    Returns one of: "positive", "negative", "off_topic", "question", "neutral"
    """
    text = message.strip()
    if not text:
        return "neutral"

    # Check negative first (takes priority)
    for pattern in NEGATIVE_INTENT_PATTERNS:
        if pattern.search(text):
            return "negative"

    # Check off-topic
    for pattern in OFF_TOPIC_PATTERNS:
        if pattern.search(text):
            return "off_topic"

    # Check positive
    for pattern in POSITIVE_INTENT_PATTERNS:
        if pattern.search(text):
            return "positive"

    # If it ends with a question mark, it's likely a question
    if text.rstrip().endswith("?"):
        return "question"

    return "neutral"


# ═══════════════════════════════════════════════════════════════════════════
# LANGUAGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_language_instruction(merchant: dict, customer: Optional[dict] = None) -> str:
    """Build a language instruction string from merchant/customer prefs."""
    if customer:
        lang_pref = customer.get("identity", {}).get("language_pref", "")
        if "hi" in lang_pref.lower():
            return (
                "Write in natural Hindi-English code-mix (Hinglish) — "
                "the way a real person texts on WhatsApp in India. "
                "Mix Hindi and English naturally within sentences."
            )
        if any(x in lang_pref.lower() for x in ["te", "ta", "kn", "mr"]):
            return (
                f"Write primarily in English with light {lang_pref} phrases "
                "where natural, matching the customer's preference."
            )
        return "Write in clear, warm English."

    languages = merchant.get("identity", {}).get("languages", ["en"])
    if "hi" in languages:
        return (
            "Write in natural Hindi-English code-mix (Hinglish) — "
            "the way a real Delhi/NCR merchant texts on WhatsApp. "
            "Mix Hindi and English naturally. Don't force pure Hindi or pure English."
        )
    return "Write in clear, professional English."


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def extract_digest_item(category: dict, item_id: str) -> Optional[dict]:
    """Find a specific digest item by ID from the category context."""
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return None


def get_merchant_name(merchant: dict) -> str:
    """Extract the merchant's display name."""
    identity = merchant.get("identity", {})
    return identity.get("name", identity.get("owner_first_name", "there"))


def get_owner_salutation(merchant: dict, category: dict) -> str:
    """Build the right salutation for this merchant+category combo."""
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", "")
    salutation_templates = category.get("voice", {}).get("salutation_examples", [])

    if salutation_templates and owner:
        template = salutation_templates[0]
        return (template
                .replace("{first_name}", owner)
                .replace("{chef_or_owner_first_name}", owner)
                .replace("{pharmacist_name}", owner)
                .replace("{gym_name}", identity.get("name", owner))
                .replace("{salon_name}", identity.get("name", owner))
                .replace("{restaurant_name}", identity.get("name", owner))
                .replace("{pharmacy_name}", identity.get("name", owner)))
    return owner or identity.get("name", "Hi")


def format_performance_summary(merchant: dict) -> str:
    """Format a concise performance summary string."""
    perf = merchant.get("performance", {})
    if not perf:
        return "No performance data available."
    parts = []
    if "views" in perf:
        parts.append(f"Views: {perf['views']:,}")
    if "calls" in perf:
        parts.append(f"Calls: {perf['calls']}")
    if "directions" in perf:
        parts.append(f"Directions: {perf['directions']}")
    if "ctr" in perf:
        parts.append(f"CTR: {perf['ctr']:.1%}")
    delta = perf.get("delta_7d", {})
    if delta:
        delta_parts = []
        for k, v in delta.items():
            sign = "+" if v > 0 else ""
            delta_parts.append(f"{k.replace('_pct', '')}: {sign}{v:.0%}")
        if delta_parts:
            parts.append(f"7d change: {', '.join(delta_parts)}")
    return " | ".join(parts)


def check_voice_violations(body: str, category: dict) -> list[str]:
    """Check for vocab_taboo violations in the composed message."""
    voice = category.get("voice", {})
    taboos = voice.get("vocab_taboo", [])
    violations = []
    body_lower = body.lower()
    for taboo in taboos:
        # Handle parenthetical notes like "FDA-approved (use only when...)"
        clean_taboo = re.sub(r'\(.*?\)', '', taboo).strip().lower()
        if clean_taboo and clean_taboo in body_lower:
            violations.append(taboo)
    return violations


def sanitize_taboos(body: str, category: dict) -> str:
    """Sanitize any vocab_taboo terms found in the body text."""
    voice = category.get("voice", {})
    taboos = voice.get("vocab_taboo", [])
    replacements = {
        r"\bguaranteed\b": "trusted",
        r"\bguarantee\b": "assurance",
        r"\bcures?\b": "treatment",
        r"\bcured\b": "treated",
        r"\b100%\s*(satisfaction|results?|cure)\b": "proven care",
        r"\bpermanent\s+fix\b": "long-lasting care",
        r"\bmiracle\b": "clinically effective",
        r"\bbest\s+in\s+city\b": "reputed",
        r"\bfree\s+for\s+everyone\b": "special consultation",
        r"\bflat\s+\d+%\s+off\b": "special package",
    }
    sanitized = body
    for taboo in taboos:
        clean = re.sub(r'\(.*?\)', '', taboo).strip()
        if clean and clean.lower() in sanitized.lower():
            for pat, repl in replacements.items():
                sanitized = re.sub(pat, repl, sanitized, flags=re.IGNORECASE)
    return sanitized


def prefers_hinglish(merchant: dict, customer: Optional[dict] = None) -> bool:
    """Determine whether the recipient prefers natural Hindi-English code-mix."""
    if customer:
        lang_pref = str(customer.get("identity", {}).get("language_pref", "")).lower()
        return "hi" in lang_pref
    languages = merchant.get("identity", {}).get("languages", ["en"])
    return any("hi" in str(l).lower() for l in languages)

