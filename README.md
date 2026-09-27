# Vera AI Bot — magicpin Merchant AI Assistant

**Team**: Vera Reborn  
**Author**: Janak Raj Goyal  
**Version**: 1.0.0  
**Deliverables**: `bot.py`, `submission.jsonl`, `conversation_handlers.py`, `README.md`

---

## 1. Approach

Our implementation rebuilds Vera around the **4-Context Composition Framework** (`CategoryContext`, `MerchantContext`, `TriggerContext`, and `CustomerContext`), specifically architected to outperform legacy production Vera across all 5 evaluation dimensions:

1. **Dual-Engine Composition (`composer.py`)**:
   - **Frontier LLM Integration**: Supports Google Gemini, OpenAI, and Anthropic with temperature=0. The system prompt enforces zero-hallucination rules, single binary CTAs in the final sentence, language matching (Delhi/NCR Hinglish vs professional English), and category voice boundaries.
   - **Deterministic Domain Expert Engine**: An ultra-reliable, zero-latency generator covering all 5 verticals (`dentists`, `salons`, `restaurants`, `gyms`, `pharmacies`) and 26 trigger kinds. It extracts concrete verifiable metrics (views, call deltas, review milestones, exact catalog prices, trial numbers) and ensures 100% deterministic success without depending on external API quotas.
   - **Post-Generation Sanitizer**: Enforces `vocab_taboo` rules (e.g., stripping "guaranteed", "cure", "100%") and automatically resolves `send_as` (`"vera"` vs `"merchant_on_behalf"`).

2. **Multi-Turn Conversation Lifecycle (`conversation_handlers.py`)**:
   - **Auto-Reply Detection**: Bilingual regex patterns and repeated-message hashing identify WhatsApp Business canned messages. On the 1st auto-reply, Vera prompts for the owner/manager directly; on the 2nd, she exits gracefully with `action: "end"` rather than burning turns.
   - **Zero-Friction Intent Handoff**: When a merchant says *"Yes"*, *"I want to join"*, or *"go ahead"*, Vera immediately switches to **Action Mode** (initiating the update or draft) and never loops back to qualifying questions.
   - **Scope Boundaries**: Off-topic inquiries (e.g., GST filings, loans) are politely redirected back to marketing growth.
   - **Clean Exits**: Negative intent ("not interested", "stop") triggers an immediate, warm exit.

3. **Full Harness HTTP API (`bot.py`)**:
   - Exposes `GET /v1/healthz`, `GET /v1/metadata`, `POST /v1/context` (idempotent with version checking and `409 stale_version`), `POST /v1/tick` (deduplicated via `suppression_key`), `POST /v1/reply`, and `POST /v1/teardown`.
   - Exposes synchronous `compose(...)` per §7.1 and multi-turn `respond(...)` per §7.4.

---

## 2. Compulsion Levers & Strategic Tradeoffs

| Challenge Tradeoff | Decision Taken | Rationale |
|---|---|---|
| **Service+Price vs Discounts** | Anchored on exact services and prices (`Dental Cleaning @ ₹299`, `Haircut @ ₹99`) | Generic discount percentages ("20% off") fatigue Indian merchants; concrete prices create tangible walk-in value. |
| **CTA Structure** | Enforced a single binary CTA (`Reply YES / STOP`) or single open-ended question | Multiple competing choices cause analysis paralysis and reduce WhatsApp reply rates. |
| **Language Tone** | Natural Hindi-English code-mix when `hi` is in merchant languages | Standard WhatsApp communication in Tier 1 & 2 cities is colloquial Hinglish; pure English feels robotic. |
| **Resilience Strategy** | Dual LLM + deterministic domain generator | Guarantees sub-second responses and prevents zero-scores if API keys, rate limits, or network timeouts occur. |

---

## 3. What Additional Context Would Have Helped Most

1. **Merchant Active Hours / Rush Times**: Knowing peak operation hours (e.g., lunch rush 1-3 PM for restaurants, morning 6-9 AM for gyms) would enable optimal tick dispatch without interrupting operations.
2. **Historical Offer Conversion Benchmarks**: Category-level data on which specific price points (e.g., ₹99 vs ₹149) yield highest walk-in conversion in specific micro-markets.
3. **Multi-Channel Media Support**: The ability to send or preview drafted Google Business Profile banners and photo cards directly within WhatsApp.
