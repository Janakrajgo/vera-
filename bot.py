"""
Vera AI Bot — FastAPI server implementing all 5 challenge endpoints.

Endpoints:
  GET  /v1/healthz     — liveness probe
  GET  /v1/metadata    — bot identity
  POST /v1/context     — receive context pushes (idempotent by version)
  POST /v1/tick        — periodic wake-up; bot initiates conversations
  POST /v1/reply       — receive and respond to merchant/customer replies
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
from composer import compose, compose_async
from conversation_handlers import respond, ConversationManager, ConversationState

__all__ = ["app", "compose", "compose_async", "respond", "ConversationManager", "ConversationState"]

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger("vera.bot")

# ── FastAPI App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Vera AI Bot",
    description="magicpin Merchant AI Assistant — Challenge Submission",
    version=config.BOT_VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# ── In-Memory Stores ─────────────────────────────────────────────────────
# (scope, context_id) → {version, payload}
contexts: dict[tuple[str, str], dict] = {}

# Sent suppression keys — tracks what we've already sent
sent_suppressions: set[str] = set()

# Conversation manager
conv_manager = ConversationManager()

# Active conversation metadata
# conversation_id → {merchant_id, customer_id, trigger_id, category_slug}
active_conversations: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _get_context(scope: str, context_id: str) -> Optional[dict]:
    """Retrieve a context payload by scope and ID."""
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def _get_category_for_merchant(merchant_id: str) -> Optional[dict]:
    """Look up the category context for a given merchant."""
    merchant = _get_context("merchant", merchant_id)
    if not merchant:
        return None
    cat_slug = merchant.get("category_slug", "")
    return _get_context("category", cat_slug)


def _get_customer(customer_id: str) -> Optional[dict]:
    """Look up a customer context."""
    return _get_context("customer", customer_id)


def _count_contexts() -> dict[str, int]:
    """Count loaded contexts by scope."""
    counts: dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return counts


def preload_dataset():
    """Preload expanded dataset into contexts store if available."""
    expanded_dir = config.EXPANDED_DIR
    if expanded_dir.exists() and (expanded_dir / "categories").exists():
        # Categories
        for f in (expanded_dir / "categories").glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
                contexts[("category", data["slug"])] = {"version": 1, "payload": data}

        # Merchants
        for f in (expanded_dir / "merchants").glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
                contexts[("merchant", data["merchant_id"])] = {"version": 1, "payload": data}

        # Customers
        for f in (expanded_dir / "customers").glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
                contexts[("customer", data["customer_id"])] = {"version": 1, "payload": data}

        # Triggers
        for f in (expanded_dir / "triggers").glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
                contexts[("trigger", data["id"])] = {"version": 1, "payload": data}
        logger.info(f"Loaded contexts from directory: {expanded_dir}")
    else:
        # Fallback to embedded dataset for zero-config cloud deployments
        try:
            from embedded_dataset import get_embedded_contexts
            embedded = get_embedded_contexts()
            contexts.update(embedded)
            logger.info(f"Loaded {len(embedded)} contexts from embedded dataset fallback.")
        except Exception as e:
            logger.warning(f"Could not load embedded dataset fallback: {e}")


STATIC_DIR = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the interactive WhatsApp Web Simulator UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    try:
        from ui_html import HTML_CONTENT
        return HTMLResponse(content=HTML_CONTENT)
    except Exception as e:
        logger.warning(f"Could not load embedded UI: {e}")
    return HTMLResponse("<h1>Vera AI Assistant Running</h1>")


@app.get("/api/dataset_summary")
async def dataset_summary():
    """Helper for the UI to retrieve populated directory data."""
    cats = [v["payload"] for k, v in contexts.items() if k[0] == "category"]
    mxs = [v["payload"] for k, v in contexts.items() if k[0] == "merchant"]
    cxs = [v["payload"] for k, v in contexts.items() if k[0] == "customer"]
    trgs = [v["payload"] for k, v in contexts.items() if k[0] == "trigger"]
    return {
        "categories_count": len(cats),
        "merchants": mxs[:50],
        "customers": cxs[:50],
        "triggers": trgs[:100],
    }


class InteractiveComposeReq(BaseModel):
    merchant_id: str
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    trigger: Optional[dict] = None


@app.post("/api/compose_interactive")
async def compose_interactive(req: InteractiveComposeReq):
    """Simulate a live trigger composition for the UI."""
    merchant = _get_context("merchant", req.merchant_id)
    if not merchant:
        return {"body": "Merchant not found", "cta": "none", "send_as": "vera", "rationale": "Missing merchant context"}
    category = _get_category_for_merchant(req.merchant_id) or {"slug": merchant.get("category_slug", "salons")}
    customer = _get_customer(req.customer_id) if req.customer_id else None
    trigger = req.trigger or _get_context("trigger", req.trigger_id or "") or {
        "id": "trg_manual", "kind": "perf_spike", "suppression_key": f"manual:{req.merchant_id}", "payload": {}
    }
    composed = await compose_async(category=category, merchant=merchant, trigger=trigger, customer=customer)
    return composed


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINT 1: GET /v1/healthz
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": _count_contexts(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINT 2: GET /v1/metadata
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": config.TEAM_NAME,
        "team_members": config.TEAM_MEMBERS,
        "model": (
            config.GEMINI_MODEL if config.LLM_PROVIDER == "gemini"
            else config.OPENAI_MODEL if config.LLM_PROVIDER == "openai"
            else config.ANTHROPIC_MODEL
        ),
        "approach": (
            "4-context composition framework with trigger-kind routing, "
            "LLM-powered message generation, auto-reply detection, "
            "intent transition handling, and multi-turn conversation management"
        ),
        "contact_email": config.CONTACT_EMAIL,
        "version": config.BOT_VERSION,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINT 3: POST /v1/context
# ═══════════════════════════════════════════════════════════════════════════

class ContextPush(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: ContextPush):
    key = (body.scope, body.context_id)

    # Idempotent by (scope, context_id, version)
    current = contexts.get(key)
    if current and current["version"] > body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current["version"],
        }
    if current and current["version"] == body.version:
        contexts[key] = {"version": body.version, "payload": body.payload}
        return {
            "accepted": True,
            "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.now(timezone.utc).isoformat() + "Z",
        }

    # Validate scope
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return {
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"Scope must be one of {valid_scopes}",
        }

    # Store
    contexts[key] = {"version": body.version, "payload": body.payload}
    logger.info(
        f"Context stored: scope={body.scope} id={body.context_id} "
        f"v={body.version} (payload_size={len(str(body.payload))} chars)"
    )

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat() + "Z",
    }


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINT 4: POST /v1/tick
# ═══════════════════════════════════════════════════════════════════════════

class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickRequest):
    actions = []

    for trg_id in body.available_triggers:
        # Skip if already sent for this suppression key
        trigger = _get_context("trigger", trg_id)
        if not trigger:
            logger.debug(f"Trigger {trg_id} not found in context store")
            continue

        suppression_key = trigger.get("suppression_key", "")
        if suppression_key in sent_suppressions:
            logger.debug(f"Skipping {trg_id}: suppression_key already sent")
            continue

        # Check expiry
        expires_at = trigger.get("expires_at", "")
        if expires_at and body.now > expires_at:
            logger.debug(f"Skipping {trg_id}: expired at {expires_at}")
            continue

        # Resolve merchant
        merchant_id = trigger.get("merchant_id")
        if not merchant_id:
            continue
        merchant = _get_context("merchant", merchant_id)
        if not merchant:
            logger.debug(f"Skipping {trg_id}: merchant {merchant_id} not found")
            continue

        # Resolve category
        cat_slug = merchant.get("category_slug", "")
        category = _get_context("category", cat_slug)
        if not category:
            logger.debug(f"Skipping {trg_id}: category {cat_slug} not found")
            continue

        # Resolve optional customer
        customer_id = trigger.get("customer_id")
        customer = _get_customer(customer_id) if customer_id else None

        # Compose the message
        try:
            composed = await compose_async(
                category=category,
                merchant=merchant,
                trigger=trigger,
                customer=customer,
            )
        except Exception as e:
            logger.error(f"Composition failed for {trg_id}: {e}")
            continue

        # Skip if body is empty (composition failed)
        if not composed.get("body", "").strip():
            logger.warning(f"Empty body for {trg_id}, skipping")
            continue

        # Generate conversation ID
        conv_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:8]}"

        # Build template params from merchant identity
        identity = merchant.get("identity", {})
        template_params = [
            identity.get("name", ""),
            trigger.get("kind", ""),
            composed.get("body", "")[:50],  # preview
        ]

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed["send_as"],
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": template_params,
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"],
        }
        actions.append(action)

        # Track the suppression key and conversation
        sent_suppressions.add(suppression_key)
        active_conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
            "category_slug": cat_slug,
        }
        conv_manager.record_bot_send(conv_id, composed["body"])
        conv_manager.get_or_create(conv_id, merchant_id, customer_id, trg_id)

        logger.info(
            f"Action: {conv_id} | trigger={trg_id} | "
            f"merchant={merchant_id} | cta={composed['cta']} | "
            f"body_len={len(composed['body'])}"
        )

    logger.info(f"Tick at {body.now}: {len(actions)} actions from "
                f"{len(body.available_triggers)} available triggers")

    return {"actions": actions}


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINT 5: POST /v1/reply
# ═══════════════════════════════════════════════════════════════════════════

class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyRequest):
    conv_id = body.conversation_id
    logger.info(
        f"Reply in {conv_id} | from={body.from_role} | "
        f"turn={body.turn_number} | msg_len={len(body.message)}"
    )

    # Look up conversation metadata
    conv_meta = active_conversations.get(conv_id, {})
    merchant_id = body.merchant_id or conv_meta.get("merchant_id", "")
    customer_id = body.customer_id or conv_meta.get("customer_id")
    trigger_id = conv_meta.get("trigger_id", "")
    cat_slug = conv_meta.get("category_slug", "")

    # Resolve contexts
    category = _get_context("category", cat_slug) if cat_slug else None
    merchant = _get_context("merchant", merchant_id) if merchant_id else None
    trigger = _get_context("trigger", trigger_id) if trigger_id else None
    customer = _get_customer(customer_id) if customer_id else None

    # Handle through conversation manager
    result = await conv_manager.handle_reply(
        conversation_id=conv_id,
        merchant_id=merchant_id,
        customer_id=customer_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number,
        category=category,
        merchant=merchant,
        trigger=trigger,
        customer=customer,
    )

    # Normalize output for the judge
    action = result.get("action", "send")

    if action == "send":
        return {
            "action": "send",
            "body": result.get("body", ""),
            "cta": result.get("cta", "open_ended"),
            "rationale": result.get("rationale", ""),
        }
    elif action == "wait":
        return {
            "action": "wait",
            "wait_seconds": result.get("wait_seconds", 1800),
            "rationale": result.get("rationale", ""),
        }
    elif action == "end":
        resp = {
            "action": "end",
            "rationale": result.get("rationale", ""),
        }
        if result.get("body"):
            resp["body"] = result["body"]
        return resp
    else:
        return {
            "action": "end",
            "rationale": result.get("rationale", "Conversation ended; maintaining silence."),
        }


# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL: POST /v1/teardown — cleanup
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/v1/teardown")
async def teardown():
    """Wipe all state — called by judge at end of test."""
    contexts.clear()
    sent_suppressions.clear()
    conv_manager.conversations.clear()
    conv_manager.merchant_auto_replies.clear()
    active_conversations.clear()
    logger.info("Teardown complete — all state wiped.")
    return {"status": "wiped"}


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    preload_dataset()
    logger.info(f"Vera AI Bot v{config.BOT_VERSION} starting...")
    logger.info(f"  Contexts loaded: {_count_contexts()}")
    logger.info(f"  LLM Provider: {config.LLM_PROVIDER}")
    logger.info(f"  Model: {config.GEMINI_MODEL if config.LLM_PROVIDER == 'gemini' else config.OPENAI_MODEL if config.LLM_PROVIDER == 'openai' else config.ANTHROPIC_MODEL}")
    logger.info(f"  Temperature: {config.TEMPERATURE}")
    logger.info(f"  Team: {config.TEAM_NAME}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "bot:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info",
    )
