"""
Multi-turn conversation handler for Vera.

Manages conversation state, auto-reply detection, intent transitions,
graceful exits, and turn-count awareness.

Implements `respond(state, merchant_message)` per challenge-brief.md §7.4.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from utils import (
    is_auto_reply,
    is_repeated_message,
    detect_intent,
    prefers_hinglish,
    get_owner_salutation,
)
from composer import compose_reply, _pick_non_repeating

logger = logging.getLogger("vera.conversation")


@dataclass
class ConversationState:
    """Tracks the state of an ongoing conversation."""
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    turns: list[dict] = field(default_factory=list)
    auto_reply_count: int = 0
    unanswered_nudge_count: int = 0
    intent_history: list[str] = field(default_factory=list)
    ended: bool = False
    end_reason: Optional[str] = None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def last_bot_message(self) -> Optional[str]:
        for turn in reversed(self.turns):
            if turn.get("from") == "bot":
                return turn.get("msg", "")
        return None

    def add_turn(self, from_role: str, message: str):
        self.turns.append({"from": from_role, "msg": message})


def respond(
    state: ConversationState,
    merchant_message: str,
    category: Optional[dict] = None,
    merchant: Optional[dict] = None,
    trigger: Optional[dict] = None,
    customer: Optional[dict] = None,
) -> dict:
    """
    Given the conversation state + latest message, produce the reply.
    Implements challenge-brief.md §7.4:
      def respond(state: ConversationState, merchant_message: str) -> dict:
    """
    message = merchant_message.strip()
    role = "customer" if state.customer_id else "merchant"
    state.add_turn(role, message)

    prev_bot_msgs = {
        turn.get("msg", "").strip().lower()
        for turn in state.turns
        if turn.get("from") == "bot" and turn.get("msg")
    }

    # 1. Conversation already ended check
    if state.ended:
        # Check if merchant/customer is explicitly asking to re-engage/restart
        reopen_keywords = ["restart", "reopen", "start again", "changed my mind", "actually yes", "proceed now", "start over", "re-activate", "help me after all"]
        wants_reopen = any(kw in message.lower() for kw in reopen_keywords) and state.end_reason in ("merchant_declined", "bot_decided_to_end", "max_turns_reached")

        if wants_reopen:
            # Reactivate conversation and freshly compose an acknowledgment
            state.ended = False
            state.end_reason = None
            state.intent_history.append("reopen")
            result = compose_reply(
                category=category,
                merchant=merchant,
                trigger=trigger,
                conversation_history=state.turns,
                merchant_message=message,
                intent="positive",
                customer=customer,
            )
            if result.get("action") == "send" and result.get("body"):
                state.add_turn("bot", result["body"])
            return result

        # Rule 1: Once ended, the bot should NOT send another message by default.
        # Silence is correct here, not a repeated goodbye.
        return {
            "action": "end",
            "rationale": f"Conversation already ended ({state.end_reason}); maintaining silence.",
        }

    # 2. WhatsApp Business auto-reply detection
    if is_auto_reply(message):
        state.auto_reply_count += 1
        logger.info(f"Auto-reply #{state.auto_reply_count} in {state.conversation_id}")

        if state.auto_reply_count >= 2:
            state.ended = True
            state.end_reason = "repeated_auto_reply"
            return {
                "action": "end",
                "rationale": "Detected repeated canned auto-reply; gracefully exiting to avoid burning turns.",
            }
        else:
            hinglish = prefers_hinglish(merchant or {}, customer)
            biz_name = merchant.get("identity", {}).get("name", "your business") if merchant else "your business"
            salutation = get_owner_salutation(merchant or {}, category or {}) if merchant and category else "there"
            if hinglish:
                ar_options = [
                    f"Looks like an automated reply! Kya {biz_name} ke owner ya manager available hain? Listing update ke regarding connect karna tha. Reply anytime — I'll be here.",
                    f"Auto-reply received! Jab bhi {salutation} ya management team free hon, ek quick ping kar dijiyega. Main listing draft ready rakhti hoon.",
                    f"Automated acknowledgement noted. I'll hold the update for {biz_name} until the owner is back online. Text back anytime!",
                ]
            else:
                ar_options = [
                    f"Looks like an automated reply! Is the manager or owner of {biz_name} available? I have a specific update regarding your business listing.",
                    f"Auto-reply noted! When {salutation} or the management team is available, please reply here — I have a quick listing update ready.",
                    f"Automated response received. Holding the performance update for {biz_name} until the owner is available. Text back anytime!",
                ]
            body = _pick_non_repeating(ar_options, prev_bot_msgs)
            state.add_turn("bot", body)
            return {
                "action": "send",
                "body": body,
                "cta": "open_ended",
                "rationale": "First auto-reply detected; asking for owner directly before backing off.",
            }

    # 3. Repeated message detection (canned bot pattern or duplicate user input)
    if is_repeated_message(message, state.turns):
        state.auto_reply_count += 1
        if state.auto_reply_count >= 2:
            state.ended = True
            state.end_reason = "repeated_same_message"
            return {
                "action": "end",
                "rationale": "Same message received multiple times; identified as automated loop. Exiting gracefully.",
            }

    # 4. Intent detection
    intent = detect_intent(message)
    state.intent_history.append(intent)

    # 5. Negative intent → immediate graceful exit (farewell sent once, subsequent turns remain silent)
    if intent == "negative":
        state.ended = True
        state.end_reason = "merchant_declined"
        result = compose_reply(
            category=category,
            merchant=merchant,
            trigger=trigger,
            conversation_history=state.turns,
            merchant_message=message,
            intent=intent,
            customer=customer,
        )
        farewell_body = result.get("body", "")
        if farewell_body:
            state.add_turn("bot", farewell_body)
        return {
            "action": "end",
            "body": farewell_body,
            "cta": "none",
            "rationale": "Merchant/customer signaled not interested; gracefully exiting.",
        }

    # 6. Turn limit safeguard
    if state.turn_count >= 10:
        state.ended = True
        state.end_reason = "max_turns_reached"
        return {
            "action": "end",
            "rationale": "Reached 10 turns limit; ending conversation gracefully to avoid over-messaging.",
        }

    # 7. Core reply composition (intent transitions, questions, off-topic, etc.)
    result = compose_reply(
        category=category,
        merchant=merchant,
        trigger=trigger,
        conversation_history=state.turns,
        merchant_message=message,
        intent=intent,
        customer=customer,
    )

    if result.get("action") == "send" and result.get("body"):
        state.add_turn("bot", result["body"])

    if result.get("action") == "end":
        state.ended = True
        state.end_reason = "bot_decided_to_end"

    return result


class ConversationManager:
    """Manages all active conversations and coordinates reply lifecycle."""

    def __init__(self):
        self.conversations: dict[str, ConversationState] = {}
        self.merchant_auto_replies: dict[str, int] = {}

    def get_or_create(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: Optional[str] = None,
        trigger_id: Optional[str] = None,
    ) -> ConversationState:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationState(
                conversation_id=conversation_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                trigger_id=trigger_id,
            )
        return self.conversations[conversation_id]

    def record_bot_send(self, conversation_id: str, body: str):
        """Record that the bot initiated or sent a message in this conversation."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].add_turn("bot", body)

    async def handle_reply(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: Optional[str],
        from_role: str,
        message: str,
        turn_number: int,
        category: Optional[dict] = None,
        merchant: Optional[dict] = None,
        trigger: Optional[dict] = None,
        customer: Optional[dict] = None,
    ) -> dict:
        """Process incoming reply and return the next bot move."""
        state = self.get_or_create(conversation_id, merchant_id, customer_id)
        if is_auto_reply(message):
            self.merchant_auto_replies[merchant_id] = self.merchant_auto_replies.get(merchant_id, 0) + 1
            if self.merchant_auto_replies[merchant_id] >= 2:
                state.auto_reply_count = max(state.auto_reply_count, self.merchant_auto_replies[merchant_id] - 1)

        return respond(
            state=state,
            merchant_message=message,
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
        )
