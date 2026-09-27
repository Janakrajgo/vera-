"""
Comprehensive verification test for:
1. Silence on ended conversations (no repeated canned lines)
2. Non-identical outputs when messages are repeated
3. Transition to silence after 2+ repeats / auto-replies
4. Clean HTTP API responses (/v1/reply)
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
from fastapi.testclient import TestClient

from bot import app
from conversation_handlers import respond, ConversationState

client = TestClient(app)

# Load test contexts
with open("dataset/categories/dentists.json", encoding="utf-8") as f:
    category = json.load(f)
with open("dataset/expanded/merchants/m_001_drmeera_dentist_delhi.json", encoding="utf-8") as f:
    merchant = json.load(f)
with open("dataset/expanded/triggers/trg_023_competitor_opened_dentist.json", encoding="utf-8") as f:
    trigger = json.load(f)


def test_repeated_messages_never_identical_and_end_in_silence():
    print("\n── Test 1: Sending Same Message 3 Times in Same Conversation ──")
    state = ConversationState(conversation_id="conv_repeat_test", merchant_id="m_001")
    test_msg = "Hello, what is this promotion about?"

    # Turn 1: First send
    r1 = respond(state, test_msg, category=category, merchant=merchant, trigger=trigger)
    assert r1["action"] == "send", f"Turn 1 expected send, got {r1['action']}"
    assert "body" in r1 and r1["body"], "Turn 1 must have a body"
    body1 = r1["body"]
    print(f"  [Turn 1] action={r1['action']} | body: {body1[:80]}...")

    # Turn 2: Second send (identical input)
    r2 = respond(state, test_msg, category=category, merchant=merchant, trigger=trigger)
    assert r2["action"] == "send", f"Turn 2 expected send, got {r2['action']}"
    assert "body" in r2 and r2["body"], "Turn 2 must have a body"
    body2 = r2["body"]
    print(f"  [Turn 2] action={r2['action']} | body: {body2[:80]}...")

    # Requirement 4: outputs must NEVER be identical
    assert body1 != body2, f"Turn 1 and Turn 2 outputs must NOT be identical!\nTurn 1: {body1}\nTurn 2: {body2}"
    print("  ✓ Turn 1 and Turn 2 are non-identical!")

    # Turn 3: Third send (identical input -> detected loop -> ends gracefully with silence)
    r3 = respond(state, test_msg, category=category, merchant=merchant, trigger=trigger)
    assert r3["action"] == "end", f"Turn 3 expected end, got {r3['action']}"
    assert not r3.get("body"), f"Turn 3 should be silent (no body), got {r3.get('body')}"
    assert state.ended is True, "Conversation state must be ended"
    print(f"  [Turn 3] action={r3['action']} | body={r3.get('body', '<SILENCE>')} | rationale={r3['rationale']}")
    print("  ✓ Turn 3 went silent and marked conversation ended!")

    # Turn 4: Subsequent message after conversation ended
    r4 = respond(state, "Are you still there?", category=category, merchant=merchant, trigger=trigger)
    assert r4["action"] == "end", f"Turn 4 expected end, got {r4['action']}"
    assert not r4.get("body"), f"Turn 4 should be silent (no body), got {r4.get('body')}"
    print(f"  [Turn 4] action={r4['action']} | body={r4.get('body', '<SILENCE>')} | rationale={r4['rationale']}")
    print("  ✓ Turn 4 remained silent — no repeated canned lines!")


def test_repeated_auto_reply_ends_in_silence():
    print("\n── Test 2: Auto-reply Detection & Subsequent Silence ──")
    state = ConversationState(conversation_id="conv_ar_test", merchant_id="m_001")
    ar_msg = "Thank you for contacting us. We will get back to you shortly."

    # Turn 1: 1st auto-reply
    r1 = respond(state, ar_msg, category=category, merchant=merchant, trigger=trigger)
    assert r1["action"] == "send"
    assert "body" in r1 and r1["body"]
    print(f"  [Auto-reply 1] action={r1['action']} | body: {r1['body'][:80]}...")

    # Turn 2: 2nd auto-reply -> exits with silence
    r2 = respond(state, ar_msg, category=category, merchant=merchant, trigger=trigger)
    assert r2["action"] == "end"
    assert not r2.get("body"), "2nd auto-reply must NOT have a body (silence)"
    assert state.ended is True
    print(f"  [Auto-reply 2] action={r2['action']} | body={r2.get('body', '<SILENCE>')}")

    # Turn 3: Subsequent message after ended
    r3 = respond(state, "We are currently closed.", category=category, merchant=merchant, trigger=trigger)
    assert r3["action"] == "end"
    assert not r3.get("body"), "Post-ended message must NOT have a body (silence)"
    print(f"  [Post-ended] action={r3['action']} | body={r3.get('body', '<SILENCE>')}")
    print("  ✓ Auto-reply exit and subsequent silence verified!")


def test_negative_intent_exit_once_then_silent():
    print("\n── Test 3: Negative Intent (Goodbye Once, Then Silence) ──")
    state = ConversationState(conversation_id="conv_neg_test", merchant_id="m_001")

    # Turn 1: Decline
    r1 = respond(state, "Stop messaging me, not interested", category=category, merchant=merchant, trigger=trigger)
    assert r1["action"] == "end"
    assert r1.get("body"), "First negative turn sends a polite goodbye"
    farewell = r1["body"]
    assert state.ended is True
    print(f"  [Turn 1 Farewell] body: {farewell}")

    # Turn 2: Subsequent message -> must be SILENT, no repeated goodbye
    r2 = respond(state, "Don't ever contact again", category=category, merchant=merchant, trigger=trigger)
    assert r2["action"] == "end"
    assert not r2.get("body"), f"Subsequent turn after end must be SILENT, got: {r2.get('body')}"
    print(f"  [Turn 2 Post-End] body={r2.get('body', '<SILENCE>')}")

    # Turn 3: Another subsequent message -> SILENCE
    r3 = respond(state, "Ok?", category=category, merchant=merchant, trigger=trigger)
    assert r3["action"] == "end"
    assert not r3.get("body"), f"Turn 3 after end must be SILENT, got: {r3.get('body')}"
    print(f"  [Turn 3 Post-End] body={r3.get('body', '<SILENCE>')}")
    print("  ✓ Negative exit sent farewell once, followed by absolute silence!")


def test_reopen_after_end_fresh_composition():
    print("\n── Test 4: Explicit Re-engagement After Ended Conversation ──")
    state = ConversationState(conversation_id="conv_reopen_test", merchant_id="m_001")
    
    # End conversation
    respond(state, "Stop", category=category, merchant=merchant, trigger=trigger)
    assert state.ended is True

    # User re-engages with explicit positive intent
    r_reopen = respond(state, "Wait, actually yes, I want to proceed now", category=category, merchant=merchant, trigger=trigger)
    assert r_reopen["action"] == "send", f"Expected send on reopen, got {r_reopen['action']}"
    assert r_reopen.get("body"), "Reopened conversation must compose fresh reply"
    assert state.ended is False, "State should be active again"
    print(f"  [Reopened] action={r_reopen['action']} | body: {r_reopen['body'][:80]}...")
    print("  ✓ Explicit re-contact reactivated cleanly with fresh response!")


def test_fastapi_http_reply_silence():
    print("\n── Test 5: FastAPI HTTP /v1/reply Endpoint ──")
    conv_id = "http_conv_repeat_test"

    # Turn 1
    res1 = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": "Can you give me more details?",
        "received_at": "2026-09-27T12:00:00Z",
        "turn_number": 2
    })
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["action"] == "send"
    assert "body" in d1

    # Turn 2: Same message
    res2 = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": "Can you give me more details?",
        "received_at": "2026-09-27T12:01:00Z",
        "turn_number": 3
    })
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["action"] == "send"
    assert d2["body"] != d1["body"], "HTTP reply 1 and 2 must not be identical"

    # Turn 3: Same message -> Ends in silence
    res3 = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": "Can you give me more details?",
        "received_at": "2026-09-27T12:02:00Z",
        "turn_number": 4
    })
    assert res3.status_code == 200
    d3 = res3.json()
    assert d3["action"] == "end"
    assert "body" not in d3 or not d3["body"], f"Turn 3 HTTP response must not contain body: {d3}"

    # Turn 4: Subsequent message
    res4 = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": "Hello hello",
        "received_at": "2026-09-27T12:03:00Z",
        "turn_number": 5
    })
    assert res4.status_code == 200
    d4 = res4.json()
    assert d4["action"] == "end"
    assert "body" not in d4 or not d4["body"], f"Turn 4 HTTP response must not contain body: {d4}"
    print(f"  ✓ HTTP /v1/reply properly omits body on ended state: {d4}")


if __name__ == "__main__":
    test_repeated_messages_never_identical_and_end_in_silence()
    test_repeated_auto_reply_ends_in_silence()
    test_negative_intent_exit_once_then_silent()
    test_reopen_after_end_fresh_composition()
    test_fastapi_http_reply_silence()
    print("\n✅ ALL REPEAT & SILENCE TESTS PASSED PERFECTLY!")
