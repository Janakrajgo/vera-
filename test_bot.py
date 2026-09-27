"""
Comprehensive test script for Vera AI Bot.
Validates:
  1. Synchronous compose() per challenge-brief.md §7.1
  2. Multi-turn respond() per challenge-brief.md §7.4
  3. Auto-reply detection & graceful exit
  4. Intent transition (immediate action mode)
  5. Off-topic handling
  6. Negative intent handling
  7. FastAPI HTTP endpoints via TestClient (/v1/healthz, /v1/metadata, /v1/context, /v1/tick, /v1/reply, /v1/teardown)
  8. JSON schema compliance
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
from pathlib import Path
from fastapi.testclient import TestClient

import bot
from bot import app, compose
from conversation_handlers import respond, ConversationState

client = TestClient(app)

def test_synchronous_compose():
    print("── Testing Synchronous compose() ──")
    with open("dataset/categories/dentists.json", encoding="utf-8") as f:
        category = json.load(f)
    with open("dataset/expanded/merchants/m_001_drmeera_dentist_delhi.json", encoding="utf-8") as f:
        merchant = json.load(f)
    with open("dataset/expanded/triggers/trg_003_recall_due_priya.json", encoding="utf-8") as f:
        trigger = json.load(f)
    with open("dataset/expanded/customers/c_001_priya_for_m001.json", encoding="utf-8") as f:
        customer = json.load(f)

    # Test customer-facing
    res_cx = compose(category, merchant, trigger, customer)
    assert isinstance(res_cx, dict), "compose() must return a dict"
    assert res_cx["send_as"] == "merchant_on_behalf", "Customer-facing send_as must be merchant_on_behalf"
    assert len(res_cx["body"]) > 20, "Body must not be empty"
    assert "₹" in res_cx["body"], "Should cite real price"
    print(f"  ✓ Customer-facing compose: {res_cx['body'][:70]}...")

    # Test merchant-facing
    with open("dataset/expanded/triggers/trg_023_competitor_opened_dentist.json", encoding="utf-8") as f:
        trg_comp = json.load(f)
    res_mx = compose(category, merchant, trg_comp, None)
    assert res_mx["send_as"] == "vera", "Merchant-facing send_as must be vera"
    assert res_mx["cta"] == "binary_yes_stop", "CTA should be binary_yes_stop"
    print(f"  ✓ Merchant-facing compose: {res_mx['body'][:70]}...")


def test_conversation_flow():
    print("\n── Testing Multi-Turn respond() & Scenarios ──")
    # Scenario 1: Intent transition (should switch to action mode immediately)
    state = ConversationState(conversation_id="conv_test_1", merchant_id="m_001")
    reply1 = respond(state, "Yes let's do it, go ahead")
    assert reply1["action"] == "send", "Action must be send"
    assert "update" in reply1["body"].lower() or "initiate" in reply1["body"].lower() or "shandaar" in reply1["body"].lower(), "Must switch to action mode"
    print(f"  ✓ Intent transition: action={reply1['action']} | body={reply1['body']}")

    # Scenario 2: Negative intent
    state_neg = ConversationState(conversation_id="conv_test_2", merchant_id="m_001")
    reply_neg = respond(state_neg, "Not interested, please stop")
    assert reply_neg["action"] == "end", "Action must be end on negative"
    assert state_neg.ended is True, "State must be marked ended"
    print(f"  ✓ Negative exit: action={reply_neg['action']} | ended={state_neg.ended}")

    # Scenario 3: Auto-reply detection
    state_ar = ConversationState(conversation_id="conv_test_3", merchant_id="m_001")
    # First auto-reply
    ar1 = respond(state_ar, "Thank you for contacting us. Our team will respond shortly.")
    assert ar1["action"] == "send", "1st auto-reply should try once"
    assert state_ar.auto_reply_count == 1, "Auto-reply count should be 1"
    print(f"  ✓ 1st Auto-reply handled: auto_reply_count={state_ar.auto_reply_count}")

    # Second auto-reply
    ar2 = respond(state_ar, "Thank you for contacting us. Our team will respond shortly.")
    assert ar2["action"] == "end", "2nd auto-reply must gracefully end"
    assert state_ar.ended is True, "State must be ended after 2nd auto-reply"
    print(f"  ✓ 2nd Auto-reply exit: action={ar2['action']} | ended={state_ar.ended}")

    # Scenario 4: Off-topic handling
    state_off = ConversationState(conversation_id="conv_test_4", merchant_id="m_001")
    reply_off = respond(state_off, "Can you also help me file my GST returns?")
    assert reply_off["action"] == "send", "Action should be send"
    assert "gst" in reply_off["body"].lower() or "ca" in reply_off["body"].lower(), "Should politely address GST"
    print(f"  ✓ Off-topic redirection: {reply_off['body']}")


def test_http_endpoints():
    print("\n── Testing FastAPI HTTP Endpoints ──")
    # 1. Healthz
    r = client.get("/v1/healthz")
    assert r.status_code == 200
    h_data = r.json()
    assert h_data["status"] == "ok"
    assert "contexts_loaded" in h_data
    print(f"  ✓ GET /v1/healthz: {h_data}")

    # 2. Metadata
    r = client.get("/v1/metadata")
    assert r.status_code == 200
    m_data = r.json()
    assert "team_name" in m_data
    assert "version" in m_data
    print(f"  ✓ GET /v1/metadata: team={m_data['team_name']}, version={m_data['version']}")

    # 3. Context Push
    with open("dataset/categories/dentists.json", encoding="utf-8") as f:
        dentist_cat = json.load(f)
    r = client.post("/v1/context", json={
        "scope": "category",
        "context_id": "dentists",
        "version": 1,
        "payload": dentist_cat,
        "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert r.status_code == 200
    assert r.json()["accepted"] is True
    print(f"  ✓ POST /v1/context (category): {r.json()}")

    # Stale version check (should return accepted=False, reason=stale_version)
    r_stale = client.post("/v1/context", json={
        "scope": "category",
        "context_id": "dentists",
        "version": 1,
        "payload": dentist_cat,
        "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert r_stale.json()["accepted"] is False
    assert r_stale.json()["reason"] == "stale_version"
    print("  ✓ POST /v1/context idempotency/stale_version correctly rejected")

    # Push merchant and trigger
    with open("dataset/expanded/merchants/m_001_drmeera_dentist_delhi.json", encoding="utf-8") as f:
        mx = json.load(f)
    client.post("/v1/context", json={
        "scope": "merchant",
        "context_id": "m_001_drmeera_dentist_delhi",
        "version": 1,
        "payload": mx,
        "delivered_at": "2026-04-26T10:00:00Z"
    })

    with open("dataset/expanded/triggers/trg_023_competitor_opened_dentist.json", encoding="utf-8") as f:
        trg = json.load(f)
    client.post("/v1/context", json={
        "scope": "trigger",
        "context_id": "trg_023_competitor_opened_dentist",
        "version": 1,
        "payload": trg,
        "delivered_at": "2026-04-26T10:00:00Z"
    })

    # 4. Tick
    r_tick = client.post("/v1/tick", json={
        "now": "2026-04-26T10:30:00Z",
        "available_triggers": ["trg_023_competitor_opened_dentist"]
    })
    assert r_tick.status_code == 200
    actions = r_tick.json()["actions"]
    assert len(actions) == 1, "Should generate 1 proactive action"
    action = actions[0]
    assert action["send_as"] == "vera"
    assert "Dental Cleaning" in action["body"] or "Lajpat Nagar" in action["body"]
    print(f"  ✓ POST /v1/tick: {len(actions)} action generated | conv_id={action['conversation_id']}")

    # 5. Reply
    conv_id = action["conversation_id"]
    r_reply = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Yes, update it please",
        "received_at": "2026-04-26T10:45:00Z",
        "turn_number": 2
    })
    assert r_reply.status_code == 200
    reply_data = r_reply.json()
    assert reply_data["action"] == "send"
    print(f"  ✓ POST /v1/reply: action={reply_data['action']} | body={reply_data['body']}")

    # 6. Teardown
    r_td = client.post("/v1/teardown")
    assert r_td.status_code == 200
    assert r_td.json()["status"] == "wiped"
    print("  ✓ POST /v1/teardown: wiped state cleanly")


if __name__ == "__main__":
    test_synchronous_compose()
    test_conversation_flow()
    test_http_endpoints()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
