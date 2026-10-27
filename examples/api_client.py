"""Create a session through the API, hand out the link, wait for the result.

    python examples/api_client.py --url https://shiivamchauhan-practice-studio.hf.space
    python examples/api_client.py --url http://127.0.0.1:8000 --token <one of API_TOKENS> --agent phone

Open the printed join_url in Chrome or Edge, talk to the agent, and the script prints every answer with its
scores, the delivery metrics and the agent's report. A real integration passes webhook_url instead of polling.
"""
import argparse
import json
import time

import httpx

SESSIONS = {
    "interview": {
        "agent": "interview",
        "setup": {"candidate_name": "Priya", "role": "Backend Engineer", "level": "junior", "duration_min": 10,
                  "interviewer_name": "Arjun",
                  "jd": "Build payment APIs in Python (FastAPI) on PostgreSQL: caching, idempotency, testing."},
        "metadata": {"candidate_id": "c-1042"},
    },
    "phone": {"agent": "phone", "setup": {"scenario": "book_doctor", "duration_min": 5}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default="", help="one of the server's API_TOKENS, if it has any")
    ap.add_argument("--agent", choices=list(SESSIONS), default="interview")
    args = ap.parse_args()
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}

    with httpx.Client(base_url=args.url, headers=headers, timeout=30) as api:
        session = api.post("/api/v1/sessions", json=SESSIONS[args.agent]).raise_for_status().json()
        print("Open this link and talk to the agent:", session["join_url"])
        while session["status"] in ("created", "live"):
            time.sleep(5)
            session = api.get(f"/api/v1/sessions/{session['id']}").raise_for_status().json()

    print("\nstatus:", session["status"])
    result = session.get("result") or {}
    for i, turn in enumerate(result.get("turns", []), 1):
        grades = (turn.get("notes") or {}).get("assessment") or {}
        scores = {name: g["score"] for name, g in grades.items() if isinstance(g, dict) and "score" in g}
        print(f"\nQ{i}: {turn['partner_said']}\nA{i}: {turn['verbatim'] or turn['text']}")
        if scores:
            print(f"    scores (0-3): {scores}")
    delivery = {k: v for k, v in (result.get("delivery") or {}).items() if k != "per_turn"}
    print("\ndelivery:", json.dumps(delivery))
    print("\n" + result.get("report_markdown", ""))


if __name__ == "__main__":
    main()
