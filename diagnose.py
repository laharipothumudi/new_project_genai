"""
Quick diagnostic — run while Flask is running on port 8000.
python diagnose.py
"""
import requests, json

BASE = "http://localhost:8000"

tests = [
    {
        "name": "Health check",
        "method": "GET",
        "url": f"{BASE}/health",
        "body": None,
    },
    {
        "name": "Vague query (should clarify)",
        "method": "POST",
        "url": f"{BASE}/chat",
        "body": {"messages": [{"role": "user", "content": "I need an assessment"}]},
    },
    {
        "name": "Clear Java role (should recommend)",
        "method": "POST",
        "url": f"{BASE}/chat",
        "body": {"messages": [
            {"role": "user", "content": "I am hiring a Java developer"},
            {"role": "assistant", "content": "What seniority level?"},
            {"role": "user", "content": "Mid-level, 4 years, Core Java and problem solving"},
        ]},
    },
    {
        "name": "Compare OPQ vs MQ (should compare)",
        "method": "POST",
        "url": f"{BASE}/chat",
        "body": {"messages": [{"role": "user", "content": "What is the difference between OPQ32r and Motivation Questionnaire MQM5?"}]},
    },
    {
        "name": "Off-topic (should refuse)",
        "method": "POST",
        "url": f"{BASE}/chat",
        "body": {"messages": [{"role": "user", "content": "What is the best way to fire an employee?"}]},
    },
]

for t in tests:
    print(f"\n{'='*60}")
    print(f"TEST: {t['name']}")
    try:
        if t["method"] == "GET":
            r = requests.get(t["url"], timeout=35)
        else:
            r = requests.post(t["url"], json=t["body"], timeout=35)
        data = r.json()
        print(f"STATUS: {r.status_code}")
        if "reply" in data:
            print(f"REPLY: {data['reply']}")
            print(f"RECOMMENDATIONS: {len(data.get('recommendations', []))} items")
            for rec in data.get("recommendations", []):
                print(f"  - {rec['name']} [{rec['test_type']}]")
            print(f"END_OF_CONV: {data.get('end_of_conversation')}")
        else:
            print(f"RESPONSE: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"ERROR: {e}")
