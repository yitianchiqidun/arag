"""Stage 9 acceptance of the demo page, run inside the demo container:

    docker compose exec -T demo python -m research_agent.service.demo_check

Drives the container's own demo page (apps/chat_demo.py, with CHAT_DEMO_API_URL pointing at the
API container) headlessly through one parameter question and one diagnosis with a write
confirmation, and prints what the page showed as JSON. Inject the memory quota fault on the
host first (research_agent.diagnosis.scenarios.MemoryQuota) and clean it up afterwards. The
diagnosis question names the variable, so that the run reaches a write proposal; the
diagnosis itself is evaluated by the fault scenarios (D34, D35).
"""

import json
import sys
import time

from streamlit.testing.v1 import AppTest

PARAM = "tidb_mem_quota_query 的默认值是多少？"
DIAG = (
    "每天跑的用户消费排行报表从今天开始报错 ERROR 8175，帮我看看 tidb_mem_quota_query 是不是被改小了；"
    "如果是，请提出恢复成默认值的修改。"
)


def main() -> None:
    out: dict = {}
    at = AppTest.from_file("/app/apps/chat_demo.py", default_timeout=400).run()
    out["page_title"] = [t.value for t in at.title]

    t = time.monotonic()
    at.chat_input[0].set_value(PARAM).run()
    out["param"] = {
        "seconds": round(time.monotonic() - t, 1),
        "answer": [
            m.value
            for m in at.markdown
            if "1" in m.value and ("GB" in m.value or "1073741824" in m.value)
        ][-1:],
        "captions": [c.value for c in at.caption],
    }

    t = time.monotonic()
    at.chat_input[0].set_value(DIAG).run()
    out["diagnosis_ask"] = {
        "seconds": round(time.monotonic() - t, 1),
        "pending_sql": [c.value for c in at.code],
        "approve_button": any(b.key == "approve" for b in at.button),
    }
    if out["diagnosis_ask"]["approve_button"]:
        t = time.monotonic()
        at.button(key="approve").click().run()
        out["diagnosis_confirm"] = {
            "seconds": round(time.monotonic() - t, 1),
            "final_answer": [m.value for m in at.markdown][-2:],
            "captions": [c.value for c in at.caption][-3:],
            "approve_button_left": any(b.key == "approve" for b in at.button),
        }
    out["exceptions"] = [str(e) for e in at.exception]
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
