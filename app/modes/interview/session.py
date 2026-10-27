"""The AI interviewer: does its homework on the JD, resume and repos, then interviews you by voice."""
import asyncio
import json
import logging
import random
import re
import time

from app.config import load
from app.core.session import BaseSession, patience_labels
from app.core.turns import UserTurn
from app.llm import decisions
from app.llm.providers import scaledown_summarize
from app.modes.interview import prep
from app.modes.interview.glossary import fix_transcript

log = logging.getLogger("interview")


def options() -> dict:
    """What the setup screen offers."""
    cfg = load("interview")
    steps = cfg["prep"]["steps"]
    return {
        "defaults": cfg["defaults"], "levels": cfg["levels"], "companies": cfg["companies"],
        "styles": list(cfg["styles"]), "focus_areas": {k: v["title"] for k, v in cfg["focus_areas"].items()},
        "patience": patience_labels(),
        "prep": {"enabled": cfg["prep"]["enabled"], "show_plan": cfg["prep"]["show_plan_to_candidate"],
                 "steps": {name: step.get("enabled", True) for name, step in steps.items()},
                 "question_temperature": steps["questions"]["temperature"],
                 "per_section": steps["questions"]["per_section"], "history": cfg["prep"]["history"]["enabled"]},
        "repos": {"reuse_cached": cfg["repos"]["reuse_cached"]},
    }


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def _scores(grades: dict) -> str:
    """Jev's 0-3 scores as "depth 3.0/4, ..."."""
    return ", ".join(f"{name} {grade['score'] + 1:.1f}/4" for name, grade in grades.items()
                     if isinstance(grade, dict) and "score" in grade)


def _overlap(said: str, planned: str) -> float:
    """Share of the planned question's words that were actually said."""
    planned_words = _words(planned)
    return len(_words(said) & planned_words) / len(planned_words) if planned_words else 0.0


class InterviewSession(BaseSession):
    mode = "interview"

    async def setup(self, msg: dict) -> None:
        self.cfg = cfg = load("interview", msg.get("overrides"))
        chosen = {k: v for k, v in (msg.get("setup") or {}).items() if v not in (None, "")}
        brief = {**cfg["defaults"], **chosen}
        brief["duration_min"] = self.capped(max(5, min(90, int(brief["duration_min"]))))
        brief["focus"] = [f for f in brief.get("focus", []) if f in cfg["focus_areas"]] or cfg["defaults"]["focus"]
        brief["focus_titles"] = [cfg["focus_areas"][f]["title"] for f in brief["focus"]]
        brief["candidate_name"] = brief.get("candidate_name") or "the candidate"
        self.brief = brief
        self.use_llm(cfg, 0.75, 400)
        await self.set_voice(brief.get("tts_engine", "edge"), brief.get("voice", ""), brief.get("speed", 1.0),
                             brief.get("interviewer_accent", "Indian"), brief.get("interviewer_gender", "male"), 40)
        self.apply_turn_config(brief.get("patience", "patient"), bool(brief.get("manual_turns")))

        self.sections = self._standard_plan()
        self.section = 0
        self.section_started = 0.0
        self.prep_result: dict = {}
        self.asked: set[str] = set()
        self.assessment = ""  # the grade of the last answer, for the interviewer's eyes only
        self.repos = [r for r in msg.get("repos") or [] if (r.get("source") or "").strip()]
        self.prep_task = asyncio.create_task(self._prepare()) if cfg["prep"].get("enabled", True) else None
        await self.send({"type": "ready", "sections": self._outline(), "preparing": bool(self.prep_task)})

    def _outline(self) -> list[dict]:
        return [{"title": s["title"], "minutes": s["minutes"]} for s in self.sections]

    async def _prepare(self) -> None:
        async def progress(step: str, status: str, detail: str) -> None:
            await self.send({"type": "prep", "step": step, "status": status, "detail": detail})

        try:
            self.prep_result = await prep.run(self.brief, self.cfg, self.repos, progress)
            if plan := self.prep_result.get("plan"):
                self.sections = prep.normalise_sections(plan, self.brief["duration_min"])
            done = {"type": "prep_done", "seconds": self.prep_result.get("seconds"), "sections": self._outline()}
            if self.cfg["prep"].get("show_plan_to_candidate"):
                done["plan"] = self.sections
            await self.send(done)
        except Exception as e:  # prep is a bonus; the interview goes ahead without it
            log.exception("prep failed")
            await self.send({"type": "prep_done", "error": str(e)[:300]})

    async def ready(self) -> None:
        if self.prep_task and not self.prep_task.done():
            await self.send({"type": "status", "text": "Finishing interview preparation..."})
            await self.prep_task
        self.section_started = time.monotonic()

    def _standard_plan(self) -> list[dict]:
        """Used when prep is off or fails: sections from the chosen focus areas, sized by section_weights."""
        cfg, brief, weight = self.cfg, self.brief, self.cfg["section_weights"]
        extra = [k for k in ("system_design", "behavioral") if k in brief["focus"]]
        technical = [f for f in brief["focus"] if f not in extra]
        has_resume = bool((brief.get("resume") or "").strip())
        technical_share = (weight["technical_total"] - (weight["resume"] if has_resume else 0)
                           - sum(weight[k] for k in extra))
        parts = ([("intro", weight["intro"])]
                 + ([("resume", weight["resume"])] if has_resume else [])
                 + [(f, max(technical_share, 0.2) / max(len(technical), 1)) for f in technical]
                 + [(k, weight[k]) for k in extra]
                 + [("close", weight["close"])])
        total = sum(share for _, share in parts)
        plan = []
        for key, share in parts:
            if key in cfg["standard_sections"]:
                title, goal = cfg["standard_sections"][key]["title"], cfg["standard_sections"][key]["goal"]
            else:
                title, goal = cfg["focus_areas"][key]["title"], cfg["focus_areas"][key]["guidance"]
            plan.append({"title": title, "goal": goal, "questions": [],
                         "minutes": round(brief["duration_min"] * share / total, 1)})
        return plan

    def normalize(self, text: str) -> str:
        return fix_transcript(text)

    def whisper_hint(self) -> str:
        return self.cfg.get("whisper_hint", "")

    def _goto(self, index: int) -> None:
        self.section, self.section_started = index, time.monotonic()
        self.spawn(self.send({"type": "section", "index": index, "title": self.sections[index]["title"]}))

    def _minutes_left(self) -> float:
        return self.brief["duration_min"] - (time.monotonic() - self.started_at) / 60

    def before_reply(self) -> None:
        """Keep to the clock: jump to the close near the end, and leave a section that ran far over time."""
        last = len(self.sections) - 1
        spent = (time.monotonic() - self.section_started) / 60
        if self._minutes_left() <= 0.75 and self.section < last:
            self._goto(last)
        elif self.section < last - 1 and spent > self.sections[self.section]["minutes"] * 1.8:
            self._goto(self.section + 1)

    def system_prompt(self) -> str:
        cfg, brief = self.cfg, self.brief
        last, left = len(self.sections) - 1, self._minutes_left()
        current, spent = self.sections[self.section], (time.monotonic() - self.section_started) / 60
        plan = "\n".join(
            f"{'->' if i == self.section else 'done' if i < self.section else '  '} {i + 1}. {s['title']} "
            f"(~{s['minutes']:g} min): {s.get('goal', '')}" for i, s in enumerate(self.sections))
        if left <= 1.5 and self.section != last:
            pace = "TIME IS ALMOST UP: move to candidate questions and close now."
        elif spent > current["minutes"] * 1.15 and self.section < last:
            pace = (f"This section is over time: wrap it up and move to "
                    f"'{self.sections[self.section + 1]['title']}' (advance=true).")
        else:
            pace = ""
        notes = f"\nEarlier in this interview (summary):\n{self.summary}\n" if self.summary else ""
        if self.assessment:
            notes += f"\nPrivate assessment of the candidate's last answer (never mention it): {self.assessment}\n"
        return cfg["persona"].format(
            interviewer=brief["interviewer_name"], company=brief["company"], name=brief["candidate_name"],
            level=brief["level"], role=brief["role"], style=cfg["styles"].get(brief["style"], ""),
            context=self._context(), duration=brief["duration_min"], plan=plan, current=current["title"],
            spent=spent, budget=current["minutes"], left=left, pace=pace,
            section_detail=self._section_detail(current), notes=notes)

    def _section_detail(self, section: dict) -> str:
        todo = [q for q in section.get("questions", []) if q.get("q") not in self.asked][:4]
        if not todo:
            return ""
        header = "Planned questions for this section (not yet asked; adapt wording, improvise follow-ups):"
        lines = [f"- {q['q']}\n  follow-ups: {'; '.join(q.get('follow_ups', [])[:3])}\n"
                 f"  strong answer covers: {'; '.join(q.get('strong_answer', [])[:4])}" for q in todo]
        return "\n".join([header, *lines])

    def _context(self) -> str:
        """What prep learned about the role and the candidate, or the raw JD and resume without prep."""
        found, brief = self.prep_result or {}, self.brief
        context = []
        if (found.get("plan") or {}).get("opening_context"):
            context.append(f"Role context: {found['plan']['opening_context']}")
        if found.get("jd"):
            context.append("JD must-haves: " + "; ".join(found["jd"].get("must_haves", [])[:8]))
        if found.get("resume"):
            context.append(f"Candidate: {found['resume'].get('candidate_summary', '')} | Claims to verify: "
                           + "; ".join(found["resume"].get("claims_to_verify", [])[:4]))
        for repo in found.get("repos", [])[:3]:
            context.append(f"Their project '{repo['project']}' (from its code): {repo.get('summary', '')} "
                           f"Weak spots: {'; '.join(repo.get('weaknesses_or_risks', [])[:3])}")
        if not context and brief.get("resume"):
            context.append("Resume:\n" + brief["resume"][:3000])
        if brief.get("jd") and not found.get("jd"):
            context.append("Job description:\n" + brief["jd"][:1500])
        return "\n".join(context)

    def opening(self) -> str | None:
        return self.cfg["opening"].format(name=self.brief["candidate_name"], duration=self.brief["duration_min"])

    def cue_hint(self) -> str:
        return (self.cfg.get("style_cues") or {}).get(self.brief["style"], "")

    def _planned(self) -> list[dict]:
        return [q for section in self.sections for q in section.get("questions", []) if q.get("q")]

    def on_reply(self, spoken: str, meta: dict) -> None:
        if meta.get("advance") is True and self.section < len(self.sections) - 1:
            self._goto(self.section + 1)
        newly_asked = [q["q"] for q in self._planned()
                       if q["q"] not in self.asked and _overlap(spoken, q["q"]) >= 0.45]
        if newly_asked:
            self.asked.update(newly_asked)
            prep.history_add(self.cfg["prep"], prep.candidate_key(self.brief), newly_asked)

    def on_idle(self, idle_s: float) -> tuple[str, str] | None:
        nudges = self.cfg.get("nudges") or []
        if self.idle_step < len(nudges) and idle_s > nudges[self.idle_step]["after_s"]:
            return "say", random.choice(nudges[self.idle_step]["lines"])
        return None

    def time_limit_min(self) -> float:
        return float(self.brief["duration_min"])

    def memory_cfg(self) -> dict:
        return self.cfg["memory"]

    async def summarize(self, text: str, prompt: str) -> str | None:
        if self.cfg["memory"].get("summarizer") == "scaledown" and (notes := await scaledown_summarize(text, prompt)):
            return notes
        return await super().summarize(text, prompt)

    async def on_user_turn(self, turn: UserTurn) -> None:
        """Jev grades each answer quietly. The interviewer sees the grade; the candidate never does."""
        grading = self.cfg.get("assessment", {})
        if not grading.get("enabled", True) or not decisions.available():
            return
        if len(turn.text.split()) < grading.get("min_words", 8):
            return
        turn.notes["section"] = self.sections[self.section]["title"]
        self.spawn(self._assess(turn, grading["questions"]))

    async def _assess(self, turn: UserTurn, questions: dict) -> None:
        best = max(self._planned(), key=lambda q: _overlap(turn.partner_said, q["q"]), default=None)
        rubric = best.get("strong_answer", []) if best and _overlap(turn.partner_said, best["q"]) >= 0.3 else []
        state = (f"Interview section: {turn.notes['section']}\nInterviewer question: {turn.partner_said}\n"
                 + (f"A strong answer covers: {'; '.join(rubric)}\n" if rubric else "")
                 + f"Candidate answer: {turn.text}")
        if not (grades := await decisions.decide(state, questions, timeout=10)):
            return
        turn.notes["assessment"] = grades
        parts = [f"{name} {grades[name]['score'] + 1:.1f}/4" for name in ("correctness", "depth", "communication")
                 if "score" in grades.get(name, {})]
        if grades.get("on_topic", {}).get("noul", 1) < 0.5:
            parts.append("did NOT really answer the question")
        if "choice" in grades.get("next_move", {}):
            parts.append(f"suggested next move: {grades['next_move']['choice'].replace('_', ' ')}")
        self.assessment = "; ".join(parts)
        await self.send({"type": "assessment", "text": self.assessment})

    async def mode_report(self, transcript: str) -> str:
        rubric = "\n".join(f"- {q['q']} -> {'; '.join(q.get('strong_answer', []))}"
                           for q in self._planned() if q["q"] in self.asked) or "(no planned questions)"
        scores = "\n".join(f"A{i}: {_scores(turn.notes['assessment'])}"
                           for i, turn in enumerate(self.turns, 1) if turn.notes.get("assessment")) or "(none)"
        report = self.cfg["report"]
        must_haves = ((self.prep_result or {}).get("jd") or {}).get("must_haves", [])
        return await self.write_report(report, report["prompt"].format(
            level=self.brief["level"], role=self.brief["role"], company=self.brief["company"],
            name=self.brief["candidate_name"], minutes=self.turns[-1].minute if self.turns else 0,
            must_haves=json.dumps(must_haves), rubric=rubric, scores=scores,
            transcript=transcript.replace("Partner:", "Interviewer:").replace("User:", "Candidate:")[:60000]),
            max_tokens=3500)

    def extra_record(self) -> dict:
        prep_record = {k: v for k, v in (self.prep_result or {}).items() if k != "plan"}
        return {"setup": {k: v for k, v in self.brief.items() if k != "resume"}, "sections": self.sections,
                "asked": sorted(self.asked), "prep": prep_record}
