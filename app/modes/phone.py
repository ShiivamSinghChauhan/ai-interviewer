"""Phone calls: a stranger rings you with their own agenda, or you have to call someone and get a thing done.

Scenarios in config/phone.yaml are only seeds. A generator at high temperature invents the actual caller, facts
and twist for every call, so the delivery guy is never the same delivery guy twice. The caller hangs up by
sending `##META {"end": true}`.
"""
import json
import logging
import random

from app.config import load, settings
from app.core.session import BaseSession, patience_labels
from app.llm.providers import LLMUnavailable, complete_json

log = logging.getLogger("phone")


def options() -> dict:
    """What the setup screen offers."""
    cfg, common = load("phone"), load("common")
    return {
        "defaults": cfg["defaults"], "categories": cfg["categories"], "patience": patience_labels(),
        "difficulties": {k: v["label"] for k, v in cfg["difficulties"].items()},
        "lines": {k: v["label"] for k, v in cfg["lines"].items()}, "accents": list(common["accents"]),
        "scenarios": [{k: s[k] for k in ("id", "title", "category", "direction")} for s in cfg["scenarios"]],
    }


def pick_scenario(cfg: dict, prefs: dict) -> dict:
    chosen = [s for s in cfg["scenarios"] if s["id"] == prefs.get("scenario")]
    matching = [s for s in cfg["scenarios"]
                if prefs.get("category", "any") in ("any", s["category"])
                and prefs.get("direction", "any") in ("any", s["direction"])]
    return random.choice(chosen or matching or cfg["scenarios"])


class PhoneSession(BaseSession):
    mode = "phone"

    async def setup(self, msg: dict) -> None:
        self.cfg = cfg = load("phone", msg.get("overrides"))
        chosen = {k: v for k, v in (msg.get("setup") or {}).items() if v not in (None, "")}
        self.prefs = prefs = {**cfg["defaults"], **chosen}
        prefs["duration_min"] = self.capped(float(prefs.get("duration_min") or 8))
        self.difficulty = cfg["difficulties"].get(prefs["difficulty"]) or cfg["difficulties"]["medium"]
        self.scenario = scenario = pick_scenario(cfg, prefs)
        self.incoming = scenario["direction"] == "incoming"
        self.declined = 0
        self.call = await self._invent_call()
        self.use_llm(cfg, 0.8, 220)

        accents = self.common["accents"]
        accent = self.call.get("caller_accent") if self.call.get("caller_accent") in accents else "Indian"
        gender = "female" if str(self.call.get("caller_gender", "")).lower().startswith("f") else "male"
        try:
            age = int(self.call.get("age") or 0) or None
        except (TypeError, ValueError):
            age = None
        speed = self.difficulty.get("speed", 1.0) * float(prefs.get("speed") or 1.0)
        await self.set_voice(prefs.get("tts_engine", "edge"), prefs.get("voice", ""), speed, accent, gender, age)
        self.apply_turn_config(prefs.get("patience") or self.difficulty.get("patience", "patient"),
                               bool(prefs.get("manual_turns")))
        line = prefs.get("line") or self.difficulty.get("line", "mobile")
        await self.send({"type": "ready", "call": {
            "direction": scenario["direction"],
            "caller_id": self.call.get("caller_id_display") or "Unknown number",
            "callee": "" if self.incoming else (self.call.get("organisation") or self.call.get("caller_name", "")),
            "user_task": "" if self.incoming else (self.call.get("user_task") or scenario.get("user_task", "")),
            "line": cfg["lines"].get(line) or {},
            "ring_delay_s": prefs.get("ring_delay_s", [2, 8]),
            "surprise_window_min": prefs.get("surprise_window_min", 0),
            "callback_after_decline_s": prefs.get("callback_after_decline_s", 20)}})

    async def _invent_call(self) -> dict:
        """The concrete caller for this call: name, facts, mood, twist. Falls back to the bare scenario."""
        cfg, prefs, scenario = self.cfg, self.prefs, self.scenario
        generator = cfg["generator"]
        prompt = generator["prompt"].format(
            region=prefs.get("region", "India"), seed=json.dumps(scenario), direction=scenario["direction"],
            direction_hint=("the caller phones the user" if self.incoming
                            else "the user phones this person, who answers"),
            difficulty=self.difficulty["caller"], accents=", ".join(self.common["accents"]),
            language=prefs.get("language", "English"),
            user_name=prefs.get("user_name") or ("unknown - the caller may know it from their records "
                                                 "(invent one only if natural) or ask"))
        try:
            call = await complete_json([{"role": "user", "content": prompt}],
                                       generator.get("chain") or settings.report_chain,
                                       generator.get("temperature", 1.1), 1200, retries=0)
        except LLMUnavailable as e:
            log.warning("couldn't invent the call (%s); using the scenario as it is", e)
            call = {}
        defaults = {"caller_name": "Rahul", "purpose": scenario.get("purpose", ""), "opening_line": "Hello?",
                    "needs_from_user": scenario.get("needs_from_user", []),
                    "end_when": "the purpose of the call is resolved"}
        return {**defaults, **(call if isinstance(call, dict) else {})}

    def system_prompt(self) -> str:
        cfg, scenario, call = self.cfg, self.scenario, self.call
        return cfg["prompt"].format(
            caller_name=call["caller_name"], age=call.get("age", 35), role=scenario["caller"],
            org=f" (you are from {call['organisation']})" if call.get("organisation") else "",
            purpose=call["purpose"], facts="; ".join(map(str, call.get("facts", []))) or "-",
            needs="; ".join(map(str, call.get("needs_from_user", []))) or "-",
            twist=call.get("twist") or random.choice(scenario.get("twists") or ["none"]),
            rules=scenario.get("rules", "none"), mood=call.get("mood", "neutral"),
            speaking_style=call.get("speaking_style", ""), difficulty_caller=self.difficulty["caller"],
            direction_line=cfg["text"]["incoming_line" if self.incoming else "outgoing_line"],
            end_when=call["end_when"], language=self.prefs.get("language", "English"),
            summary=f"\nEarlier in this call (summary):\n{self.summary}" if self.summary else "")

    def opening(self) -> str | None:
        key = "incoming_opening" if self.incoming else "outgoing_opening"
        return self.cfg["text"][key].format(line=self.call["opening_line"])

    def speakable(self, sentence: str) -> str:
        """One delivery for the whole call (e.g. [in a hurry tone]); a cue the caller wrote itself wins."""
        tag = self.difficulty.get("tags", "")
        if tag and self.acting() and not sentence.lstrip().startswith("["):
            return f"{tag} {sentence}"
        return sentence

    def cue_hint(self) -> str:
        return self.difficulty.get("cue_hint", "")

    def on_idle(self, idle_s: float) -> tuple[str, str] | None:
        idle = self.cfg["idle"]
        if idle_s > idle.get("hang_up_after_s", 45) and self.idle_step >= 2:
            self.idle_step = 99  # one goodbye is enough
            return "llm", idle["hang_up"]
        hello_after = self.difficulty.get("hello_after_s", 9) * (self.idle_step + 1)
        if self.idle_step < len(idle["lines"]) and idle_s > hello_after:
            return "say", idle["lines"][self.idle_step]
        return None

    def time_limit_min(self) -> float:
        return self.prefs["duration_min"]

    def memory_cfg(self) -> dict:
        return self.cfg["memory"]

    async def control(self, msg: dict) -> None:
        if msg.get("type") == "declined":
            self.declined += 1

    async def mode_report(self, transcript: str) -> str:
        report = self.cfg["report"]
        caller = self.call["caller_name"]
        return await self.write_report(report, report["prompt"].format(
            scenario=json.dumps({"title": self.scenario["title"], **self.call}), direction=self.scenario["direction"],
            transcript=transcript.replace("Partner:", f"{caller}:").replace("User:", "You:")[:40000],
            success="; ".join(map(str, self.call.get("success_criteria", []))) or self.call["end_when"]))

    def extra_record(self) -> dict:
        return {"scenario": self.scenario, "call": self.call, "setup": self.prefs, "declined": self.declined}
