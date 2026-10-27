"""A persona agent you design: who they are, their accent, job and personality, and what the conversation is for
(a support call, a sales pitch, a role-play, language practice or just a chat)."""
import random

from app.config import load
from app.core.session import BaseSession, patience_labels

CHOICES = {"mode": "modes", "scene": "scenes", "relationship": "relationships", "level": "levels",
           "length": "lengths", "corrections": "corrections"}  # setup field -> table in conversation.yaml


def options() -> dict:
    """What the setup screen offers."""
    cfg, accents = load("conversation"), load("common")["accents"]
    labels = cfg["labels"]
    return {
        "defaults": cfg["defaults"], "accents": list(accents), "patience": patience_labels(),
        "modes": labels["modes"], "scenes": cfg["scenes"], "relationships": cfg["relationships"],
        "levels": labels["levels"], "lengths": labels["lengths"], "corrections": labels["corrections"],
        "traits": cfg["traits"], "presets": cfg["presets"], "interests": cfg["random"]["interests"],
    }


def surprise() -> dict:
    """A random person, for the "Surprise me" button."""
    cfg = load("conversation")
    pick = cfg["random"]
    accent, gender = random.choice(list(pick["names"])), random.choice(["male", "female"])
    return {"name": random.choice(pick["names"][accent][gender]), "age": random.randint(20, 68), "gender": gender,
            "accent": accent, "hometown": "", "occupation": random.choice(pick["jobs"]),
            "interests": random.sample(pick["interests"], 3),
            "traits": {trait: random.randint(1, 5) for trait in cfg["traits"]},
            "mode": random.choice(pick["modes"]), "scene": random.choice(list(cfg["scenes"])),
            "relationship": "stranger"}


class ConversationSession(BaseSession):
    mode = "conversation"

    async def setup(self, msg: dict) -> None:
        self.cfg = cfg = load("conversation", msg.get("overrides"))
        chosen = {k: v for k, v in (msg.get("setup") or {}).items() if v not in (None, "")}
        persona = {**cfg["defaults"], **chosen}
        persona["traits"] = {t: max(1, min(5, int((persona.get("traits") or {}).get(t, 3)))) for t in cfg["traits"]}
        for key, table in CHOICES.items():
            if persona.get(key) not in cfg[table]:
                persona[key] = next(iter(cfg[table]))
        accents = self.common["accents"]
        if persona.get("accent") not in accents:
            persona["accent"] = next(iter(accents))
        persona["duration_min"] = self.capped(float(persona.get("duration_min") or 15))
        self.persona = persona
        self.use_llm(cfg, 0.85, 350)

        engine = persona.get("tts_engine", "edge")
        gender = "male" if persona["gender"] == "male" else "female"
        voice = persona.get("voice") or (accents[persona["accent"]][gender] if engine == "edge" else "")
        await self.set_voice(engine, voice, persona.get("speed", 1.0), persona["accent"], persona["gender"],
                             int(persona.get("age") or 0) or None)
        self.apply_turn_config(persona.get("patience", "very_patient"), bool(persona.get("manual_turns")))
        await self.send({"type": "ready", "persona": persona})

    def system_prompt(self) -> str:
        p, cfg, text = self.persona, self.cfg, self.cfg["text"]
        return cfg["prompt"].format(
            name=p["name"], age=p["age"], gender=p["gender"], occupation=p["occupation"], accent=p["accent"],
            hometown=p.get("hometown") or cfg["homes"].get(p["accent"], "a city you can describe if asked"),
            backstory=f"Your background: {p['backstory']}" if p.get("backstory") else text["no_backstory"],
            relationship=cfg["relationships"][p["relationship"]], scene=cfg["scenes"][p["scene"]],
            mode=cfg["modes"][p["mode"]],
            topic=text["topic"].format(topic=p["topic"]) if p.get("topic") else "",
            scenario=f"Scenario: {p['scenario']}\n" if p.get("scenario") else "",
            traits=", ".join(cfg["traits"][trait][level - 1] for trait, level in p["traits"].items()),
            interests=", ".join(p.get("interests") or []) or "everyday life",
            length=cfg["lengths"][p["length"]], level=cfg["levels"][p["level"]],
            keep_talking=text["keep_talking"] if p.get("keep_me_talking", True) else "",
            corrections=cfg["corrections"][p["corrections"]],
            character=text["in_character" if p.get("stay_in_character", True) else "may_admit_ai"],
            user=f"The user's name is {p['user_name']}.\n" if p.get("user_name") else "",
            summary=f"\nEarlier in this conversation (summary):\n{self.summary}\n" if self.summary else "")

    def cue_hint(self) -> str:
        """Acting style from the personality sliders: cue_hints.<trait>.<level> in the config."""
        hints = [levels.get(self.persona["traits"].get(trait, 3))
                 for trait, levels in (self.cfg.get("cue_hints") or {}).items()]
        hints = [h for h in hints if h]
        return f"Your delivery: {'; '.join(hints)}." if hints else ""

    def opening(self) -> str | None:
        if self.persona.get("who_starts", "them") == "them":
            return self.cfg["opening"].format(scene=self.cfg["scenes"][self.persona["scene"]])
        self._add("user", self.cfg["user_starts"])
        return None

    def on_idle(self, idle_s: float) -> tuple[str, str] | None:
        idle = self.cfg.get("idle", {})
        if self.idle_step == 0 and idle.get("after_s") and idle_s > idle["after_s"]:
            return "llm", idle["instruction"]
        return None

    def time_limit_min(self) -> float:
        return self.persona["duration_min"]

    def memory_cfg(self) -> dict:
        return self.cfg["memory"]

    async def mode_report(self, transcript: str) -> str:
        p, feedback = self.persona, self.cfg["feedback"]
        return await self.write_report(feedback, feedback["prompt"].format(
            partner=f"{p['name']}, {p['age']}, {p['occupation']}", minutes=self.turns[-1].minute if self.turns else 0,
            transcript=transcript.replace("Partner:", f"{p['name']}:")[:50000]))

    def extra_record(self) -> dict:
        return {"persona": self.persona}
