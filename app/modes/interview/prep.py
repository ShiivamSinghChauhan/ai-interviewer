"""Interview homework, done before the first question:

  1. read the JD (it wins every argument)
  2. read the resume against the JD
  3. read the code behind the projects (cached per commit, see repos.py)
  4. plan the sections, then write questions per section at high temperature, skipping earlier sessions' ones
  5. let a bar-raiser swap out the weakest questions

Everything (steps, models, temperatures, prompts) lives under `prep:` in config/interview.yaml. Steps 1-2 are
cached by text hash, so the same JD and resume are only read once. The work is split into small calls on
purpose: Groq's free tier allows 8K tokens a minute, and one big call doesn't fit.
"""
import hashlib
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.config import ROOT, settings
from app.llm.providers import LLMUnavailable, complete_json
from app.modes.interview import repos

log = logging.getLogger("prep")
Progress = Callable[[str, str, str], Awaitable[None]]  # step, status (running | done | failed), detail


def _hash(*parts: str) -> str:
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


async def _ask(step: dict, prompt: str, **params) -> dict:
    return await complete_json([{"role": "user", "content": prompt}], step.get("chain") or settings.report_chain,
                               step.get("temperature", 0.3), step.get("max_tokens", 3000), retries=1, **params)


async def _cached(prep_cfg: dict, kind: str, key: str, step: dict, prompt: str, progress: Progress,
                  running: str, summary: Callable[[dict], str]) -> dict | None:
    """Run one analysis step, or reuse the saved result for the same input."""
    path = ROOT / prep_cfg["cache_dir"] / f"{kind}-{key}.json"
    if prep_cfg.get("cache_analyses", True) and (saved := _read_json(path)):
        await progress(kind, "done", "reused the saved analysis")
        return saved
    await progress(kind, "running", running)
    try:
        result = await _ask(step, prompt)
    except (LLMUnavailable, ValueError) as e:
        await progress(kind, "failed", str(e)[:200])
        return None
    _write_json(path, result)
    await progress(kind, "done", summary(result))
    return result


# The question history keeps interviews from repeating themselves.

def candidate_key(brief: dict) -> str:
    return _hash((brief.get("candidate_name") or "").lower().strip(), (brief.get("resume") or "")[:3000])


def history_load(prep_cfg: dict, key: str) -> list[str]:
    if not prep_cfg["history"].get("enabled", True):
        return []
    return (_read_json(ROOT / prep_cfg["history"]["dir"] / f"{key}.json") or {}).get("questions", [])


def history_add(prep_cfg: dict, key: str, questions: list[str]) -> None:
    if not prep_cfg["history"].get("enabled", True) or not questions:
        return
    known = history_load(prep_cfg, key)
    known += [q for q in questions if q not in known]
    keep = prep_cfg["history"].get("max_remembered", 300)
    _write_json(ROOT / prep_cfg["history"]["dir"] / f"{key}.json", {"questions": known[-keep:]})


async def extract_projects(resume: str, cfg: dict) -> list[dict]:
    """Projects named in the resume, for the "attach a repo" rows on the setup screen."""
    if not resume.strip():
        return []
    path = ROOT / cfg["prep"]["cache_dir"] / f"projects-{_hash('projects', resume)}.json"
    if saved := _read_json(path):
        return saved.get("projects", [])
    prompt = ("List the distinct projects in this resume (personal, academic or work projects). "
              'Return ONLY JSON: {"projects": [{"name": "...", "one_line": "...", "links": ["any URLs mentioned"]}]}'
              "\n\n" + resume[:12000])
    try:
        found = await _ask({"chain": settings.llm_chain, "temperature": 0.1, "max_tokens": 1200}, prompt)
    except (LLMUnavailable, ValueError) as e:
        log.warning("project extraction failed: %s", e)
        return []
    _write_json(path, found)
    return found.get("projects", [])


async def run(brief: dict, cfg: dict, attached_repos: list[dict], progress: Progress) -> dict:
    prep_cfg, steps = cfg["prep"], cfg["prep"]["steps"]
    jd, resume = (brief.get("jd") or "").strip(), (brief.get("resume") or "").strip()
    started = time.time()
    result: dict = {"jd": None, "resume": None, "repos": [], "plan": None}

    if jd and steps["jd"].get("enabled", True):
        result["jd"] = await _cached(
            prep_cfg, "jd", _hash(jd, brief["role"], brief["level"], brief["company"]), steps["jd"],
            steps["jd"]["prompt"].format(jd=jd[:15000], role=brief["role"], level=brief["level"],
                                         company=brief["company"]),
            progress, "analysing the job description", lambda d: f"{len(d.get('must_haves', []))} must-haves found")

    if resume and steps["resume"].get("enabled", True):
        result["resume"] = await _cached(
            prep_cfg, "resume", _hash(resume, json.dumps(result["jd"], sort_keys=True)), steps["resume"],
            steps["resume"]["prompt"].format(jd_analysis=json.dumps(result["jd"] or {"note": "no JD given"}),
                                             resume=resume[:15000]),
            progress, "reading your resume against the JD",
            lambda d: f"{len(d.get('projects', []))} projects, {len(d.get('claims_to_verify', []))} claims to verify")

    if steps["repos"].get("enabled", True):
        for repo in attached_repos:
            result["repos"].extend(await _study_repo(repo, cfg, progress))

    avoid = "\n".join(f"- {q}" for q in history_load(prep_cfg, candidate_key(brief))[-60:]) or "none"
    if steps["outline"].get("enabled", True) and steps["questions"].get("enabled", True):
        result["plan"] = await _plan(brief, steps, result, avoid, progress)
    if result["plan"] and steps["critique"].get("enabled", True):
        await _critique(steps["critique"], result, avoid, progress)
    result["seconds"] = round(time.time() - started, 1)
    return result


async def _study_repo(repo: dict, cfg: dict, progress: Progress) -> list[dict]:
    label = repo.get("project") or repo["source"]
    step = f"repo:{label}"

    async def report(stage: str, detail: str) -> None:
        await progress(step, "done" if stage in ("done", "cached") else "running", detail)

    try:
        analysis = await repos.analyze(repo["source"], cfg["repos"], report, bool(repo.get("force")))
        return [{"project": label, **analysis}]
    except Exception as e:  # one broken repo shouldn't cancel the interview
        log.warning("repo %s failed: %s", repo["source"], e)
        await progress(step, "failed", str(e)[:200])
        return []


async def _plan(brief: dict, steps: dict, result: dict, avoid: str, progress: Progress) -> dict | None:
    """Stage 1 designs the sections; stage 2 writes the questions, one section per call."""
    outline_step, questions_step = steps["outline"], steps["questions"]
    await progress("plan", "running", "designing the interview strategy")
    repo_notes = "\n".join(
        f"- {r['project']}: {r.get('summary', '')} | stack: {', '.join(r.get('tech_stack', [])[:8])} | "
        f"weak spots: {'; '.join(r.get('weaknesses_or_risks', [])[:3])} | verify: "
        f"{'; '.join(r.get('claims_to_verify', [])[:3])}" for r in result["repos"]) or "none attached"
    try:
        outline = await _ask(outline_step, outline_step["prompt"].format(
            duration_min=brief["duration_min"], style=brief["style"], level=brief["level"], role=brief["role"],
            company=brief["company"], jd_priority=outline_step.get("jd_priority", ""),
            jd_analysis=json.dumps(result["jd"] or {"note": "no JD given; infer from role and level"}),
            resume_analysis=json.dumps(result["resume"] or {"note": "no resume given"}),
            repo_reports=repo_notes, focus=", ".join(brief.get("focus_titles", []))))
        sections = [s for s in outline.get("sections", []) if isinstance(s, dict) and s.get("title")]
        if not sections:
            raise ValueError("the outline has no sections")
        await progress("plan", "done", " → ".join(s["title"] for s in sections))
    except (LLMUnavailable, ValueError) as e:
        await progress("plan", "failed", f"{str(e)[:160]} - using the standard plan")
        return None

    planned: list[str] = []
    repo_questions = [q["q"] for r in result["repos"] for q in r.get("deep_questions", [])
                      if isinstance(q, dict) and q.get("q")]
    for number, section in enumerate(sections, 1):
        kind = "first" if number == 1 else "last" if number == len(sections) else "main"
        count = questions_step.get("per_section", 4) if kind == "main" else 1
        angles = list(section.get("angles", []))
        if kind == "main" and repo_questions:
            angles += [f"(idea from their code) {q}" for q in repo_questions[:2]]
            repo_questions = repo_questions[2:]
        step = f"questions:{number}"
        await progress(step, "running", section["title"])
        try:
            written = await _ask(questions_step, questions_step["prompt"].format(
                level=brief["level"], role=brief["role"], company=brief["company"], title=section["title"],
                minutes=section.get("minutes", "?"), goal=section.get("goal", ""), angles="; ".join(map(str, angles)),
                must_haves="; ".join((result["jd"] or {}).get("must_haves", [])[:8]) or "-",
                candidate=(result["resume"] or {}).get("candidate_summary", "-"),
                planned="\n".join(f"- {q}" for q in planned[-20:]) or "none", avoid=avoid,
                seed=random.randint(1, 10**9), n=count, kind=questions_step.get("kinds", {}).get(kind, "")),
                top_p=questions_step.get("top_p", 0.95))
            questions = [q for q in written.get("questions", []) if isinstance(q, dict) and q.get("q")]
            section["questions"] = questions[:count + 1]
            planned += [q["q"] for q in section["questions"]]
            await progress(step, "done", f"{section['title']}: {len(section['questions'])} question(s)")
        except (LLMUnavailable, ValueError) as e:
            section["questions"] = []
            await progress(step, "failed", f"{section['title']}: {str(e)[:120]}")
    if not planned:
        return None
    return {"opening_context": outline.get("opening_context", ""), "sections": sections,
            "red_flags_to_probe": outline.get("red_flags_to_probe", [])}


async def _critique(step: dict, result: dict, avoid: str, progress: Progress) -> None:
    """The reviewer only returns replacements, which keeps its answer small enough for the free tier."""
    await progress("critique", "running", "bar-raiser review of the plan")
    sections = result["plan"]["sections"]
    numbered = "\n".join(
        f"{si}. {section.get('title')}\n"
        + "\n".join(f"   {qi}. {q.get('q')}" for qi, q in enumerate(section.get("questions", []), 1))
        for si, section in enumerate(sections, 1))
    max_changes = step.get("max_changes", 4)
    try:
        review = await _ask(step, step["prompt"].format(
            must_haves=json.dumps((result["jd"] or {}).get("must_haves", [])), avoid=avoid, plan=numbered,
            max_changes=max_changes))
    except (LLMUnavailable, ValueError) as e:
        await progress("critique", "failed", str(e)[:200])
        return
    changed = 0
    for change in review.get("replacements", [])[:max_changes]:
        try:
            questions = sections[int(change["section"]) - 1]["questions"]
            index = int(change["question"]) - 1
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if isinstance(change.get("new"), dict) and change["new"].get("q") and 0 <= index < len(questions):
            questions[index] = change["new"]
            changed += 1
    notes = f" - {review['notes']}" if review.get("notes") else ""
    await progress("critique", "done", f"{changed} question(s) improved{notes}")


def normalise_sections(plan: dict, duration: float) -> list[dict]:
    """Scale the planned minutes so they add up to the interview length."""
    sections = [s for s in plan.get("sections", []) if s.get("title")]
    total = sum(float(s.get("minutes") or 0) for s in sections) or len(sections)
    for section in sections:
        section["minutes"] = round(duration * (float(section.get("minutes") or 0) or 1) / total, 1)
        section.setdefault("questions", [])
        section.setdefault("goal", "")
    return sections
