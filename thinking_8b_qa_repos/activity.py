"""Observable execution summaries, never hidden model reasoning or raw responses."""
from __future__ import annotations

import re
import unicodedata

from .common import clip, dump


def plain(value, budget=1600):
    text = value if isinstance(value, str) else dump(value)
    # Repository text may contain terminal escapes or bidirectional control marks.
    text = "".join(f"\\u{ord(c):04x}" if unicodedata.category(c) in {"Cc", "Cf"} else c for c in text)
    return clip(text, budget)


def markdown(value):
    text = plain(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*{}\[\]()#+.!_|~-])", r"\\\1", text)


def brief(value, budget=240):
    text = plain(" ".join(str(value).split()), budget=4000)
    shortened = clip(text, budget)
    return shortened + ("..." if shortened != text else "")


class Activity:
    def __init__(self, directory, progress, live=False, mode=None):
        self.path = directory / "activity.md"
        self.progress = progress
        self.mode = mode or ("detailed" if live else "quiet")
        if self.mode not in {"concise", "detailed", "quiet"}:
            raise ValueError("progress mode must be concise, detailed or quiet")
        self.shown_findings = 0
        self.last_open_point = None

    def concise(self, stage, message, details):
        """Summarize actual findings and decisions; never invent a model rationale."""
        def say(label, text):
            self.progress(f"[{label}] {brief(text, 320)}")

        def open_point():
            unknowns = details.get("unknowns", [])
            point = unknowns[0] if unknowns else None
            if point and point != self.last_open_point:
                say("Open point", point)
            self.last_open_point = point

        if stage == "Start":
            say("Start", "I will follow the repository evidence to answer: " + brief(message, 200))
        elif stage == "Resume":
            say("Resume", "I am combining the saved findings without reading again." if details.get("synthesize_only")
                else "I am continuing from the saved investigation.")
        elif stage == "Planning":
            say("Plan", "I am identifying the entry points and questions to investigate.")
        elif stage == "Plan":
            queries = details.get("queries", [])
            say("Direction", "I will look for " + ", ".join(brief(q, 60) for q in queries[:3]) + "."
                if queries else "I will start from the original question's search terms.")
            open_point()
        elif stage == "Read":
            self.shown_findings = 0
            paths = details.get("paths", details.get("sources", []))
            label = f"Read {details['step']}/{details['max_steps']}" if "step" in details else "Read"
            say(label, "I am checking " + ", ".join(brief(p, 75) for p in paths[:3])
                + (f" and {len(paths) - 3} more file(s)." if len(paths) > 3 else "."))
        elif stage == "Finding":
            if self.shown_findings < 2:
                say("Insight", re.sub(r"^F\d+:\s*", "", message))
            self.shown_findings += 1
        elif stage == "Checkpoint":
            if self.shown_findings > 2:
                say("Findings", f"I recorded {self.shown_findings - 2} additional findings; full details are in activity.md.")
            elif not self.shown_findings:
                say("Reconsider", "This evidence added no new finding.")
            open_point()
            if details.get("next_query") and details.get("reading_steps_left", 0) > 0:
                say("Next", "I will search for " + brief(details["next_query"], 160) + " to look for additional evidence.")
        elif stage == "Search recovery" and details.get("queries"):
            say("Change approach", "The current search stalled; I will try "
                + ", ".join(brief(q, 60) for q in details["queries"][:3]) + ".")
        elif stage == "Synthesis start":
            reasons = {"step_budget": "The reading budget is exhausted.", "no_new_findings": "Recent reads added no new findings.",
                       "retrieval_exhausted": "The available searches are exhausted.", "scan_complete": "The indexed scan is complete.",
                       "retrieval_page_budget": "The retrieval page budget is exhausted."}
            say("Connect", reasons.get(details.get("stop_reason"), "Reading is paused.")
                + " I am combining all saved findings into an answer.")
        elif stage == "Synthesis":
            say("Combine", "I am combining this batch of notes, keeping conditions, exceptions and unresolved points.")
        elif stage == "Summary":
            say("Combined insight", message)
        elif stage == "Reduction repair":
            say("Adjust", "A summary did not match the required format. I am keeping the original findings for recovery.")
        elif stage == "Reduction fallback":
            say("Simplify", "I will summarize smaller groups separately, then combine them.")
        elif stage == "Answer draft":
            say("Check", message)
        elif stage == "Audit rejected":
            say("Revise", "I cannot support this claim: " + brief(message, 150) + " Reason: " + brief(details.get("reason", ""), 100))
        elif stage == "Complete":
            say("Done", message)
        elif stage == "Stopped":
            say("Stopped", message)

    def record(self, stage, message, details):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self.path.open("a", encoding="utf-8") as stream:
            if new:
                stream.write("# Investigation activity\n\n"
                             "Observable searches, evidence, findings and checks. This is an execution summary, "
                             "not the model's internal chain of thought. Findings and audits can be wrong. "
                             "Interrupted steps are attempts; the checkpoint defines committed progress.\n\n")
            stream.write(f"- **{markdown(stage)}:** {markdown(message)}\n")
            for key, value in details.items():
                stream.write(f"  - {markdown(key)}: {markdown(value)}\n")
        if self.mode == "concise":
            self.concise(stage, message, details)
        elif self.mode == "detailed":
            self.progress(f"[{plain(stage)}] {plain(message)}")
            for key, value in details.items():
                self.progress(f"  {plain(key)}: {plain(value)}")
