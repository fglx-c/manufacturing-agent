"""Flowcept ↔ CrewAI bridge layer.

This module exposes two helper functions – ``crew_generate_options_set`` and
``crew_choose_option`` – that mimic the public contract expected by Flowcept
while delegating the decision-making to the CrewAI crew implemented in
``manufacturing_agent.crew``.

The goal is to keep Flowcept completely unaware of CrewAI internals and to
avoid any real file system I/O.  All transient files referenced in the YAML
templates are served from an in-memory *virtual file system* provided via
``manufacturing_agent.tools.file_tools.MEM_STORE``.
"""

from __future__ import annotations

import json
from threading import Lock
from typing import Dict, List, Any

from langchain_community.chat_models import ChatLiteLLM
from flowcept.configs import AGENT
from manufacturing_agent.crew import ManufacturingAgentCrew

# Virtual FS store shared with CrewAI tools
from manufacturing_agent.tools.file_tools import MEM_STORE

# ---------------------------------------------------------------------------
# Internal per-campaign session management
# ---------------------------------------------------------------------------


class _CrewSession:
    """Encapsulate a CrewAI instance and per-campaign cached data."""

    def __init__(self, campaign_id: str | None):
        self.campaign_id: str = campaign_id or "default"

        # --- LLM Configuration for CrewAI ---
        # Using OpenAI as the LLM provider for the crew.
        # This requires two things:
        #   1. `pip install langchain-openai` in your conda environment.
        #   2. `OPENAI_API_KEY` environment variable set in the server's terminal.
        # You can also set `OPENAI_MODEL_NAME` to specify a model.
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o")

        crew_definition = ManufacturingAgentCrew(llm=llm)
        self._crew = crew_definition.crew()  # build once
        self._lock = Lock()  # guarantee thread-safety for the shared crew

    # ---------------------------------------------------------------------
    # Public helpers used by the wrappers below
    # ---------------------------------------------------------------------
    def decide_option(self, layer: int, scores: Dict[str, Any]):
        """Run CrewAI tasks once and parse the result.

        The implementation assumes that the *output_task* of the crew writes
        its JSON decision file into the MEM_STORE using the *WriteFileTool*.
        We therefore inspect MEM_STORE after the crew finishes.
        """

        # Inject runtime variables expected by the YAML descriptions
        crew_inputs = {
            "layer_number": layer,
            "scores": scores,
        }

        # Run the full crew (sequential process).  The return value of
        # Crew.kickoff() may differ by CrewAI version; we capture it mainly
        # for logging/debugging.
        with self._lock:
            raw_result = self._crew.kickoff(inputs=crew_inputs)

        # -----------------------------------------------------------------
        # Parse the crew output.
        # -----------------------------------------------------------------
        # The reference implementation of *output_task* writes a JSON param
        # file and a reasoning text file.  Look them up in MEM_STORE.
        param_key = f"/scratch/ttc/manufacturing-agent/manufacturing_agent/src/output/{layer}_param.json"
        reasoning_key = f"/scratch/ttc/manufacturing-agent/manufacturing_agent/src/output/{layer}_reasoning.txt"

        best_index: int | None = None
        explanation: str = ""

        try:
            if param_key in MEM_STORE:
                param_data = json.loads(MEM_STORE[param_key])
                # Expecting {"best_option": <int>, ...}
                best_index = int(param_data.get("best_option", 0))
            if reasoning_key in MEM_STORE:
                explanation = MEM_STORE[reasoning_key]
        except Exception as exc:  # noqa: BLE001
            explanation = f"Could not parse crew output: {exc}. Raw: {raw_result}"

        # Reasonable fallback if parsing failed
        if best_index is None:
            # Choose the minimal score index as heuristic fallback
            best_index = int(min(range(len(scores["scores"])), key=scores["scores"].__getitem__))

        return {
            "best_index": best_index,
            "reasoning": explanation or "Decision produced by CrewAI.",
            "raw": raw_result,
        }


# ---------------------------------------------------------------------------
# Session registry and helper accessors
# ---------------------------------------------------------------------------


_SESSIONS: Dict[str, _CrewSession] = {}


def _get_session(campaign_id: str | None) -> _CrewSession:
    key = campaign_id or "default"
    if key not in _SESSIONS:
        _SESSIONS[key] = _CrewSession(key)
    return _SESSIONS[key]


# ---------------------------------------------------------------------------
# External API – **these names are imported by aec_agent_mock.py**
# ---------------------------------------------------------------------------


def crew_generate_options_set(
    layer: int,
    planned_controls: List[Dict[str, Any]],
    number_of_options: int = 4,
    campaign_id: str | None = None,
):
    """Cache *planned_controls* so that the subsequent CrewAI run can read them.

    Flowcept expects to receive the same structure it currently gets from the
    adhoc LLM implementation, namely a dictionary containing
    ``control_options`` and some logging metadata.  We therefore simply echo
    what we were given.
    """

    # The crew's analysis_task expects to read this file.
    path = f"/scratch/multi-user/manufacturing-agent/manufacturing_agent/src/data/control_options_L{layer}.txt"
    MEM_STORE[path] = json.dumps(planned_controls)

    return {
        "control_options": planned_controls,
        "response": json.dumps(planned_controls),
        "prompt": [],
        "llm": "CrewAI",
    }


def crew_choose_option(
    scores: Dict[str, Any],
    planned_controls: List[Dict[str, Any]],
    campaign_id: str | None = None,
):
    """Invoke CrewAI to decide which control option to apply."""

    layer = scores.get("layer", 0)
    sess = _get_session(campaign_id)

    decision = sess.decide_option(layer, scores)

    result_payload = {
        "option": decision["best_index"],
        "explanation": decision["reasoning"],
        "label": "CrewAI",
        "attention": decision["best_index"] != scores.get("human_option"),
    }

    return {
        **result_payload,
        "response": json.dumps(result_payload),
        "prompt": [],
        "llm": "CrewAI",
    } 