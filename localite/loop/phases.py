"""Phase definitions and transition rules for the 5-phase agent loop."""

from enum import Enum


class Phase(str, Enum):
    """Phase enum for the agent loop."""
    EXPLORE = "EXPLORE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    ITERATE = "ITERATE"
    COMPLETE = "COMPLETE"


# Phase transition rules:
# EXPLORE -> PLAN (model has gathered enough info)
# PLAN -> EXECUTE (after user approves plan via permission gate)
# EXECUTE -> VERIFY (after tool calls complete)
# VERIFY -> ITERATE (if tests fail, max 3 iterations)
# VERIFY -> COMPLETE (if tests pass or user satisfied)
# ITERATE -> EXECUTE (continue fixing)
# ITERATE -> COMPLETE (if max iterations reached)

VALID_TRANSITIONS: dict[Phase, list[Phase]] = {
    Phase.EXPLORE: [Phase.PLAN],
    Phase.PLAN: [Phase.EXECUTE, Phase.COMPLETE],
    Phase.EXECUTE: [Phase.VERIFY],
    Phase.VERIFY: [Phase.ITERATE, Phase.COMPLETE],
    Phase.ITERATE: [Phase.EXECUTE, Phase.COMPLETE],
    Phase.COMPLETE: [],
}


def is_valid_transition(from_phase: Phase, to_phase: Phase) -> bool:
    """Check if a phase transition is valid."""
    return to_phase in VALID_TRANSITIONS.get(from_phase, [])


def next_phase(from_phase: Phase, tests_passed: bool | None = None,
               max_iterations_reached: bool = False) -> Phase:
    """Determine the next phase based on current phase and conditions.

    Args:
        from_phase: The current phase.
        tests_passed: Whether tests passed (only relevant for VERIFY phase).
        max_iterations_reached: Whether max ITERATE cycles have been reached.

    Returns:
        The next Phase.
    """
    if from_phase == Phase.EXPLORE:
        return Phase.PLAN
    elif from_phase == Phase.PLAN:
        return Phase.EXECUTE
    elif from_phase == Phase.EXECUTE:
        return Phase.VERIFY
    elif from_phase == Phase.VERIFY:
        if tests_passed:
            return Phase.COMPLETE
        elif max_iterations_reached:
            return Phase.COMPLETE
        else:
            return Phase.ITERATE
    elif from_phase == Phase.ITERATE:
        if max_iterations_reached:
            return Phase.COMPLETE
        return Phase.EXECUTE
    return Phase.COMPLETE