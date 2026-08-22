"""Deterministic team-draft interleaving for sensitive ranking comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from .diagnostics import ExperimentDiagnosticError


@dataclass(frozen=True)
class TeamDraftResult:
    ranking: tuple[str, ...]
    ownership: tuple[str, ...]

    def credit(self, clicked_item_ids: set[str]) -> dict[str, int]:
        result = {"control": 0, "treatment": 0}
        for item_id, owner in zip(self.ranking, self.ownership, strict=True):
            if item_id in clicked_item_ids:
                result[owner] += 1
        return result


def team_draft_interleave(
    control_ranking: list[str], treatment_ranking: list[str], *, k: int
) -> TeamDraftResult:
    """Interleave unique results, alternating the first-pick team by rank."""

    if k <= 0:
        raise ExperimentDiagnosticError("k must be positive.")
    if not control_ranking or not treatment_ranking:
        raise ExperimentDiagnosticError("Both rankings must contain at least one item.")
    if len(set(control_ranking)) != len(control_ranking) or len(set(treatment_ranking)) != len(
        treatment_ranking
    ):
        raise ExperimentDiagnosticError("Input rankings must not contain duplicates.")
    rankings = {"control": control_ranking, "treatment": treatment_ranking}
    cursors = {"control": 0, "treatment": 0}
    selected: list[str] = []
    owners: list[str] = []
    while len(selected) < k:
        progress = False
        first = "control" if len(selected) % 2 == 0 else "treatment"
        for team in (first, "treatment" if first == "control" else "control"):
            ranking = rankings[team]
            while cursors[team] < len(ranking) and ranking[cursors[team]] in selected:
                cursors[team] += 1
            if cursors[team] < len(ranking):
                selected.append(ranking[cursors[team]])
                owners.append(team)
                cursors[team] += 1
                progress = True
                break
        if not progress:
            break
    return TeamDraftResult(tuple(selected), tuple(owners))
