"""
rank_features node — ranks feasible features by predictive value before feature_planner.

Ranking criteria:
  1. Novelty — how much new information vs existing columns
  2. Domain relevance — how directly related to the prediction target
  3. Transformation type — rolling > lag > ratio > simple aggregation
  4. Diversity — penalises redundant features of the same type
"""
import json

from feature_engineer.llm.setup import llm
from feature_engineer.state import AgentState


def rank_features(state: AgentState) -> dict:
    """Rank feasible_features by predictive value for the objective.

    Returns ranked_features — a reordered subset of feasible_features,
    most valuable first, capped at max_features + 2 to keep the
    feature_planner prompt lean.
    """
    feasible  = state.get("feasible_features", [])
    objective = state.get("objective", "")
    max_f     = state.get("max_features", 5)

    if not feasible:
        return {"ranked_features": []}

    # no need to rank if already within budget
    budget = max_f + 2
    if len(feasible) <= budget:
        print(f"[rank_features] {len(feasible)} candidates ≤ budget ({budget}) — no ranking needed.")
        return {"ranked_features": feasible}

    candidates_str = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(feasible))

    prompt = f"""You are a feature engineering expert. Rank these feature candidates by expected predictive value for the given objective.

Objective: {objective}

Feature candidates:
{candidates_str}

Ranking criteria (apply in order):
1. NOVELTY: features that transform existing columns (lag, rolling, ratio, interaction) 
   rank higher than simple aggregations that repeat existing information.
2. DOMAIN RELEVANCE: features directly related to the prediction target rank higher.
3. TRANSFORMATION RICHNESS: rolling > lag > ratio/interaction > groupby mean/sum.
4. DIVERSITY: if many candidates are of the same type (e.g. 8 groupby means), 
   keep only the 2-3 most relevant ones of that type and rank the rest lower.

Return ONLY a JSON array of the feature names in ranked order (best first).
Include ALL candidates but ordered by value.
Return only the JSON array, no other text.

Example format: ["feature_name_1", "feature_name_2", ...]
"""

    response = llm.invoke(prompt).content.strip()
    try:
        clean  = response.replace("```json", "").replace("```", "").strip()
        ranked = json.loads(clean)
        if not isinstance(ranked, list):
            ranked = [f.split(":")[0].strip() for f in feasible]
    except Exception:
        ranked = [f.split(":")[0].strip() for f in feasible]

    # match ranked names back to full feasible strings
    feasible_map = {}
    for f in feasible:
        name = f.split(":")[0].strip().lower().replace(" ", "_")
        feasible_map[name] = f
        # also index by full string
        feasible_map[f.split(" (")[0].strip().lower().replace(" ", "_")] = f

    ranked_full = []
    seen        = set()
    for name in ranked:
        key = name.lower().replace(" ", "_")
        match = feasible_map.get(key)
        if not match:
            # fuzzy: find feasible that starts with this name
            match = next((f for f in feasible if f.lower().startswith(key[:15])), None)
        if match and match not in seen:
            ranked_full.append(match)
            seen.add(match)

    # append any feasible not matched (preserve them at end)
    for f in feasible:
        if f not in seen:
            ranked_full.append(f)

    # cap at budget
    result = ranked_full[:budget]

    print(f"[rank_features] Ranked {len(feasible)} → top {len(result)} features:")
    for i, f in enumerate(result):
        name = f.split(":")[0].strip()
        print(f"  {i+1}. {name}")

    return {"ranked_features": result}
