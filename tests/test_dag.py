import pytest

from taskforge.dag import PlanError, Task, TaskGraph


def plan(*tasks):
    return {"tasks": [dict(t) for t in tasks]}


def t(id, deps=(), **kw):
    return {"id": id, "description": f"do {id}", "deps": list(deps), **kw}


def test_valid_diamond_levels_and_topo_order():
    g = TaskGraph.from_dict(plan(t("a"), t("b", ["a"]), t("c", ["a"]), t("d", ["b", "c"])))
    assert g.levels() == [["a"], ["b", "c"], ["d"]]
    assert g.topological_order() == ["a", "b", "c", "d"]
    assert g.descendants("a") == {"b", "c", "d"}
    assert g.ready(done=set(), started=set()) == ["a"]
    assert g.ready(done={"a"}, started={"a"}) == ["b", "c"]
    assert g.ready(done={"a", "b"}, started={"a", "b", "c"}) == []


def test_cycle_is_reported_with_its_path():
    with pytest.raises(PlanError, match=r"dependency cycle: (a -> c -> b -> a|b -> a -> c -> b|c -> b -> a -> c)"):
        TaskGraph.from_dict(plan(t("a", ["c"]), t("b", ["a"]), t("c", ["b"])))


@pytest.mark.parametrize("bad, message", [
    (plan(t("a", ["missing"])), "depends on unknown task 'missing'"),
    (plan(t("a", ["a"])), "depends on itself"),
    (plan(t("a"), t("a")), "duplicate task id"),
    (plan(t("a", role="wizard")), "unknown role 'wizard'"),
    (plan(t("a", checks=[{"type": "vibes"}])), "unknown check"),
    (plan(t("has space")), "must match"),
    ({"tasks": []}, "no tasks"),
    ({"steps": []}, "must be an object"),
    (plan({"id": "a", "description": ""}), "description must be a non-empty string"),
    (plan({"id": "a", "description": "x", "deps": "b"}), "deps must be a list of strings"),
])
def test_invalid_plans_are_rejected(bad, message):
    with pytest.raises(PlanError, match=message):
        TaskGraph.from_dict(bad, known_roles={"generalist"})


def test_long_chain_does_not_hit_recursion_limit():
    tasks = [t("n0")] + [t(f"n{i}", [f"n{i - 1}"]) for i in range(1, 5000)]
    g = TaskGraph.from_dict(plan(*tasks))
    assert len(g.levels()) == 5000


def test_round_trip_through_dict():
    g = TaskGraph.from_dict(plan(t("a", tools=["read_file"], acceptance=["ok"],
                                     checks=[{"type": "contains", "value": "x"}]), t("b", ["a"])))
    again = TaskGraph.from_dict(g.to_dict())
    assert again.tasks == g.tasks
    assert isinstance(again["a"], Task) and again["a"].tools == ("read_file",)
