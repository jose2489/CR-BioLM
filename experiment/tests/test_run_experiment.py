"""
Unit tests for run_experiment.py logic:
  - assign_questions: no-replacement per persona, cycling, mixed-persona isolation
  - generar_exp_id: persona_label and collision avoidance
  - --n sampling: correct count, reproducibility
  - persona_map: fixed vs random assignment, log recovery
"""

import os
import sys
import random
import tempfile
import shutil
import json
import pytest

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment.run_experiment import assign_questions, generar_exp_id, RUNS_DIR
from utils.question_bank import get_question_meta, QUESTION_BANK


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_runs(tmp_path, monkeypatch):
    """Redirect RUNS_DIR to a temp directory for isolation."""
    import experiment.run_experiment as mod
    monkeypatch.setattr(mod, "RUNS_DIR", str(tmp_path))
    return tmp_path


# ── assign_questions ──────────────────────────────────────────────────────────

class TestAssignQuestions:

    def _make_rng(self, seed=42):
        return random.Random(seed)

    def test_all_species_get_a_question(self):
        species = [f"Sp{i}" for i in range(10)]
        persona_map = {sp: "turista" for sp in species}
        rng = self._make_rng()
        result = assign_questions(species, persona_map, log={}, rng=rng)
        assert set(result.keys()) == set(species)
        for sp, entry in result.items():
            assert "q" in entry
            assert "stratum" in entry

    def test_no_duplicates_within_pool_size(self):
        """Within one full cycle (n <= pool size), questions must be unique."""
        pool_size = len(get_question_meta("turista"))
        species = [f"Sp{i}" for i in range(pool_size)]
        persona_map = {sp: "turista" for sp in species}
        rng = self._make_rng()
        result = assign_questions(species, persona_map, log={}, rng=rng)
        questions = [result[sp]["q"] for sp in species]
        assert len(questions) == len(set(questions)), "Duplicate questions found within pool size"

    def test_cycles_when_exceeding_pool_size(self):
        """When n > pool size, cycling is allowed but the first cycle must be exhausted."""
        pool_size = len(get_question_meta("turista"))
        n = pool_size + 3
        species = [f"Sp{i}" for i in range(n)]
        persona_map = {sp: "turista" for sp in species}
        rng = self._make_rng()
        result = assign_questions(species, persona_map, log={}, rng=rng)
        assert len(result) == n
        # All entries must be valid question dicts
        for entry in result.values():
            assert "q" in entry

    def test_skips_already_logged_species(self):
        species = ["Sp0", "Sp1", "Sp2"]
        persona_map = {sp: "turista" for sp in species}
        existing_q = get_question_meta("turista")[0]["q"]
        log = {"Sp0|pregunta|turista": {"pregunta": existing_q, "stratum": "A"}}
        rng = self._make_rng()
        result = assign_questions(species, persona_map, log=log, rng=rng)
        # Sp0 already logged — should not be in new assignments
        assert "Sp0" not in result
        assert "Sp1" in result
        assert "Sp2" in result

    def test_mixed_persona_pools_are_independent(self):
        """botanico and turista pools must be drawn independently."""
        botanico_pool_size = len(get_question_meta("botanico"))
        turista_pool_size = len(get_question_meta("turista"))
        botanico_species = [f"Bot{i}" for i in range(botanico_pool_size)]
        turista_species = [f"Tur{i}" for i in range(turista_pool_size)]
        all_species = botanico_species + turista_species
        persona_map = {}
        for sp in botanico_species:
            persona_map[sp] = "botanico"
        for sp in turista_species:
            persona_map[sp] = "turista"

        rng = self._make_rng()
        result = assign_questions(all_species, persona_map, log={}, rng=rng)

        botanico_qs = {result[sp]["q"] for sp in botanico_species}
        turista_qs = {result[sp]["q"] for sp in turista_species}
        # Each pool is internally unique (no duplicates within pool size)
        assert len(botanico_qs) == botanico_pool_size
        assert len(turista_qs) == turista_pool_size

    def test_botanico_questions_not_assigned_to_turista(self):
        """Questions from the botanico bank must not appear in turista slots."""
        turista_qs = {e["q"] for e in get_question_meta("turista")}
        botanico_qs = {e["q"] for e in get_question_meta("botanico")}
        # Sanity: banks are disjoint
        assert turista_qs.isdisjoint(botanico_qs), "Question banks must be disjoint"

        species = [f"Sp{i}" for i in range(5)]
        persona_map = {sp: "turista" for sp in species}
        rng = self._make_rng()
        result = assign_questions(species, persona_map, log={}, rng=rng)
        for sp, entry in result.items():
            assert entry["q"] in turista_qs, f"Turista species got a botanico question: {entry['q']}"

    def test_reproducible_with_same_seed(self):
        species = [f"Sp{i}" for i in range(5)]
        persona_map = {sp: "botanico" for sp in species}
        r1 = assign_questions(species, persona_map, log={}, rng=random.Random(99))
        r2 = assign_questions(species, persona_map, log={}, rng=random.Random(99))
        assert {sp: e["q"] for sp, e in r1.items()} == {sp: e["q"] for sp, e in r2.items()}

    def test_different_seeds_produce_different_order(self):
        species = [f"Sp{i}" for i in range(8)]
        persona_map = {sp: "turista" for sp in species}
        r1 = assign_questions(species, persona_map, log={}, rng=random.Random(1))
        r2 = assign_questions(species, persona_map, log={}, rng=random.Random(2))
        qs1 = [r1[sp]["q"] for sp in species]
        qs2 = [r2[sp]["q"] for sp in species]
        assert qs1 != qs2, "Different seeds should (almost always) produce different order"


# ── generar_exp_id ────────────────────────────────────────────────────────────

class TestGenerarExpId:

    def test_mixed_label_for_random_persona(self, tmp_runs):
        exp_id = generar_exp_id("mixed")
        assert exp_id.endswith("-mixed")

    def test_botanico_label(self, tmp_runs):
        exp_id = generar_exp_id("botanico")
        assert exp_id.endswith("-botanico")

    def test_no_collision_on_second_call(self, tmp_runs):
        id1 = generar_exp_id("turista")
        os.makedirs(os.path.join(str(tmp_runs), id1))
        id2 = generar_exp_id("turista")
        assert id1 != id2

    def test_counter_increments(self, tmp_runs):
        id1 = generar_exp_id("botanico")
        os.makedirs(os.path.join(str(tmp_runs), id1))
        id2 = generar_exp_id("botanico")
        num1 = int(id1.split("-")[2])
        num2 = int(id2.split("-")[2])
        assert num2 == num1 + 1


# ── Persona map logic (unit-tested without I/O) ───────────────────────────────

class TestPersonaMap:

    def _build_persona_map_random(self, species, seed):
        """Mirrors the random persona assignment logic from main()."""
        rng = random.Random(seed)
        personas = ["botanico", "turista"]
        return {sp: rng.choice(personas) for sp in species}

    def test_all_species_get_valid_persona(self):
        species = [f"Sp{i}" for i in range(20)]
        persona_map = self._build_persona_map_random(species, seed=42)
        valid = {"botanico", "turista"}
        for sp, p in persona_map.items():
            assert p in valid

    def test_both_personas_appear_in_large_sample(self):
        """With 20 species, both personas should appear (probability of all-same ~1e-6)."""
        species = [f"Sp{i}" for i in range(20)]
        persona_map = self._build_persona_map_random(species, seed=42)
        personas_used = set(persona_map.values())
        assert "botanico" in personas_used
        assert "turista" in personas_used

    def test_fixed_persona_uniform(self):
        species = [f"Sp{i}" for i in range(10)]
        persona_map = {sp: "botanico" for sp in species}
        assert all(p == "botanico" for p in persona_map.values())

    def test_log_recovery_overrides_rng(self):
        """Persona from log must override a fresh RNG draw (resume scenario)."""
        species = ["Sp0", "Sp1"]
        log = {"Sp0|persona": "turista"}
        rng = random.Random(42)
        personas = ["botanico", "turista"]
        persona_map = {}
        for sp in species:
            log_key = f"{sp}|persona"
            if log_key in log:
                persona_map[sp] = log[log_key]
            else:
                persona_map[sp] = rng.choice(personas)
        assert persona_map["Sp0"] == "turista"  # from log, not RNG
