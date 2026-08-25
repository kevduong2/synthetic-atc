import numpy as np
import pytest

from atcgen.rl.policy import CrossEntropyMethod, RandomSearch, ReinforceGaussian


DIM = 6
TARGET = np.full(DIM, 0.7)
ITERATIONS = 10
POP = 6            # 60 evaluations, the realistic order for a ~10 min reward


def toy_reward(vector, rng):
    """Smooth unimodal reward with a little observation noise.

    Stands in for the real thing: an optimum inside the cube, and a reward you
    only ever see through the noise of a short fine-tune.
    """
    return -float(np.sum((np.asarray(vector) - TARGET) ** 2)) + 0.01 * rng.standard_normal()


def run(optimizer, seed=7, iterations=ITERATIONS, pop=POP):
    rng = np.random.default_rng(seed)
    best = -np.inf
    for _ in range(iterations):
        vectors = optimizer.ask(pop)
        rewards = [toy_reward(vector, rng) for vector in vectors]
        optimizer.tell(vectors, rewards)
        best = max(best, max(rewards))
    return best


def test_ask_stays_inside_the_unit_cube():
    for optimizer in (RandomSearch(DIM, seed=0),
                      ReinforceGaussian(DIM, seed=0, init_sigma=0.35),
                      CrossEntropyMethod(DIM, seed=0, init_sigma=0.9)):
        vectors = optimizer.ask(20)
        assert len(vectors) == 20
        for vector in vectors:
            assert vector.shape == (DIM,)
            assert np.all((vector >= 0.0) & (vector <= 1.0))


@pytest.mark.parametrize("factory", [
    lambda: CrossEntropyMethod(DIM, seed=1),
    lambda: ReinforceGaussian(DIM, seed=1),
])
def test_learners_beat_random_search_on_equal_budget(factory):
    baseline = run(RandomSearch(DIM, seed=1))
    learned = run(factory())
    assert learned > baseline


def test_cem_concentrates_on_the_optimum():
    optimizer = CrossEntropyMethod(DIM, seed=3)
    run(optimizer)
    assert np.allclose(optimizer.mean, TARGET, atol=0.12)
    assert np.all(optimizer.sigma >= optimizer.sigma_decay_floor)


def test_reinforce_moves_the_mean_toward_the_optimum():
    optimizer = ReinforceGaussian(DIM, seed=3)
    start = np.linalg.norm(optimizer.mean - TARGET)
    run(optimizer, iterations=20)
    assert np.linalg.norm(optimizer.mean - TARGET) < start
    assert np.all(optimizer.sigma >= optimizer.sigma_min)
    assert np.all(optimizer.sigma <= optimizer.sigma_max)


def test_reinforce_scores_the_pre_clip_sample():
    """A mean pinned at the wall must still be pushed inward, not held there."""
    optimizer = ReinforceGaussian(DIM, seed=5, init_mean=np.zeros(DIM), init_sigma=0.3)
    vectors = optimizer.ask(8)
    # Reward the samples that came from furthest *below* zero the least.
    rewards = [float(np.sum(vector)) for vector in vectors]
    optimizer.tell(vectors, rewards)
    assert np.all(optimizer.mean >= 0.0)
    assert np.any(optimizer.mean > 0.0)


def test_tell_tolerates_degenerate_batches():
    for optimizer in (RandomSearch(DIM, seed=0), ReinforceGaussian(DIM, seed=0),
                      CrossEntropyMethod(DIM, seed=0)):
        optimizer.ask(3)
        optimizer.tell([np.full(DIM, 0.4)], [1.0])              # single sample
        optimizer.ask(3)
        optimizer.tell([np.full(DIM, 0.4), np.full(DIM, 0.6)], [2.0, 2.0])  # flat
        optimizer.tell([], [])
        assert np.all(np.isfinite(optimizer.ask(1)[0]))

    with pytest.raises(ValueError):
        CrossEntropyMethod(DIM, seed=0).tell([np.zeros(DIM)], [1.0, 2.0])
    with pytest.raises(ValueError):
        ReinforceGaussian(DIM, seed=0).tell([np.zeros(DIM)], [1.0, 2.0])


@pytest.mark.parametrize("factory", [
    lambda: RandomSearch(DIM, seed=11),
    lambda: ReinforceGaussian(DIM, seed=11),
    lambda: CrossEntropyMethod(DIM, seed=11),
])
def test_state_dict_round_trip_resumes_the_same_ask_sequence(factory):
    import json

    original = factory()
    run(original, iterations=3)
    state = json.loads(json.dumps(original.state_dict()))   # must be JSON-clean

    restored = factory()
    restored.load_state_dict(state)

    for _ in range(3):
        for left, right in zip(original.ask(POP), restored.ask(POP)):
            assert np.array_equal(left, right)


def test_reinforce_resumes_mid_batch_without_losing_the_pre_clip_samples():
    import json

    optimizer = ReinforceGaussian(DIM, seed=2, init_mean=np.zeros(DIM))
    vectors = optimizer.ask(4)
    restored = ReinforceGaussian(DIM, seed=2)
    restored.load_state_dict(json.loads(json.dumps(optimizer.state_dict())))

    rewards = [float(i) for i in range(4)]
    optimizer.tell(vectors, rewards)
    restored.tell(vectors, rewards)
    assert np.allclose(optimizer.mean, restored.mean)
    assert np.allclose(optimizer.sigma, restored.sigma)
