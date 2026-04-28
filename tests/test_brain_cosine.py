"""Unit tests for cosine similarity helper."""
import numpy as np
import pytest

from agents.storyAgent.brain.memory.semantic import _cosine_similarity


def test_identical_vectors_score_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6


def test_orthogonal_vectors_score_zero():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_opposite_vectors_score_minus_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(_cosine_similarity(v, -v) - (-1.0)) < 1e-6


def test_zero_vector_returns_zero():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 3.0])
    assert _cosine_similarity(a, b) == 0.0
    assert _cosine_similarity(b, a) == 0.0
