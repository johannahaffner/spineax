# Run the lineax tests, but for Spineax operators
import spineax as spx
import sys
import jax.numpy as jnp
from .helpers import make_coo_operator

from .lineax_test_utils import lx_tests

import pytest


@pytest.mark.parametrize("make_operator", (make_coo_operator,))
@pytest.mark.parametrize(
    "solver, tags, pseudoinverse", lx_tests.solvers_tags_pseudoinverse
)
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize(
    "make_matrix",
    (
        lx_tests.construct_matrix,
        lx_tests.construct_singular_matrix,
    ),
)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_jvp(
    getkey, solver, tags, pseudoinverse, make_operator, use_state, make_matrix, dtype
):
    return lx_tests.test_jvp(
        getkey,
        solver,
        tags,
        pseudoinverse,
        make_operator,
        use_state,
        make_matrix,
        dtype,
    )
