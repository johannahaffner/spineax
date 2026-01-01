# Run the lineax tests, but for Spineax operators
import spineax as spx
from .helpers import make_coo_operator
from lineax.tests.helpers import solvers_tags_pseudoinverse
from lineax.tests.test_jvp import test_jvp as lx_test_jvp


@pytest.mark.parametrize("make_operator", (make_coo_operator,))
@pytest.mark.parametrize("solver, tags, pseudoinverse", solvers_tags_pseudoinverse)
@pytest.mark.parametrize("use_state", (True, False))
@pytest.mark.parametrize(
    "make_matrix",
    (
        construct_matrix,
        construct_singular_matrix,
    ),
)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_jvp(
    getkey, solver, tags, pseudoinverse, make_operator, use_state, make_matrix, dtype
):
    return lx_test_jvp(getkey, solver, tags, pseudoinverse, make_operator, use_state, make_matrix, dtype)
