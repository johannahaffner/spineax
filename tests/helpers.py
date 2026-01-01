import lineax as lx
import jax.experimental.sparse as jsp

def make_coo_operator(getkey, matrix, tags):
    # The experimental module is not recognised as a known module of JAX by pyright
    return lx.COOLinearOperator(jsp.COO.fromdense(matrix), tags)  # pyright: ignore[reportGeneralTypeIssues]
