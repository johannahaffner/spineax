import lineax as lx
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import equinox as eqx
from typing import Iterable


# Stolen from Lineax itself
def _frozenset(x: object | Iterable[object]) -> frozenset[object]:
    try:
        iter_x = iter(x)  # pyright: ignore
    except TypeError:
        return frozenset([x])
    else:
        return frozenset(iter_x)


class COOLinearOperator(lx.AbstractLinearOperator):
    """Sparse linear operator in COO format, which stores a sparse representation of
    a linear operator with values and their associated row and column indices.
    Slightly less memory efficient than the compressed sparse row and compressed
    sparse column formats wrapped in [`lineax.CSLinearOperator`][].

    Note that the `as_matrix` method will materialise the operator densely.
    """

    operator: jsp.COO
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(self, operator: jsp.COO, tags: object | Iterable[object] = ()):
        self.operator = operator
        self.tags = _frozenset(tags)

    def mv(self, vector):
        return self.operator @ vector

    def as_matrix(self):
        return self.operator.todense()

    def transpose(self):
        return COOLinearOperator(self.operator.transpose(), lx.transpose_tags(self.tags))

    def in_structure(self):
        _, in_size = self.operator.shape
        return jax.ShapeDtypeStruct(shape=(in_size,), dtype=self.operator.data.dtype)

    def out_structure(self):
        out_size, _ = self.operator.shape
        return jax.ShapeDtypeStruct(shape=(out_size,), dtype=self.operator.data.dtype)


@lx.conj.register(COOLinearOperator)
def _(operator):
    conj_data = jnp.conj(operator.operator.data)
    conj_coo = jsp.COO(
        (conj_data, operator.operator.row, operator.operator.col),
        shape=operator.operator.shape,
    )
    return COOLinearOperator(conj_coo)

@lx.diagonal.register(COOLinearOperator)
def _(operator):
    # TODO(jhaffner): Inefficient materialised method
    return jnp.diag(operator.as_matrix())

@lx.has_unit_diagonal.register(COOLinearOperator)
def _(operator):
    return lx.unit_diagonal_tag in operator.tags

@lx.is_diagonal.register(COOLinearOperator)
def _(operator):
    return lx.diagonal_tag in operator.tags or (
        operator.in_size() == 1 and operator.out_size() == 1
    )

@lx.is_lower_triangular.register(COOLinearOperator)
def _(operator):
    return lx.lower_triangular_tag in operator.tags

@lx.is_negative_semidefinite.register(COOLinearOperator)
def _(operator):
    return lx.negative_semidefinite_tag in operator.tags

@lx.is_positive_semidefinite.register(COOLinearOperator)
def _(operator):
    return lx.positive_semidefinite_tag in operator.tags

@lx.is_symmetric.register(COOLinearOperator)
def _(operator):
    return any(
        tag in operator.tags
        for tag in (
            lx.symmetric_tag,
            lx.positive_semidefinite_tag,
            lx.negative_semidefinite_tag,
            lx.diagonal_tag,
        )
    )

@lx.is_tridiagonal.register(COOLinearOperator)
def _(operator):
    return lx.tridiagonal_tag in operator.tags or lx.diagonal_tag in operator.tags

@lx.is_upper_triangular.register(COOLinearOperator)
def _(operator):
    return lx.upper_triangular_tag in operator.tags

@lx.linearise.register(COOLinearOperator)
def _(operator):
    return operator

@lx.materialise.register(COOLinearOperator)
def _(operator):
    return operator

@lx.tridiagonal.register(COOLinearOperator)
def _(operator):
    matrix = operator.as_matrix()
    assert matrix.ndim == 2
    diagonal = jnp.diagonal(matrix, offset=0)
    upper_diagonal = jnp.diagonal(matrix, offset=1)
    lower_diagonal = jnp.diagonal(matrix, offset=-1)
    return diagonal, lower_diagonal, upper_diagonal