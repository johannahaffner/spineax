import lineax as lx
import jax
import jax.experimental.sparse as jsp
import equinox as eqx


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
        return COOLinearOperator(self.operator.transpose(), transpose_tags(self.tags))

    def in_structure(self):
        _, in_size = self.operator.shape
        return jax.ShapeDtypeStruct(shape=(in_size,), dtype=self.operator.data.dtype)

    def out_structure(self):
        out_size, _ = self.operator.shape
        return jax.ShapeDtypeStruct(shape=(out_size,), dtype=self.operator.data.dtype)


