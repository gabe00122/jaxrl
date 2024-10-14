import orbax.checkpoint as ocp
from flax import nnx
from pathlib import Path
import jax


class Checkpointer:
    def __init__(self, directory: str | Path):
        directory = Path(directory)
        directory = directory.absolute()
        self.mngr = ocp.CheckpointManager(directory)

    def save(self, model, global_step: int):
        state = nnx.state(model)
        self.mngr.save(global_step, args=ocp.args.StandardSave(state))

    def restore[T](self, model: T, step: int) -> T:
        graphdef, state = nnx.split(model)
        abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
        restored_state = self.mngr.restore(
            step, args=ocp.args.StandardRestore(abstract_state)
        )
        return nnx.merge(graphdef, restored_state)

    def restore_latest[T](self, model: T) -> T:
        return self.restore(model, self.mngr.latest_step())

    def close(self):
        self.mngr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
