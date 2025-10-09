import orbax.checkpoint as ocp
from flax import nnx
from pathlib import Path

import jax
from jax.sharding import Mesh


class Checkpointer:
    def __init__(self, directory: str):
        if not directory.startswith("gs://"):
            directory = Path(directory).absolute().as_posix()
        self.mngr = ocp.CheckpointManager(directory)

    def save(self, model: object, global_step: int):
        state = nnx.state(model, nnx.Param)
        self.mngr.save(global_step, args=ocp.args.StandardSave(state))

    def restore[T](self, model: T, step: int) -> T:
        device = jax.devices()[0]
        mesh = Mesh((device,), ("batch",))

        graphdef, abstract_state = nnx.get_abstract_model(lambda: model, mesh)
        restored_state = self.mngr.restore(
            step, args=ocp.args.StandardRestore(abstract_state)
        )

        return nnx.merge(graphdef, restored_state)

    def restore_latest[T](self, model: T) -> T:
        return self.restore(model, self.mngr.latest_step() or 0)

    def close(self):
        self.mngr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
