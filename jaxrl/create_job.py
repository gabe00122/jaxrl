from typing import NamedTuple
import typer
import os
import stat

from jaxrl.experiment import get_git_hash

app = typer.Typer()


class TPUSettings(NamedTuple):
    zone: str
    accelerator_type: str
    version: str


@app.command()
def queue_job(
    tpu: str,
    config_file: str,
    node: str = "node-1",
    queue: bool = False,
    preemptible: bool = False,
):
    wandb_key = os.environ["WANDB_API_KEY"]
    git_hash = get_git_hash()

    with open(config_file, "r") as f:
        config_text = f.read()

    node_name = node
    service_account = "tpu-account@gen-lang-client-0325319159.iam.gserviceaccount.com"

    match tpu:
        case "v2-8":
            zone = "us-central1-f"
            accelerator_type = "v2-8"
            version = "tpu-ubuntu2204-base"
            preemptible = True
        case "v3-8":
            zone = "europe-west4-a"
            accelerator_type = "v3-8"
            version = "tpu-ubuntu2204-base"
            preemptible = True
        case "v4-8":
            zone = "us-central2-b"
            accelerator_type = "v4-8"
            version = "tpu-ubuntu2204-base"
        case "v5e-1":
            zone = "europe-west4-a"
            accelerator_type = "v5e-1"
            version = "v2-alpha-tpuv5-lite"
            preemptible = True
        case "v6e-1":
            zone = "us-east1-d"
            accelerator_type = "v6e-1"
            version = "v2-alpha-tpuv6e"
            preemptible = True
        case _:
            raise ValueError(f"Invalid TPU type: {tpu}")

    startup_text = f"""
#!/bin/bash

echo "Setting up environment..."

export WANDB_API_KEY="{wandb_key}"

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/gabe00122/jaxrl.git
cd jaxrl
git switch {git_hash} --detach
uv sync --extra tpu

cat << EOF >> config/config.json
{config_text}
EOF

echo "Starting JAX training job... 🚀"
uv run pmarl train --config ./config/config.json --distributed --base-dir "gs://training_results_gabe00122/results"

echo "JAX training complete. Initiating TPU shutdown."

gcloud compute tpus tpu-vm delete {node_name} --zone={zone} --quiet
"""

    write_script(startup_text, "scripts/startup.sh")

    create_text = f"""
#!/bin/bash

gcloud compute tpus tpu-vm create {node_name} \
--zone={zone} \
--accelerator-type={accelerator_type} \
--version={version} \
--service-account={service_account} \
{"--preemptible" if preemptible else ""}
"""

    write_script(create_text, "scripts/create.sh")

    queue_text = f"""
#!/bin/bash

gcloud alpha compute tpus queued-resources create my-queued-tpu-request \
--node-id={node_name} \
--zone={zone} \
--accelerator-type={accelerator_type} \
--runtime-version={version} \
--service-account={service_account} \
{"--preemptible" if preemptible else ""}
"""

    write_script(queue_text, "scripts/queue.sh")

    connect_text = f"""
#!/bin/bash

gcloud compute tpus tpu-vm ssh {node_name} --zone={zone}
"""

    write_script(connect_text, "scripts/connect.sh")


def write_script(text: str, path: str):
    with open(path, "w") as f:
        f.write(text)

    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)


if __name__ == "__main__":
    app()
