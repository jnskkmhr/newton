# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Export the deterministic actor from W&B run jqur5jhm for the G1 MPM example."""

import argparse
import tempfile
from pathlib import Path

import torch

RUN_PATH = "jnskkmr/g1_29dof_mpm_mixed_terrain/jqur5jhm"
CHECKPOINT_NAME = "model_19999.pt"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "policies" / "jqur5jhm.pt"


def export_policy(checkpoint: Path, output: Path) -> None:
    """Export this run's unnormalized ELU actor, excluding its training distribution."""
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = data["actor_state_dict"]
    expected = {"distribution.std_param"} | {
        f"mlp.{layer}.{parameter}" for layer in (0, 2, 4, 6) for parameter in ("weight", "bias")
    }
    if set(state) != expected:
        raise ValueError("Checkpoint does not match the jqur5jhm actor; unexpected parameters or normalization.")
    actor = torch.nn.Sequential(
        torch.nn.Linear(937, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 29),
    ).eval()
    actor.load_state_dict(
        {name.removeprefix("mlp."): value for name, value in state.items() if name.startswith("mlp.")}
    )
    exported = torch.jit.script(actor)
    with torch.inference_mode():
        sample = torch.randn(4, 937)
        torch.testing.assert_close(exported(sample), actor(sample))
    output.parent.mkdir(parents=True, exist_ok=True)
    exported.save(str(output))
    print(f"Exported {RUN_PATH}/{CHECKPOINT_NAME} actor to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Use a previously downloaded model_19999.pt.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.checkpoint is not None:
        export_policy(args.checkpoint, args.output)
        return
    import wandb  # noqa: PLC0415 -- optional dependency needed only for downloading

    # Authenticate against the service hosting this run, using the user's W&B credentials.
    run = wandb.Api(overrides={"base_url": "https://api.forge.coreweave.com"}).run(RUN_PATH)
    with tempfile.TemporaryDirectory(prefix="newton-jqur5jhm-") as directory:
        run.file(CHECKPOINT_NAME).download(root=directory)
        export_policy(Path(directory) / CHECKPOINT_NAME, args.output)


if __name__ == "__main__":
    main()
