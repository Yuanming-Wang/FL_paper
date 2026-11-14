"""Export an actual client-upload observation for inversion runners."""

from pathlib import Path
import sys
import torch

from flcore.attacks.gi_nas.bundle import build_victim_model, save_gradient_bundle


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: attack_make_bundle.py OBSERVATION.pt OUTPUT.pt")
    source, output = map(Path, sys.argv[1:])
    bundle = torch.load(source, map_location="cpu", weights_only=False)
    if bundle.get("format") == "fl-paper-adaptive-observation-bundle":
        attack = bundle["attack"]
        mechanism = "Adap-CTA_actual_selected_client_upload"
        algorithm = "Adap-CTA"
    elif bundle.get("format") == "fl-paper-upload-observation-bundle":
        attack = bundle["attack"]
        mechanism = str(attack.get("mechanism", "actual_selected_client_upload"))
        algorithm = str(attack.get("algorithm", "unknown"))
    else:
        raise ValueError("unsupported observation format; capture a real training upload")

    scale = int(attack["num_batches"]) * int(attack["local_epochs"]) * float(attack["learning_rate"])
    if scale <= 0:
        raise ValueError("invalid local update scale")
    model = build_victim_model("resnet", int(attack["num_classes"]))
    model.load_state_dict(attack["model_state_dict"], strict=True)
    # This is an update attack: the server-visible full local update is the
    # target, normalized by the known SGD step scale. Raw gradients and private
    # Adap-CTA control variates are not read.
    gradients = [-value / scale for value in attack["model_update"]]
    evaluation = bundle["evaluation_only"]
    save_gradient_bundle(
        output,
        model,
        "resnet",
        int(attack["num_classes"]),
        gradients,
        evaluation["labels"],
        evaluation["images"],
        metadata={
            "source": "actual_selected_client_upload",
            "observation_mode": "actual_selected_client_local_model_update",
            "threat_model_id": "B_observed_client_upload",
            "approximation_statement": (
                "The attacker observes the complete uploaded local model update "
                "from one selected client and uses its known-SGD-scale effective "
                "update as the optimization target; raw gradients and private "
                "control variates are unavailable."
            ),
            "source_observation": str(source.resolve()),
            "round_index": int(attack["round_index"]),
            "client_id": int(attack["client_id"]),
            "algorithm": algorithm,
            "mechanism": mechanism,
            "local_epochs": int(attack["local_epochs"]),
            "num_batches": int(attack["num_batches"]),
        },
    )
    print(output)


if __name__ == "__main__":
    main()
