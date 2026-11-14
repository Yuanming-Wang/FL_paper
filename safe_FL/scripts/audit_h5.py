#!/usr/bin/env python3
"""Strict smoke audit for one current_protocol H5 result."""
import argparse
import math
import os
import sys

import h5py
import numpy as np


def finite(name, values):
    values = np.asarray(values)
    if values.size and not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN/Inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5")
    parser.add_argument("--learning-rate", type=float, help="Reject results from a different training learning rate")
    args = parser.parse_args()
    failures = []
    try:
        with h5py.File(args.h5, "r") as handle:
            attrs = dict(handle.attrs)
            if args.learning_rate is not None and not math.isclose(float(attrs.get('local_learning_rate', math.nan)), args.learning_rate, rel_tol=1e-9, abs_tol=0):
                failures.append("local learning rate does not match requested protocol")
            required_attrs = {"algorithm", "dataset", "model", "distribution", "total_layer",
                              "rounds_requested", "num_clients", "join_ratio", "batch_size",
                              "local_epochs", "privacy_protocol", "evaluation_protocol"}
            missing = required_attrs - set(attrs)
            if missing:
                failures.append(f"missing metadata: {sorted(missing)}")
            accuracy = np.asarray(handle["rs_test_acc"])
            finite("rs_test_acc", accuracy)
            if "rounds_requested" in attrs and len(accuracy) != int(attrs["rounds_requested"]):
                failures.append("accuracy length does not match requested rounds")
            if "total_train_cost" in handle:
                total_train_cost = np.asarray(handle["total_train_cost"])
                finite("total_train_cost", total_train_cost)
                if len(total_train_cost) != len(accuracy):
                    failures.append("total_train_cost length does not match accuracy")
            if "rs_test_loss" not in handle:
                failures.append("missing rs_test_loss")
            else:
                test_loss = np.asarray(handle["rs_test_loss"])
                finite("rs_test_loss", test_loss)
                if len(test_loss) != len(accuracy):
                    failures.append("test loss length does not match accuracy")
            if accuracy.size and ((accuracy < 0).any() or (accuracy > 1).any()):
                failures.append("accuracy outside [0,1]")
            cost_names = ["round_total_cost", "round_privacy_cost", "round_energy_cost", "round_model_cost"]
            costs = {name: np.asarray(handle[name]) for name in cost_names if name in handle}
            for name, values in costs.items():
                finite(name, values)
                if (values < -1e-9).any():
                    failures.append(f"{name} has negative values")
            if costs and accuracy.size and any(len(values) != len(accuracy) for values in costs.values()):
                failures.append("accuracy/cost lengths differ")
            if all(name in costs for name in cost_names):
                if not np.allclose(costs["round_total_cost"], costs["round_privacy_cost"] +
                                   costs["round_energy_cost"] + costs["round_model_cost"],
                                   rtol=1e-6, atol=1e-8):
                    failures.append("total cost does not equal component sum")
            if attrs.get("algorithm") == "Adap-CTA":
                if attrs.get("cc_initialization") != "g_i(w0)":
                    failures.append("bad cc initialization metadata")
                expected_sc = 'mean_initial_client_controls' if attrs.get('control_protocol') == 'consistent_v2' else 'zero'
                if attrs.get("sc_initialization") != expected_sc:
                    failures.append("bad sc initialization metadata")
                if attrs.get('control_protocol') == 'consistent_v2' and attrs.get('control_delta_denominator') != 'num_clients':
                    failures.append('bad server-control delta normalization')
                for name in ("decision_history", "round_raw_update_norm", "round_corrected_update_norm",
                             "round_cc_norm", "round_sc_norm", "round_sc_cc_norm",
                             "round_correction_magnitude"):
                    if name not in handle:
                        failures.append(f"missing Adap-CTA mechanism array: {name}")
                    else:
                        finite(name, handle[name])
                decisions = np.asarray(handle["decision_history"])
                total_layer = int(attrs["total_layer"])
                if decisions.ndim != 2 or decisions.shape[0] != len(accuracy):
                    failures.append("decision history shape does not match rounds")
                if decisions.size and ((decisions < 6).any() or (decisions > total_layer).any()):
                    failures.append("decision outside admissible layer range")
                if "round_total_cost" in handle and len(handle["round_total_cost"]) != len(decisions):
                    failures.append("decision/cost lengths differ")
    except Exception as exc:
        failures.append(f"exception: {exc}")
    if failures:
        print("FAIL / INVALID")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
