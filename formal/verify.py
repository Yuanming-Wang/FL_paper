"""Kernel-check every theorem and audit its axioms using the pinned Lake project."""

import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent


def main():
    source = HERE / "AdapCTA_Formal.lean"
    report = dict(passed=False, checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    output = ""
    try:
        code = source.read_text(encoding="utf-8-sig")
        assert not re.search(r"\b(?:sorry|admit|native_decide|axiom)\b", code), "Unproved declaration or verification shortcut in source"
        names = re.findall(r"(?m)^theorem\s+(\w+)", code)
        assert names and len(names) == len(set(names)), "Missing or duplicate theorem declarations"
        expected = {"AdapCTAFormal." + name for name in names}
        manifest = json.loads((HERE / "lake-manifest.json").read_text())
        revisions = {}
        for package in manifest["packages"]:
            checkout = HERE / manifest["packagesDir"] / package["name"]
            revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                               text=True).strip()
            assert revision == package["rev"], f"Dependency revision mismatch: {package['name']}"
            revisions[package["name"]] = revision
        report["dependency_revisions"] = revisions
        version = subprocess.check_output(["lake", "env", "lean", "--version"], cwd=HERE, text=True).strip()
        required = (HERE / "lean-toolchain").read_text().strip().split(":v")[-1]
        assert f"version {required}," in version, version
        report["lean"] = version
        result = subprocess.run(["lake", "env", "lean", source.name], cwd=HERE,
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        output = result.stdout + result.stderr
        report["return_code"] = result.returncode
        audits = dict(re.findall(r"'([^']+)' depends on axioms:\s*\[([^\]]*)\]", output))
        used = {axiom.strip() for name in expected for axiom in audits.get(name, "").split(",") if axiom.strip()}
        report.update(theorem_count=len(names), audited_theorem_count=len(expected.intersection(audits)),
                      all_theorems_audited=expected <= audits.keys(), axioms=sorted(used),
                      warning_count=len(re.findall(r"(?m)^.*: warning:", output)))
        assert result.returncode == 0, "Lean compilation failed; see verification.log"
        assert expected <= audits.keys(), "A theorem is missing its printed axiom audit"
        assert used <= {"propext", "Classical.choice", "Quot.sound"} and "sorryAx" not in output, "Unexpected axiom dependency"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == report["source_sha256"], "Proof source changed during verification"
        report["passed"] = True
        report["scope"] = ("Analytic game threshold and equilibria; control and projection identities; "
                           "Lyapunov closure from the stated one-round hypotheses; finite-round and "
                           "tolerance bounds; correction coverage and reconstruction identities.")
    except (AssertionError, OSError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
        output += "\nVerification error: " + str(error) + "\n"
    log = HERE / "verification.log"
    log.write_text(output, encoding="utf-8", newline="\n")
    report["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
    report["project_sha256"] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                 for name in ("lean-toolchain", "lakefile.toml", "lake-manifest.json")}
    (HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
