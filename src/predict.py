from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

MODEL_PATH = MODEL_DIR / "decision_tree.joblib"
FAMILY_MODEL_PATH = MODEL_DIR / "attack_family.joblib"

RECOMMENDATIONS = {
    "DOS": [
        "Review request/connection rates and resource utilization.",
        "Correlate source and destination IPs with firewall, flow, and server logs.",
        "Check for service degradation or repeated connection attempts."
    ],
    "PROBE": [
        "Review destination ports and services contacted by the source.",
        "Look for sequential port scans or connections to many hosts.",
        "Check firewall and IDS logs for related reconnaissance activity."
    ],
    "R2L": [
        "Review authentication failures and remote-login events.",
        "Check account activity, source IP reputation, and unusual access times.",
        "Inspect relevant host and application logs for unauthorized access."
    ],
    "U2R": [
        "Review privilege changes, sudo/admin events, and process execution.",
        "Inspect endpoint logs for unexpected privilege escalation.",
        "Preserve relevant logs and investigate the affected host."
    ],
    "Other/Unmapped": [
        "The family model could not assign a known NSL-KDD family.",
        "Review raw traffic and host/service logs; do not infer a specific attack type."
    ],
    "Normal": [
        "No attack was identified by the primary model for this record.",
        "Continue normal monitoring; a model prediction does not guarantee traffic is safe."
    ]
}


def load_model():
    """Load the existing binary Normal-vs-Attack pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Binary model not found: {MODEL_PATH}. Run your training script first."
        )
    return joblib.load(MODEL_PATH)


def load_family_model():
    """Load the separately trained attack-family classifier."""
    if not FAMILY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Attack-family model not found: {FAMILY_MODEL_PATH}. "
            "Run: python src/train_attack_family.py"
        )
    return joblib.load(FAMILY_MODEL_PATH)


def _pipeline_parts(model):
    if not hasattr(model, "named_steps"):
        raise ValueError("Expected a scikit-learn Pipeline for the Decision Tree model.")
    if "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
        raise ValueError("The Decision Tree pipeline must contain preprocessor and classifier steps.")
    return model.named_steps["preprocessor"], model.named_steps["classifier"]


def _explain_decision_path(model, input_data, max_conditions=6):
    """Describe conditions actually followed through the fitted Decision Tree."""
    try:
        preprocessor, classifier = _pipeline_parts(model)
        transformed = preprocessor.transform(input_data)
        tree = classifier.tree_
        node_indicator = classifier.decision_path(transformed)
        nodes = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        transformed_names = list(preprocessor.get_feature_names_out())

        # Build exact transformed-feature -> original categorical field/category mappings.
        categorical_map = {}
        try:
            cat_encoder = preprocessor.named_transformers_["categorical"]
            cat_feature_names = cat_encoder.get_feature_names_out(
                ["protocol_type", "service", "flag"]
            )
            for name in cat_feature_names:
                for original in ("protocol_type", "service", "flag"):
                    prefix = original + "_"
                    if name.startswith(prefix):
                        categorical_map["categorical__" + name] = (
                            original, name[len(prefix):]
                        )
                        break
        except Exception:
            categorical_map = {}

        numerical_scaler = preprocessor.named_transformers_.get("numerical")
        numerical_names = [
            name for name in FEATURE_NAMES
            if name not in {"protocol_type", "service", "flag"}
        ]
        conditions = []

        for node_id in nodes:
            feature_idx = int(tree.feature[node_id])
            if feature_idx < 0:
                continue

            transformed_name = transformed_names[feature_idx]
            threshold = float(tree.threshold[node_id])
            value = float(transformed[0, feature_idx])
            went_left = value <= threshold

            if transformed_name in categorical_map:
                original, category = categorical_map[transformed_name]
                if went_left:
                    condition = f"{original} is not {category}"
                else:
                    condition = f"{original} is {category}"
            else:
                original = transformed_name.split("__", 1)[-1]
                raw_threshold = threshold
                raw_value = input_data.iloc[0][original]
                try:
                    raw_value = float(raw_value)
                except (ValueError, TypeError):
                    raw_value = value

                # The numerical branch is standardized; convert its threshold back to raw units.
                if numerical_scaler is not None and original in numerical_names:
                    try:
                        idx = numerical_names.index(original)
                        raw_threshold = threshold * float(numerical_scaler.scale_[idx]) + float(
                            numerical_scaler.mean_[idx]
                        )
                    except (AttributeError, IndexError, TypeError, ValueError):
                        raw_threshold = threshold

                comparator = "<=" if went_left else ">"
                condition = (
                    f"{original} ({raw_value:g}) {comparator} "
                    f"{raw_threshold:.4g}"
                )

            conditions.append(condition)
            if len(conditions) >= max_conditions:
                break

        if not conditions:
            return ["The model returned a prediction, but its decision path could not be summarized."]
        return conditions
    except Exception:
        # Do not let explanation generation prevent the primary prediction from working.
        return ["Decision-path details were unavailable for this record."]


def predict_traffic(model, features):
    """Backward-compatible binary prediction plus model-based explanation."""
    if not isinstance(features, (list, tuple)) or len(features) != len(FEATURE_NAMES):
        received = len(features) if isinstance(features, (list, tuple)) else "non-array"
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, but received {received}."
        )

    input_data = pd.DataFrame([features], columns=FEATURE_NAMES)
    prediction = int(model.predict(input_data)[0])
    is_attack = prediction == 1

    family = "Not assigned (primary model classified Normal)"
    if is_attack:
        family = "Unavailable"

    return {
        "prediction": prediction,
        "label": "Attack" if is_attack else "Normal",
        "explanation": _explain_decision_path(model, input_data),
        "attack_family": family,
        "recommendations": RECOMMENDATIONS["Normal" if not is_attack else "Other/Unmapped"]
    }


def analyze_traffic(binary_model, family_model, features, simplified=False):
    """Return consistent binary prediction, family estimate, path explanation and checks."""
    if not isinstance(features, (list, tuple)) or len(features) != len(FEATURE_NAMES):
        received = len(features) if isinstance(features, (list, tuple)) else "non-array"
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, but received {received}."
        )

    input_data = pd.DataFrame([features], columns=FEATURE_NAMES)
    prediction = int(binary_model.predict(input_data)[0])
    is_attack = prediction == 1
    explanation = _explain_decision_path(binary_model, input_data)

    if not is_attack:
        family = "Not applicable — primary model classified Normal"
        recommendations = RECOMMENDATIONS["Normal"]
        family_note = (
            "The family model is only reported when the primary binary model "
            "classifies the record as an attack."
        )
    else:
        try:
            predicted_family = str(family_model.predict(input_data)[0]).upper()
            if predicted_family in {"DOS", "PROBE", "R2L", "U2R"}:
                family = predicted_family
                recommendations = RECOMMENDATIONS[predicted_family]
                family_note = (
                    "Estimated family from a separate classifier. On KDDTest+, "
                    "R2L and U2R recall was low, so these estimates can be wrong."
                )
            elif predicted_family == "NORMAL":
                family = "Uncertain — family model predicted Normal"
                recommendations = RECOMMENDATIONS["Other/Unmapped"]
                family_note = (
                    "The primary model says Attack but the family model says Normal; "
                    "the family result is conflicting and should not be trusted."
                )
            else:
                family = "Other/Unmapped"
                recommendations = RECOMMENDATIONS["Other/Unmapped"]
                family_note = "The family model did not return a mapped attack family."
        except Exception:
            family = "Unavailable"
            recommendations = RECOMMENDATIONS["Other/Unmapped"]
            family_note = "Attack-family prediction failed; use the binary result cautiously."

    return {
        "prediction": prediction,
        "label": "Attack" if is_attack else "Normal",
        "attack_family": family,
        "explanation": explanation,
        "recommendations": recommendations,
        "family_note": family_note,
        "explanation_note": (
            "Simplified single-traffic mode uses only 8 entered values; the other "
            "33 features are set to zero. This is an approximation."
            if simplified else
            "Explanation conditions are from the actual Decision Tree path. "
            "They describe model behavior, not proof of malicious activity."
        )
    }
