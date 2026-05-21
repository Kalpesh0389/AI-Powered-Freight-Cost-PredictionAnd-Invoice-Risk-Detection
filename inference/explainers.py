import os
import joblib
import pandas as pd
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path
import streamlit as st

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "invoice_flagging" / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "invoice_flagging" / "models" / "scaler.pkl"
BACKGROUND_PATH = BASE_DIR / "invoice_flagging" / "models" / "background_data.pkl"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

@st.cache_resource
def load_xai_assets():
    """
    Load the model, scaler, and background data, and initialize the explainers.
    Uses st.cache_resource to cache them in Streamlit for fast performance.
    """
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(f"Model or Scaler pickle file not found. Paths:\nModel: {MODEL_PATH}\nScaler: {SCALER_PATH}")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Load background data
    if BACKGROUND_PATH.exists():
        background_df = joblib.load(BACKGROUND_PATH)
    else:
        # Fallback if background data file is missing (e.g. generate a dummy or return empty)
        # We will make sure to generate it, but a fallback prevents crashing
        background_df = pd.DataFrame(
            [[50, 350.0, 1.73, 162, 2476.0]],
            columns=FEATURES
        )
        
    features = list(background_df.columns)
    
    # Initialize SHAP explainer
    # TreeExplainer is fast for tree-based models like Random Forest
    shap_explainer = shap.TreeExplainer(model)
    
    # Initialize LIME explainer
    # Lime Tabular Explainer needs to predict on raw (unscaled) inputs.
    # So we define a prediction function that scales before predicting.
    def lime_predict_fn(x_raw):
        # Ensure we convert 1D to 2D if needed
        if len(x_raw.shape) == 1:
            x_raw = x_raw.reshape(1, -1)
        x_scaled = scaler.transform(pd.DataFrame(x_raw, columns=FEATURES))
        return model.predict_proba(x_scaled)
        
    lime_explainer = LimeTabularExplainer(
        training_data=background_df.values,
        feature_names=FEATURES,
        class_names=["Safe", "Flagged"],
        mode="classification",
        random_state=42
    )
    
    return model, scaler, background_df, shap_explainer, lime_explainer, lime_predict_fn


def explain_invoice_shap(shap_explainer, scaler, input_row):
    """
    Generate SHAP values for a single input row.
    """
    row_values = [input_row[f][0] if isinstance(input_row[f], list) else input_row[f] for f in FEATURES]
    
    # Construct a DataFrame
    df_raw = pd.DataFrame([row_values], columns=FEATURES)
    df_scaled = scaler.transform(df_raw)
    
    # Calculate SHAP values
    shap_vals = shap_explainer.shap_values(df_scaled)
    
    # Handle version compatibility of shap output
    # TreeExplainer for sklearn RF classifier returns a list of length n_classes,
    # or a numpy array of shape (samples, features, classes), or (samples, features)
    if isinstance(shap_vals, list):
        # Class 1 is 'Flagged' (risk)
        single_shap = shap_vals[1][0]
    elif isinstance(shap_vals, np.ndarray):
        if len(shap_vals.shape) == 3:
            single_shap = shap_vals[0, :, 1]
        elif len(shap_vals.shape) == 2:
            # If 2D (samples x features)
            single_shap = shap_vals[0]
        else:
            single_shap = shap_vals
    else:
        single_shap = shap_vals
        
    # Return as a dictionary of feature -> shap_value
    shap_dict = {feat: float(val) for feat, val in zip(FEATURES, single_shap)}
    
    # Calculate the base value (expected value) of the model for Class 1
    if hasattr(shap_explainer, "expected_value"):
        expected_val = shap_explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
            base_value = float(expected_val[1])
        else:
            base_value = float(expected_val)
    else:
        base_value = 0.5
        
    return shap_dict, base_value


def explain_invoice_lime(lime_explainer, lime_predict_fn, input_row, num_features=5):
    """
    Generate LIME explanation for a single input row.
    """
    row_values = [input_row[f][0] if isinstance(input_row[f], list) else input_row[f] for f in FEATURES]
    
    exp = lime_explainer.explain_instance(
        data_row=np.array(row_values),
        predict_fn=lime_predict_fn,
        num_features=num_features
    )
    
    # Extract explanations as a list of (feature_rule, weight)
    return exp.as_list(), exp


def get_top_reasons(input_row, shap_dict, background_df):
    """
    Extract natural language reasons for why an invoice is flagged or safe.
    """
    reasons = []
    
    invoice_qty = float(input_row["invoice_quantity"][0]) if isinstance(input_row["invoice_quantity"], list) else float(input_row["invoice_quantity"])
    invoice_dlrs = float(input_row["invoice_dollars"][0]) if isinstance(input_row["invoice_dollars"], list) else float(input_row["invoice_dollars"])
    freight = float(input_row["Freight"][0]) if isinstance(input_row["Freight"], list) else float(input_row["Freight"])
    total_qty = float(input_row["total_item_quantity"][0]) if isinstance(input_row["total_item_quantity"], list) else float(input_row["total_item_quantity"])
    total_dlrs = float(input_row["total_item_dollars"][0]) if isinstance(input_row["total_item_dollars"], list) else float(input_row["total_item_dollars"])
    
    # 1. Quantity Mismatch
    qty_diff = abs(invoice_qty - total_qty)
    qty_pct = (qty_diff / total_qty) * 100 if total_qty > 0 else 0
    if qty_diff > 0:
        if invoice_qty > total_qty:
            reasons.append({
                "type": "risk",
                "text": f"Quantity mismatch: Invoice qty ({invoice_qty:.0f}) is higher than PO qty ({total_qty:.0f}) by +{qty_pct:.1f}%",
                "importance": abs(shap_dict.get("invoice_quantity", 0)) + abs(shap_dict.get("total_item_quantity", 0)) + 0.1
            })
        else:
            reasons.append({
                "type": "risk",
                "text": f"Quantity mismatch: Invoice qty ({invoice_qty:.0f}) is lower than PO qty ({total_qty:.0f}) by -{qty_pct:.1f}%",
                "importance": abs(shap_dict.get("invoice_quantity", 0)) + abs(shap_dict.get("total_item_quantity", 0)) + 0.1
            })
            
    # 2. Price/Dollars Mismatch
    dlrs_diff = abs(invoice_dlrs - total_dlrs)
    dlrs_pct = (dlrs_diff / total_dlrs) * 100 if total_dlrs > 0 else 0
    if dlrs_diff > 5:
        if invoice_dlrs > total_dlrs:
            reasons.append({
                "type": "risk",
                "text": f"Price mismatch: Invoice is ${invoice_dlrs:,.2f} but PO is ${total_dlrs:,.2f} (Difference: +${dlrs_diff:,.2f} / +{dlrs_pct:.1f}%)",
                "importance": abs(shap_dict.get("invoice_dollars", 0)) + abs(shap_dict.get("total_item_dollars", 0)) + 0.1
            })
        else:
            reasons.append({
                "type": "risk",
                "text": f"Price mismatch: Invoice is ${invoice_dlrs:,.2f} but PO is ${total_dlrs:,.2f} (Difference: -${dlrs_diff:,.2f} / -{dlrs_pct:.1f}%)",
                "importance": abs(shap_dict.get("invoice_dollars", 0)) + abs(shap_dict.get("total_item_dollars", 0)) + 0.1
            })
            
    # 3. Freight Cost Check
    avg_freight = float(background_df["Freight"].mean())
    freight_ratio_to_invoice = (freight / invoice_dlrs) * 100 if invoice_dlrs > 0 else 0
    if freight > 1.5 * avg_freight:
        excess_pct = ((freight - avg_freight) / avg_freight) * 100
        reasons.append({
            "type": "risk",
            "text": f"Freight cost unusually high: ${freight:,.2f} (+{excess_pct:.1f}% above historical average of ${avg_freight:,.2f})",
            "importance": abs(shap_dict.get("Freight", 0)) + 0.05
        })
    elif freight > 0 and freight_ratio_to_invoice > 15:
        reasons.append({
            "type": "risk",
            "text": f"Freight cost ratio is high: Freight represents {freight_ratio_to_invoice:.1f}% of total invoice cost",
            "importance": abs(shap_dict.get("Freight", 0)) + 0.02
        })
    
    # Sort features by positive contributions (risk factors)
    positive_contribs = sorted([(feat, val) for feat, val in shap_dict.items() if val > 0.02], key=lambda x: x[1], reverse=True)
    for feat, val in positive_contribs:
        # Avoid duplicate reasons if they relate to quantity or price which are already described
        if feat == "invoice_quantity" or feat == "total_item_quantity":
            if any("Quantity mismatch" in r["text"] for r in reasons):
                continue
        if feat == "invoice_dollars" or feat == "total_item_dollars":
            if any("Price mismatch" in r["text"] for r in reasons):
                continue
        if feat == "Freight":
            if any("Freight cost" in r["text"] for r in reasons):
                continue
                
        feat_name_pretty = feat.replace("_", " ").title()
        reasons.append({
            "type": "risk",
            "text": f"High value in {feat_name_pretty} acts as a risk driver (+{val*100:.1f}% risk)",
            "importance": val
        })
        
    # If the model says it's SAFE, highlight the top negative contributions (saving factors)
    negative_contribs = sorted([(feat, val) for feat, val in shap_dict.items() if val < -0.02], key=lambda x: x[1])
    safe_reasons = []
    
    # Check if quantity matches
    if qty_diff == 0:
        safe_reasons.append({
            "type": "safe",
            "text": f"Quantity aligns perfectly: Invoice matches PO quantity exactly ({invoice_qty:.0f} items)",
            "importance": abs(shap_dict.get("invoice_quantity", 0)) + abs(shap_dict.get("total_item_quantity", 0)) + 0.1
        })
    # Check if price matches
    if dlrs_diff <= 5:
        safe_reasons.append({
            "type": "safe",
            "text": f"Price aligns perfectly: Invoice cost (${invoice_dlrs:,.2f}) matches PO cost (${total_dlrs:,.2f})",
            "importance": abs(shap_dict.get("invoice_dollars", 0)) + abs(shap_dict.get("total_item_dollars", 0)) + 0.1
        })
        
    # Check if freight is low/normal
    if freight <= avg_freight:
        safe_reasons.append({
            "type": "safe",
            "text": f"Freight cost is normal: ${freight:,.2f} is within historical average of ${avg_freight:,.2f}",
            "importance": abs(shap_dict.get("Freight", 0)) + 0.05
        })
        
    for feat, val in negative_contribs:
        if feat == "invoice_quantity" or feat == "total_item_quantity":
            if any("quantity aligns" in r["text"].lower() for r in safe_reasons):
                continue
        if feat == "invoice_dollars" or feat == "total_item_dollars":
            if any("price aligns" in r["text"].lower() for r in safe_reasons):
                continue
        if feat == "Freight":
            if any("freight cost" in r["text"].lower() for r in safe_reasons):
                continue
                
        feat_name_pretty = feat.replace("_", " ").title()
        safe_reasons.append({
            "type": "safe",
            "text": f"{feat_name_pretty} acts as a safety driver (reduces risk by {abs(val)*100:.1f}%)",
            "importance": abs(val)
        })
        
    # Merge and return
    # Sort reasons by importance descending
    reasons = sorted(reasons, key=lambda x: x["importance"], reverse=True)
    safe_reasons = sorted(safe_reasons, key=lambda x: x["importance"], reverse=True)
    
    return {
        "risk_reasons": [r["text"] for r in reasons][:3], # Top 3 risk reasons
        "safe_reasons": [r["text"] for r in safe_reasons][:3] # Top 3 safe reasons
    }
