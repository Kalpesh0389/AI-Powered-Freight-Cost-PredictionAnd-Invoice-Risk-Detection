import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag
from inference.explainers import load_xai_assets, explain_invoice_shap, explain_invoice_lime, get_top_reasons

# Page Configuration
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="🚚",
    layout="wide",
)
# Header Section
st.markdown(
    """
# 🚚 Vendor Invoice Intelligence Portal
### AI-DRiven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to
- **Forecast freight cost accurately**
- **Detect risky or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
    """
)

st.divider()

# Slider

st.sidebar.title(" Model Selection")
selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---
**Business Impact:**
- Improved cost frecasting
- Reduced invoice fraud and anomalies
- Faster finance operations
"""
)

# Freight cost prediction
if selected_model == "Freight Cost Prediction":
    st.subheader("🚚 Freight Cost Prediction")
    st.markdown(
        """
        **Objective:** 
        Predict freight cost for a vendor invoice using **Invoice Dollars**
        to support budgeting, forecasting, and vendor negotiations.
        """
    )
    with st.form("freight_form"):
        dollars = st.number_input(
        "Invoice Dollars",
        min_value=1.0,
        value=18500.0
                )
        submit_freight = st.form_submit_button("Predict Freight Cost")

    if submit_freight:
         input_data = {
              "Dollars": [dollars]
         }

         prediction = predict_freight_cost(input_data)['Predicted_Freight']
         st.success("Prediction completed successfully!")

         st.metric(
              label = "Estimated Freight Cost",
              value = f"${prediction[0]:,.2f}"
         )
# Invoice flag prediction
else:
    st.subheader("⚠️ Invoice Manual Approval Flag Prediction")
    st.markdown(
        """
        **Objective:** 
        Predict whether a vendor invoice requires manual approval based on **Quantity** and **Invoice Dollars** to identify potential risks and anomalies.
        """
    )
    with st.form("invoice_form"):
        col1,col2,col3 = st.columns(3)
        with col1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value = 50)
            freight = st.number_input(
                "Freight Cost ($)",
                min_value=0.0,
                value = 1.73
            )
        with col2:
                invoice_dollars = st.number_input(
                     "Invoice Dollars",
                     min_value=1.0,
                     value = 352.95
                )
                total_item_quantity = st.number_input(
                        "Total Item Quantity",
                        min_value=1,
                        value = 162
                )
        with col3:
                total_item_dollars = st.number_input(
                        "Total Item Dollars",
                        min_value=1.0,
                        value = 2476.0
                )
        submit_flag = st.form_submit_button("Evaluate Invoice Risk")
    
    if submit_flag:
        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }
        result = predict_invoice_flag(input_data)
        
        is_flagged = bool(result['Predicted_Flag'][0])
        confidence = result['Confidence'][0]
        
        if is_flagged:
            st.error("⚠️ This invoice requires **Manual Approval**")
            st.metric(
                "Model Confidence",
                f"{confidence*100:.1f}%"
            )
        else:
            st.success("✅ This invoice is **Safe for Auto-Approval**")
            st.metric(
                "Model Confidence",
                f"{confidence*100:.1f}%"
            )

        st.markdown("---")
        st.markdown("### 🔍 Explainable AI (XAI) Dashboard")
        
        # Load explainers
        with st.spinner("Calculating explainability models (SHAP & LIME)..."):
            try:
                model, scaler, background_df, shap_explainer, lime_explainer, lime_predict_fn = load_xai_assets()
                
                # Calculate SHAP & LIME
                shap_dict, base_value = explain_invoice_shap(shap_explainer, scaler, input_data)
                lime_list, lime_exp = explain_invoice_lime(lime_explainer, lime_predict_fn, input_data)
                reasons = get_top_reasons(input_data, shap_dict, background_df)
                
                # 1. Main indicators: Risk score and Top reasons
                col_gauge, col_reasons = st.columns([2, 3])
                
                # Calculate risk score percentage
                risk_prob = confidence if is_flagged else (1.0 - confidence)
                risk_score_pct = risk_prob * 100.0
                
                with col_gauge:
                    # Plotly Gauge Chart for Risk Score
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_score_pct,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Invoice Risk Score", 'font': {'size': 20, 'color': '#1E293B', 'weight': 'bold'}},
                        number = {'suffix': "%", 'font': {'size': 40, 'color': '#0F172A'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                            'bar': {'color': "#334155"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#CBD5E1",
                            'steps': [
                                {'range': [0, 35], 'color': '#DEF7EC'}, # Light green
                                {'range': [35, 70], 'color': '#FEF08A'}, # Light yellow
                                {'range': [70, 100], 'color': '#FDE8E8'} # Light red
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70.0
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        height=250,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                with col_reasons:
                    st.markdown("#### 💡 Key Decision Drivers")
                    if is_flagged:
                        st.markdown("##### 🚨 Top reasons this invoice was flagged:")
                        for reason in reasons["risk_reasons"]:
                            st.markdown(f"- ⚠️ {reason}")
                        if not reasons["risk_reasons"]:
                            st.markdown("- *No major anomalies detected, but cumulative risk met threshold.*")
                    else:
                        st.markdown("##### 🛡️ Safety drivers for auto-approval:")
                        for reason in reasons["safe_reasons"]:
                            st.markdown(f"- ✅ {reason}")
                        if not reasons["safe_reasons"]:
                            st.markdown("- *General feature values within normal safe bounds.*")
                            
                # 2. Tabs for SHAP and LIME detailed explanations
                st.markdown("#### Detailed Feature Contributions")
                tab_shap, tab_lime = st.tabs(["📊 SHAP Values (Global Attribution)", "🔬 LIME Explanation (Local Sensitivity)"])
                
                with tab_shap:
                    st.markdown(
                        """
                        **SHAP (SHapley Additive exPlanations)** calculates the contribution of each feature to the difference between 
                        the actual prediction and the average prediction of the model. Positive values push the risk *up*, negative values pull the risk *down*.
                        """
                    )
                    
                    # Construct Plotly Horizontal Bar Chart for SHAP
                    features_pretty = []
                    raw_vals = [
                        invoice_quantity,
                        invoice_dollars,
                        freight,
                        total_item_quantity,
                        total_item_dollars
                    ]
                    features_raw = ["invoice_quantity", "invoice_dollars", "Freight", "total_item_quantity", "total_item_dollars"]
                    
                    for feat, val in zip(features_raw, raw_vals):
                        name = feat.replace("_", " ").title()
                        if "dollars" in feat.lower() or feat == "Freight":
                            features_pretty.append(f"{name} (${val:,.2f})")
                        else:
                            features_pretty.append(f"{name} ({val:,.0f})")
                            
                    shap_vals_sorted = [shap_dict[f] for f in features_raw]
                    
                    # Sort for plot (from lowest to highest)
                    indices = np.argsort(shap_vals_sorted)
                    y_labels = [features_pretty[i] for i in indices]
                    x_values = [shap_vals_sorted[i] * 100.0 for i in indices] # convert to pct
                    colors = ["#EF4444" if x > 0 else "#10B981" for x in x_values] # red for positive, green for negative
                    
                    fig_shap = go.Figure(go.Bar(
                        x=x_values,
                        y=y_labels,
                        orientation='h',
                        marker_color=colors,
                        hovertemplate="Feature: %{y}<br>Contribution: %{x:+.1f}%<extra></extra>"
                    ))
                    
                    fig_shap.update_layout(
                        title="Feature Contributions to Risk Probability (SHAP)",
                        xaxis_title="Risk Impact Percentage Points (+/-)",
                        height=280,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                with tab_lime:
                    st.markdown(
                        """
                        **LIME (Local Interpretable Model-agnostic Explanations)** builds a local surrogate linear model around 
                        the current invoice input to approximate the decision boundary. It tells us how sensitive the model is 
                        to changes in the features right around the current inputs.
                        """
                    )
                    
                    # Construct Plotly Bar Chart for LIME
                    lime_features = [x[0] for x in lime_list]
                    lime_weights = [x[1] for x in lime_list]
                    
                    # Sort LIME list
                    lime_indices = np.argsort(lime_weights)
                    lime_y = [lime_features[i] for i in lime_indices]
                    lime_x = [lime_weights[i] for i in lime_indices]
                    lime_colors = ["#EF4444" if x > 0 else "#10B981" for x in lime_x]
                    
                    fig_lime = go.Figure(go.Bar(
                        x=lime_x,
                        y=lime_y,
                        orientation='h',
                        marker_color=lime_colors,
                        hovertemplate="Rule: %{y}<br>LIME Weight: %{x:+.3f}<extra></extra>"
                    ))
                    
                    fig_lime.update_layout(
                        title="Local Linear Approximated Weights (LIME)",
                        xaxis_title="Local Prediction Weight (+/-)",
                        height=280,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_lime, use_container_width=True)
                    
            except Exception as xai_err:
                st.warning(f"Unable to load explainability dashboard: {xai_err}")
                st.exception(xai_err)