import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# LOAD MODEL & DATA
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('xgb_vehicle_price_model.json')
    return model

@st.cache_resource
def load_encoders():
    with open('label_encoders.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_preprocessed_data():
    return pd.read_csv('riyasewana_search_20260220_174117_preprocessed.csv', encoding='utf-8-sig')

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model = load_model()
encoders = load_encoders()
df_raw = load_preprocessed_data()
explainer = load_explainer(model)

# Build make -> model mapping from original data
make_model_map = df_raw.groupby('make')['model'].unique().to_dict()
for k in make_model_map:
    make_model_map[k] = sorted(make_model_map[k].tolist())


# CUSTOM CSS

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0ea5e9 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .price-box {
        background: linear-gradient(135deg, #0ea5e9 0%, #10b981 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(14, 165, 233, 0.3);
    }
    .price-box h1 {
        font-size: 2.8rem;
        margin: 0;
        color: white;
    }
    .price-box p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    .metric-card {
        background: #f0fdf4;
        color: #1a1a2e !important;
        padding: 1.2rem;
        border-radius: 0.8rem;
        border-left: 4px solid #10b981;
        margin: 0.4rem 0;
    }
    .metric-card strong {
        color: #333 !important;
    }
    .stSidebar > div {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    .sidebar-header {
        color: #0ea5e9;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown('<div class="main-header">🚗 Sri Lankan Vehicle Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#64748b; font-size:1.05rem; margin-top:-0.5rem;">'
    'Powered by XGBoost &mdash; Predicts used vehicle prices from Riyasewana.com data</p>',
    unsafe_allow_html=True
)
st.divider()

# SIDEBAR - INPUT FORM

with st.sidebar:
    st.markdown("### 🔧 Vehicle Details")
    st.caption("Enter the vehicle specifications below")

    # Make
    all_makes = sorted(encoders['make'].classes_.tolist())
    selected_make = st.selectbox("Make", all_makes, index=all_makes.index('Toyota') if 'Toyota' in all_makes else 0)

    # Model (filtered by make)
    available_models = make_model_map.get(selected_make, ['Unknown'])
    selected_model = st.selectbox("Model", available_models)

    st.divider()

    # Year of Manufacture
    selected_yom = st.slider("Year of Manufacture", min_value=1980, max_value=2025, value=2018, step=1)

    # Mileage
    selected_mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=80000, step=5000)

    # Engine CC
    selected_engine_cc = st.number_input("Engine CC", min_value=100, max_value=4000, value=1300, step=50)

    st.divider()

    # Gear
    all_gears = sorted(encoders['gear'].classes_.tolist())
    selected_gear = st.selectbox("Transmission", all_gears)

    # Fuel Type
    all_fuels = sorted(encoders['fuel_type'].classes_.tolist())
    selected_fuel = st.selectbox("Fuel Type", all_fuels)

    # Options (checkboxes)
    st.markdown("**Options**")
    opt_ac = st.checkbox("Air Condition", value=True)
    opt_ps = st.checkbox("Power Steering", value=True)
    opt_pm = st.checkbox("Power Mirror", value=True)
    opt_pw = st.checkbox("Power Window", value=True)

    # Build options string matching training data format
    selected_opts = []
    if opt_ac:
        selected_opts.append("AIR CONDITION")
    if opt_ps:
        selected_opts.append("POWER STEERING")
    if opt_pm:
        selected_opts.append("POWER MIRROR")
    if opt_pw:
        selected_opts.append("POWER WINDOW")
    selected_options = ", ".join(selected_opts) if selected_opts else "NONE"

    # Location
    all_locations = sorted(encoders['location'].classes_.tolist())
    selected_location = st.selectbox("Location", all_locations,
        index=all_locations.index('Colombo') if 'Colombo' in all_locations else 0)

    st.divider()
    predict_btn = st.button("🔮 Predict Price", use_container_width=True, type="primary")


# ENCODE INPUT & PREDICT

def encode_input():
    """Encode user inputs using saved label encoders"""
    encoded = {}

    # Encode categoricals (handle unseen labels gracefully)
    for col, val in [('make', selected_make), ('model', selected_model),
                     ('gear', selected_gear), ('fuel_type', selected_fuel),
                     ('options', selected_options), ('location', selected_location)]:
        le = encoders[col]
        if val in le.classes_:
            encoded[col] = le.transform([val])[0]
        else:
            # Fallback: use the most common class
            encoded[col] = 0

    # Numerics
    encoded['yom'] = selected_yom
    encoded['mileage'] = float(selected_mileage)
    encoded['engine_cc'] = float(selected_engine_cc)

    # Create DataFrame with correct column order
    feature_order = ['make', 'model', 'gear', 'fuel_type', 'options', 'location', 'yom', 'mileage', 'engine_cc']
    return pd.DataFrame([encoded], columns=feature_order)

# Always predict on current inputs
input_df = encode_input()
prediction = model.predict(input_df)[0]
shap_values = explainer.shap_values(input_df)


# MAIN CONTENT

col1, col2 = st.columns([1, 1])

with col1:
    # Price display
    st.markdown(f"""
    <div class="price-box">
        <p>Estimated Price</p>
        <h1>Rs {prediction:,.0f}</h1>
        <p>{selected_yom} {selected_make} {selected_model} &bull; {selected_mileage:,} km</p>
    </div>
    """, unsafe_allow_html=True)

    # Vehicle summary
    st.markdown("#### 📋 Vehicle Summary")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Make:</strong> {selected_make}<br>
            <strong>Model:</strong> {selected_model}<br>
            <strong>Year:</strong> {selected_yom}<br>
            <strong>Mileage:</strong> {selected_mileage:,} km
        </div>
        """, unsafe_allow_html=True)
    with summary_col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Engine:</strong> {selected_engine_cc} cc<br>
            <strong>Gear:</strong> {selected_gear}<br>
            <strong>Fuel:</strong> {selected_fuel}<br>
            <strong>Location:</strong> {selected_location}
        </div>
        """, unsafe_allow_html=True)

with col2:
    # SHAP Waterfall
    st.markdown("#### 🔍 Why this price? (SHAP Explanation)")
    fig, ax = plt.subplots(figsize=(8, 5))

    feature_names = list(input_df.columns)
    # Create readable feature labels
    readable_labels = []
    for fname in feature_names:
        val = input_df[fname].iloc[0]
        if fname == 'make':
            readable_labels.append(f"make={selected_make}")
        elif fname == 'model':
            readable_labels.append(f"model={selected_model}")
        elif fname == 'gear':
            readable_labels.append(f"gear={selected_gear}")
        elif fname == 'fuel_type':
            readable_labels.append(f"fuel={selected_fuel}")
        elif fname == 'options':
            readable_labels.append(f"options")
        elif fname == 'location':
            readable_labels.append(f"loc={selected_location}")
        elif fname == 'yom':
            readable_labels.append(f"year={int(val)}")
        elif fname == 'mileage':
            readable_labels.append(f"mileage={int(val):,}km")
        elif fname == 'engine_cc':
            readable_labels.append(f"engine={int(val)}cc")
        else:
            readable_labels.append(fname)

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=readable_labels
        ),
        show=False,
        max_display=9
    )
    plt.title("Feature Contributions to Predicted Price", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# FEATURE IMPORTANCE (GLOBAL)

st.divider()
st.markdown("#### 📊 Global Feature Importance")

fi_col1, fi_col2 = st.columns([2, 1])

with fi_col1:
    feature_importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
    ax2.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
    ax2.set_xlabel('Importance Score')
    ax2.set_title('Which features matter most for pricing?')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

with fi_col2:
    st.markdown("""
    **How to interpret:**
    - **SHAP waterfall** (above) shows how each feature pushes the price up or down *for this specific vehicle*
    - **Feature importance** (left) shows which features matter most *across all vehicles*
    - Red bars push price higher, blue bars push price lower
    """)

    st.markdown("""
    **Model Info:**
    - Algorithm: XGBoost
    - Training samples: 2,466
    - Test R²: 0.7147
    - Test MAE: Rs 1,035,491
    """)

# Footer
st.divider()
st.caption("Built with Streamlit · XGBoost · SHAP | Data scraped from Riyasewana.com")
