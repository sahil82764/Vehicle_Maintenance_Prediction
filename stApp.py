import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
from vmpred.util.util import read_yaml_file, load_object
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Configuration ---
preprocessor_path = "data/transformedData/preprocessorObject/preprocessor.pkl"
clModel_path = "model/bestEvaluatedModel/DecisionTreeClassifier.pkl"
rgModel_path = "model/bestEvaluatedModel/Ridge.pkl"
schema_path = "config/schema.yaml"


def display_modal(results):
    with st.expander("What-If Analysis Results", expanded=True):
        st.subheader("What-If Scenarios and Predictions")
        for result in results:
            st.write(f"**Scenario:** {result['Scenario']}")
            st.write(f"Predicted Fuel Efficiency: **{result['Predicted_Fuel_Efficiency']:.5f} L/100km**")
            st.markdown("---")


# --- Load Models and Preprocessor ---
def load_models_and_preprocessor():
    try:
        preprocessor = load_object(preprocessor_path)
        clModel = load_object(clModel_path)
        rgModel = load_object(rgModel_path)
        return preprocessor, clModel, rgModel
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# --- Feature Engineering ---
def feature_engineering(df):
    df['last_service_date'] = pd.to_datetime(df['last_service_date'])
    df['warranty_expiry_date'] = pd.to_datetime(df['warranty_expiry_date'])
    df['time_since_last_service'] = (pd.to_datetime('2024-04-01') - df['last_service_date']).dt.days
    df['warranty_duration'] = (df['warranty_expiry_date'] - pd.to_datetime('2024-04-01')).dt.days
    df['mileage_per_year'] = df['mileage'] / df['vehicle_age']
    df['service_frequency'] = df['service_history'] / df['vehicle_age']
    df['accident_rate'] = df['accident_history'] / df['vehicle_age']
    maintenance_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
    df['maintenance_history'] = df['maintenance_history'].map(maintenance_mapping)
    condition_mapping = {'Worn Out': 1, 'Good': 2, 'New': 3}
    df['tire_condition'] = df['tire_condition'].map(condition_mapping)
    df['brake_condition'] = df['brake_condition'].map(condition_mapping)
    status_mapping = {'Weak': 1, 'Good': 2, 'New': 3}
    df['battery_status'] = df['battery_status'].map(status_mapping)
    df = df.drop(columns=['last_service_date', 'warranty_expiry_date'], errors='ignore')
    return df

# --- Predict Function ---
def predict(input_df, preprocessor, clModel, rgModel):
    try:
        input_df = feature_engineering(input_df)
        processed_data = preprocessor.transform(input_df)
        need_maintenance_prediction = clModel.predict(processed_data)[0]
        confidence = clModel.predict_proba(processed_data)[:, 1][0]
        fuel_efficiency_prediction = rgModel.predict(processed_data)[0]
        return need_maintenance_prediction, confidence, fuel_efficiency_prediction, input_df
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None, None

# --- What-If Analysis ---
def what_if_analysis(input_df, preprocessor, rgModel, scenarios):
    results = []
    for scenario in scenarios:
        modified_df = input_df.copy()
        for feature, value in scenario.items():
            if feature in modified_df.columns:
                modified_df[feature] = value
        processed_data = preprocessor.transform(modified_df)
        fuel_efficiency_prediction = rgModel.predict(processed_data)[0]
        results.append({
            "Scenario": scenario,
            "Predicted_Fuel_Efficiency": fuel_efficiency_prediction
        })
    return results

# --- Main App ---
def main():
    st.title("Vehicle Maintenance Prediction App")
    st.markdown("Please fill in the vehicle details below:")

    schema = read_yaml_file(schema_path)
    preprocessor, clModel, rgModel = load_models_and_preprocessor()

    if preprocessor is None or clModel is None or rgModel is None:
        st.error("Failed to load models or preprocessor.")
        return

    with st.form("input_form"):
        input_data = {}
        for col_name in schema['columns']:
            if col_name not in schema['target_column']:
                col_values = schema['domain_value'].get(col_name, None)
                if col_name in schema.get('numerical_columns', []):
                    input_data[col_name] = st.number_input(col_name, value=0.0, step=0.1)
                elif col_name in schema.get('categorical_columns', []):
                    input_data[col_name] = st.selectbox(col_name, col_values)
                elif col_name in schema.get('date_columns', []):
                    input_data[col_name] = st.date_input(col_name, value=datetime.now())
        
        input_df = pd.DataFrame([input_data])

        # Initialize processed_df in session state
        if "predictions" not in st.session_state:
            st.session_state["predictions"] = None
        if "processed_df" not in st.session_state:
            st.session_state["processed_df"] = None

        if st.form_submit_button("Predict"):
            need_maintenance_prediction, confidence, fuel_efficiency_prediction, processed_df = predict(input_df, preprocessor, clModel, rgModel)
            if need_maintenance_prediction is not None:
                st.session_state["processed_df"] = processed_df
                st.subheader("Prediction Results")
                if need_maintenance_prediction == 0:
                    st.success(f"The engine is predicted to be in a normal condition. As the  Confidence level of the machine is : {1 - confidence:.2%}")
                else:
                    st.warning(f"Warning! Please investigate further,As the  Confidence level of the machine is : {1 - confidence:.2%}")
                st.write(f"Predicted Fuel Efficiency: {fuel_efficiency_prediction:.5f}")

    st.markdown("---")
    st.subheader("What-If Analysis")
    st.markdown("Test the impact of different maintenance scenarios on fuel efficiency.")
    
    scenarios = [
        {"tire_condition": 3, "brake_condition": 3, "battery_status": 3},
        {"tire_condition": 1, "brake_condition": 2, "battery_status": 2},
        {"tire_condition": 1, "brake_condition": 1, "battery_status": 1}
    ]

    if st.button("Run What-If Analysis"):
        if st.session_state["processed_df"] is not None:
            results = what_if_analysis(st.session_state["processed_df"], preprocessor, rgModel, scenarios)
            display_modal(results)
            # st.write("Results of Scenarios:")
            # for result in results:
            #     st.write(f"Scenario: {result['Scenario']} - Predicted Fuel Efficiency: **{result['Predicted_Fuel_Efficiency']:.5f} L/100km**")
        else:
            st.error("Please run the prediction first to perform What-If Analysis.")

if __name__ == "__main__":
    main()
