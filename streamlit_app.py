import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title='UNSW NB15 Attack Classifier', layout='wide')

@st.cache_data
def load_data():
    df = pd.read_csv('UNSW_NB15_training-set.csv')
    return df

@st.cache_resource
def train_model(df, model_name):
    features = [
        'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
        'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
        'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
        'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
        'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
        'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
        'is_sm_ips_ports'
    ]

    X = df[features].copy()
    y = df['attack_cat'].copy()

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        X[col] = X[col].fillna('unknown').astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    X.fillna(0, inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=500, random_state=42)
    else:
        model = SVC(kernel='linear', probability=True, random_state=42)

    model.fit(X_scaled, y)
    return model, scaler, label_encoders, features

@st.cache_data
def prepare_sample(df, idx):
    return df.iloc[idx:idx+1].copy()


def transform_sample(sample, scaler, label_encoders, features):
    X_sample = sample[features].copy()
    for col, le in label_encoders.items():
        X_sample[col] = X_sample[col].fillna('unknown').astype(str)
        X_sample[col] = X_sample[col].map(lambda x: x if x in le.classes_ else 'unknown')
        X_sample[col] = le.transform(X_sample[col])
    X_sample = X_sample.fillna(0)
    return scaler.transform(X_sample)


def main():
    st.title('UNSW NB15 Attack Category Classifier')
    st.markdown(
        'This Streamlit frontend loads the UNSW NB15 training dataset, trains a simple classifier, '
        'and allows you to inspect predictions for individual examples.'
    )

    df = load_data()
    st.sidebar.header('Options')
    model_name = st.sidebar.selectbox('Model', ['Random Forest', 'Logistic Regression', 'SVM'])
    sample_index = st.sidebar.number_input('Sample index', min_value=0, max_value=len(df) - 1, value=0, step=1)
    show_sample = st.sidebar.checkbox('Show selected row features', value=True)

    st.subheader('Dataset overview')
    st.write('Training set shape:', df.shape)
    st.dataframe(df.head(10))
    st.write('Attack category distribution:')
    st.bar_chart(df['attack_cat'].value_counts())

    with st.spinner('Training the model...'):
        model, scaler, label_encoders, features = train_model(df, model_name)

    X = df[features].copy()
    for col, le in label_encoders.items():
        X[col] = X[col].fillna('unknown').astype(str)
        X[col] = le.transform(X[col])
    X = scaler.transform(X.fillna(0))
    y_pred = model.predict(X)
    st.success('Model trained successfully!')

    st.subheader('Training accuracy')
    st.write(f'**{model_name} accuracy on training data:** {accuracy_score(df["attack_cat"], y_pred):.4f}')

    sample = prepare_sample(df, sample_index)
    if show_sample:
        st.subheader('Selected sample')
        st.write(sample.T)

    X_sample = transform_sample(sample, scaler, label_encoders, features)
    prediction = model.predict(X_sample)[0]
    probabilities = model.predict_proba(X_sample)[0] if hasattr(model, 'predict_proba') else None

    st.subheader('Prediction')
    st.write('Predicted attack category:', prediction)
    if probabilities is not None:
        prob_df = pd.DataFrame({
            'attack_cat': model.classes_,
            'probability': probabilities
        }).sort_values(by='probability', ascending=False)
        st.write(prob_df)

    st.markdown('---')
    st.write('Use the sample index in the sidebar to choose a different row from the dataset.')


if __name__ == '__main__':
    main()
