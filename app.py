
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fuzzywuzzy import process
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import json, os

st.set_page_config(layout="wide", page_title="Campus Security AI — MVP")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    profiles = pd.read_csv(os.path.join(DATA_DIR,"student_profiles.csv"))
    wifi = pd.read_csv(os.path.join(DATA_DIR,"wifi_logs.csv"))
    swipe = pd.read_csv(os.path.join(DATA_DIR,"swipe_logs.csv"))
    wifi['timestamp'] = pd.to_datetime(wifi['timestamp'])
    swipe['timestamp'] = pd.to_datetime(swipe['timestamp'])
    try:
        notes = pd.read_csv(os.path.join(DATA_DIR,"text_notes.csv"))
        notes['timestamp'] = pd.to_datetime(notes['timestamp'])
    except Exception:
        notes = pd.DataFrame(columns=['student_id','note','timestamp'])
    try:
        with open(os.path.join(DATA_DIR,"image_embeddings.json"),"r") as f:
            im = json.load(f)
        im_df = pd.DataFrame(im)
        im_df['timestamp'] = pd.to_datetime(im_df['timestamp'])
    except Exception:
        im_df = pd.DataFrame(columns=['student_id','embedding','source','timestamp'])
    return profiles, wifi, swipe, notes, im_df

profiles, wifi, swipe, notes, im_df = load_data()

st.title("Campus Security AI — MVP (Railway Deployment)")

# Entity Resolution
st.header("Entity Resolution & Events")
card_map = profiles.set_index('card_id')['student_id'].to_dict()
dev_map = profiles.set_index('device_hash')['student_id'].to_dict()

wifi_ev = wifi.copy()
wifi_ev['student_id'] = wifi_ev['device_hash'].map(dev_map)
wifi_ev['event_type'] = 'wifi'
wifi_ev = wifi_ev.rename(columns={'ap_id':'location'})

swipe_ev = swipe.copy()
swipe_ev['student_id'] = swipe_ev['card_id'].map(card_map)
swipe_ev['event_type'] = 'swipe'
swipe_ev = swipe_ev.rename(columns={'location_id':'location'})

events = pd.concat([wifi_ev[['student_id','location','timestamp','event_type','device_hash']],
                    swipe_ev[['student_id','location','timestamp','event_type','card_id']]], ignore_index=True, sort=False)
events['timestamp'] = pd.to_datetime(events['timestamp'])
events = events.sort_values('timestamp').reset_index(drop=True)

# Link confidence heuristic
def linkage_score(row):
    score = 0.0
    if pd.notna(row.get('device_hash')) and row.get('device_hash') in dev_map: score += 0.6
    if pd.notna(row.get('card_id')) and row.get('card_id') in card_map: score += 0.6
    return min(score,1.0)
events['link_confidence'] = events.apply(linkage_score, axis=1)

st.subheader("Profiles (sample)")
st.dataframe(profiles.head(50))

st.subheader("Merged Events (recent)")
st.dataframe(events.sort_values('timestamp', ascending=False).head(200))

# Predictive Monitoring (ML)
st.header("Predictive Monitoring (RandomForest demo)")
df = events.copy().dropna(subset=['location']).sort_values(['student_id','timestamp']).reset_index(drop=True)
samples = []
for sid, grp in df.groupby('student_id'):
    grp = grp.sort_values('timestamp').reset_index(drop=True)
    for i in range(len(grp)-1):
        cur = grp.loc[i]
        nxt = grp.loc[i+1]
        samples.append({
            'student_id': sid,
            'cur_location': cur['location'],
            'hour': cur['timestamp'].hour,
            'dow': cur['timestamp'].dayofweek,
            'time_diff_minutes': (grp.loc[i+1,'timestamp'] - cur['timestamp']).total_seconds()/60.0,
            'label_next': nxt['location']
        })
sdf = pd.DataFrame(samples)
st.write(f"Generated {len(sdf)} training samples from event sequences.")

if len(sdf) >= 5:
    le_loc = LabelEncoder()
    le_loc.fit(pd.concat([sdf['cur_location'], sdf['label_next']]).astype(str))
    X = pd.DataFrame({
        'cur_loc_enc': le_loc.transform(sdf['cur_location']),
        'hour': sdf['hour'],
        'dow': sdf['dow'],
        'time_diff_min': sdf['time_diff_minutes']
    })
    y = le_loc.transform(sdf['label_next'])
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train,y_train)
    acc = clf.score(X_test,y_test)
    st.success(f"RandomForest test accuracy: {acc:.2f}")
    fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.subheader("Feature importances")
    st.table(fi)
    
    st.subheader("Per-entity prediction")
    sid_opt = st.selectbox("Choose entity for prediction", options=profiles['student_id'].tolist())
    if sid_opt:
        last = df[df['student_id']==sid_opt].sort_values('timestamp').tail(1)
        if last.empty:
            st.write("No history for this entity.")
        else:
            last = last.iloc[0]
            xq = pd.DataFrame({
                'cur_loc_enc':[le_loc.transform([last['location']])[0]],
                'hour':[last['timestamp'].hour],
                'dow':[last['timestamp'].dayofweek],
                'time_diff_min':[60.0]
            })
            probs = clf.predict_proba(xq)[0]
            top_idx = np.argsort(probs)[::-1][:3]
            top_locs = le_loc.inverse_transform(top_idx)
            for i, idx in enumerate(top_idx):
                st.write(f"{i+1}. {top_locs[i]} — {probs[idx]:.2f}")
            st.subheader("Approx feature contributions")
            contrib = fi * (xq.iloc[0].astype(float)/ (X.max().astype(float)+1e-9))
            st.table(contrib.sort_values(ascending=False))
else:
    st.warning("Not enough sequence samples to train the model.")

# Dashboard: search, timeline, heatmap, alerts
st.header("Dashboard")
search = st.text_input("Search by name or ID")
if search:
    matched = profiles[profiles['student_id'].str.contains(search, case=False) | profiles['name'].str.contains(search, case=False)]
else:
    matched = profiles
st.dataframe(matched.head(50))

if not matched.empty:
    sid = matched.iloc[0]['student_id']
    st.subheader(f"Timeline for {sid}")
    te = df[df['student_id']==sid].sort_values('timestamp')
    st.dataframe(te[['timestamp','location','event_type']].tail(100))
    if not te.empty:
        fig = px.scatter(te, x='timestamp', y='location')
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Activity Heatmap (hour vs location)")
heat = df.copy()
if not heat.empty:
    heat['hour'] = heat['timestamp'].dt.hour
    hm = heat.groupby(['hour','location']).size().unstack(fill_value=0)
    st.dataframe(hm)

st.subheader("Alerts (Inactivity)")
threshold = st.slider("Inactivity threshold (hours)", 1, 72, 6)
if not df.empty:
    last_seen = df.groupby('student_id')['timestamp'].max().reset_index()
    last_seen['inactive_hours'] = (pd.Timestamp.utcnow() - last_seen['timestamp']).dt.total_seconds()/3600.0
    alerts = last_seen[last_seen['inactive_hours'] > threshold].merge(profiles, on='student_id', how='left')
    st.dataframe(alerts[['student_id','name','timestamp','inactive_hours']])

# Multi-modal demo
st.header("Multi-modal demo (simulated)")
if not im_df.empty:
    st.write("Image embeddings sample:")
    st.dataframe(im_df.head())
if not notes.empty:
    st.write("Text notes sample:")
    st.dataframe(notes.head())

# Report generation
st.header("Generate a simple submission report (HTML)")
if st.button("Create HTML report"):
    rp = os.path.join(DATA_DIR, "submission_report.html")
    html = f"<html><body><h1>Campus Security AI — Submission Report</h1><p>Profiles: {len(profiles)}</p><p>Events: {len(events)}</p></body></html>"
    with open(rp,"w") as f:
        f.write(html)
    st.success(f"Report saved: {rp}")
    st.markdown(f"[Download report]({rp})")
