{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "4f38936b-952f-433c-90d4-07e22e7ee836",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-05 16:40:53.358 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-01-05 16:40:53.359 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.ensemble import IsolationForest, RandomForestClassifier\n",
    "import plotly.express as px\n",
    "\n",
    "st.set_page_config(page_title=\"AI Analytics Platform\", layout=\"wide\")\n",
    "\n",
    "st.title(\"📊 AI Forecast, Anomaly & Fraud Platform\")\n",
    "\n",
    "menu = st.sidebar.selectbox(\n",
    "    \"Select Module\",\n",
    "    [\n",
    "        \"Data Import\",\n",
    "        \"Forecast\",\n",
    "        \"Anomaly Detection\",\n",
    "        \"Credit Fraud Risk\"\n",
    "    ]\n",
    ")\n",
    "\n",
    "if \"data\" not in st.session_state:\n",
    "    st.session_state.data = None\n",
    "\n",
    "# ---------------- DATA IMPORT ----------------\n",
    "if menu == \"Data Import\":\n",
    "    file = st.file_uploader(\"Upload CSV\", type=[\"csv\"])\n",
    "    if file:\n",
    "        df = pd.read_csv(file)\n",
    "        st.session_state.data = df\n",
    "        st.dataframe(df.head())\n",
    "\n",
    "# ---------------- FORECAST ----------------\n",
    "elif menu == \"Forecast\":\n",
    "    if st.session_state.data is None:\n",
    "        st.warning(\"Upload data first\")\n",
    "    else:\n",
    "        df = st.session_state.data\n",
    "        col = st.selectbox(\"Select column\", df.select_dtypes(include=np.number).columns)\n",
    "        df[\"Forecast\"] = df[col].rolling(5).mean()\n",
    "        fig = px.line(df, y=[col, \"Forecast\"])\n",
    "        st.plotly_chart(fig, use_container_width=True)\n",
    "\n",
    "# ---------------- ANOMALY ----------------\n",
    "elif menu == \"Anomaly Detection\":\n",
    "    if st.session_state.data is None:\n",
    "        st.warning(\"Upload data first\")\n",
    "    else:\n",
    "        df = st.session_state.data\n",
    "        col = st.selectbox(\"Select column\", df.select_dtypes(include=np.number).columns)\n",
    "        model = IsolationForest(contamination=0.05, random_state=42)\n",
    "        df[\"Anomaly\"] = model.fit_predict(df[[col]])\n",
    "        fig = px.scatter(df, y=col, color=df[\"Anomaly\"].astype(str))\n",
    "        st.plotly_chart(fig, use_container_width=True)\n",
    "\n",
    "# ---------------- FRAUD ----------------\n",
    "elif menu == \"Credit Fraud Risk\":\n",
    "    amount = st.number_input(\"Transaction Amount\", 0.0, 100000.0, 5000.0)\n",
    "    freq = st.slider(\"Transaction Frequency\", 1, 50, 5)\n",
    "    risk = st.selectbox(\"Location Risk\", [0, 1])\n",
    "\n",
    "    if st.button(\"Predict\"):\n",
    "        X = np.array([[amount, freq, risk]])\n",
    "        y = np.array([0, 1, 0, 1])\n",
    "        model = RandomForestClassifier()\n",
    "        model.fit(np.random.rand(4, 3), y)\n",
    "        prob = model.predict_proba(X)[0][1]\n",
    "\n",
    "        if prob > 0.6:\n",
    "            st.error(f\"🚨 High Fraud Risk: {prob:.2f}\")\n",
    "        else:\n",
    "            st.success(f\"✅ Low Fraud Risk: {prob:.2f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cff2bc71-76d9-4d2b-a858-d98890c504ed",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
