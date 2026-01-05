{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8de80c2d-4d1f-47df-8f55-3ba48383e843",
   "metadata": {},
   "source": [
    "## Import All Liberary "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c3493864-fedd-4fd6-9768-d81a4568895a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.ensemble import IsolationForest, RandomForestClassifier\n",
    "import matplotlib.pyplot as plt\n",
    "import plotly.express as px"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd36846e-4f29-4f2f-96f8-5d3d5ff69df7",
   "metadata": {},
   "source": [
    "## PAGE CONFIG"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1da89704-95df-4be5-b03f-cb83e673e39e",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.set_page_config(\n",
    "    page_title=\"AI Forecast & Risk Analytics\",\n",
    "    page_icon=\"📊\",\n",
    "    layout=\"wide\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a36e3e37-890b-4efe-830a-eb2cf4efd738",
   "metadata": {},
   "source": [
    "# CUSTOM CSS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b62a4b62-d52a-40d7-b9d5-f6d75c242b95",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-05 13:11:34.032 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.markdown(\"\"\"\n",
    "<style>\n",
    "body {\n",
    "    background-color: #f5f7fa;\n",
    "}\n",
    ".sidebar .sidebar-content {\n",
    "    background-color: #1f2937;\n",
    "}\n",
    "</style>\n",
    "\"\"\", unsafe_allow_html=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a713a8ef-6b01-4627-a5d9-34b23119b3d3",
   "metadata": {},
   "source": [
    "# Title"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "8affdce5-bf4e-43b5-9b7a-1301a788aacc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.title(\"DT AI Forecast, Anomaly & Fraud Risk Platform\")\n",
    "st.caption(\"Professional analytics platform built using Python & Streamlit\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "238df7a3-f129-4207-9ad7-983087254411",
   "metadata": {},
   "source": [
    "## SideBAr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "43376768-e889-4025-a410-a44c17195163",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-05 13:14:11.605 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "menu = st.sidebar.selectbox(\n",
    "    \"Select Module\",\n",
    "    (\n",
    "        \"📂 Data Import\",\n",
    "        \"📈 Forecast Model\",\n",
    "        \"🚨 Anomaly Detection\",\n",
    "        \"💳 Credit Fraud Risk\"\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6273cd38-783f-44c1-a212-de4df66500b3",
   "metadata": {},
   "source": [
    "## Session Date"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "14d8d1f2-ea8d-45f1-b06a-cbd9124e8767",
   "metadata": {},
   "outputs": [],
   "source": [
    "if \"data\" not in st.session_state:\n",
    "    st.session_state.data = None"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "41d06698-4106-4df8-a961-3b42b8c30f51",
   "metadata": {},
   "source": [
    "##  DATA IMPORT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "daa95847-e73a-42c5-86d9-3cdfe2a3bcc1",
   "metadata": {},
   "outputs": [],
   "source": [
    "if menu == \" Data Import\":\n",
    "    st.header(\" Upload & Explore Data\")\n",
    "\n",
    "    file = st.file_uploader(\"Upload CSV File\", type=[\"csv\"])\n",
    "\n",
    "    if file:\n",
    "        df = pd.read_csv(file)\n",
    "        st.session_state.data = df\n",
    "\n",
    "        st.success(\" Data Uploaded Successfully\")\n",
    "        st.subheader(\"Data Preview\")\n",
    "        st.dataframe(df.head())\n",
    "\n",
    "        st.subheader(\"Summary\")\n",
    "        st.write(df.describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20990384-dcc3-4213-8ac4-9cae18a97042",
   "metadata": {},
   "source": [
    "#  FORECAST MODEL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "32f39a60-5f3a-4e6e-83b7-91fa6c1e71ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "if menu == \" Forecast Model\":\n",
    "    st.header(\" Forecasting Module\")\n",
    "\n",
    "    if st.session_state.data is None:\n",
    "        st.warning(\" Please upload data first.\")\n",
    "    else:\n",
    "        df = st.session_state.data\n",
    "        numeric_cols = df.select_dtypes(include=np.number).columns\n",
    "\n",
    "        target = st.selectbox(\"Select Target Column\", numeric_cols)\n",
    "\n",
    "        window = st.slider(\"Moving Average Window\", 3, 30, 7)\n",
    "\n",
    "        df[\"Forecast\"] = df[target].rolling(window).mean()\n",
    "\n",
    "        fig = px.line(df, y=[target, \"Forecast\"], title=\"Forecast vs Actual\")\n",
    "        st.plotly_chart(fig, use_container_width=True)\n",
    "\n",
    "        st.success(\" Forecast Generated\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15bcb669-34ee-41f5-866e-09de7c3c7860",
   "metadata": {},
   "source": [
    "#  ANOMALY DETECTION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "58849915-4d15-46df-905e-21f075f2f26a",
   "metadata": {},
   "outputs": [],
   "source": [
    "if menu == \" Anomaly Detection\":\n",
    "    st.header(\" Anomaly Detection Module\")\n",
    "\n",
    "    if st.session_state.data is None:\n",
    "        st.warning(\" Please upload data first.\")\n",
    "    else:\n",
    "        df = st.session_state.data\n",
    "        numeric_cols = df.select_dtypes(include=np.number).columns\n",
    "\n",
    "        feature = st.selectbox(\"Select Feature\", numeric_cols)\n",
    "\n",
    "        model = IsolationForest(contamination=0.05, random_state=42)\n",
    "        df[\"Anomaly\"] = model.fit_predict(df[[feature]])\n",
    "        df[\"Anomaly\"] = df[\"Anomaly\"].map({1: \"Normal\", -1: \"Anomaly\"})\n",
    "\n",
    "        fig = px.scatter(\n",
    "            df,\n",
    "            y=feature,\n",
    "            color=\"Anomaly\",\n",
    "            title=\"Anomaly Detection Result\"\n",
    "        )\n",
    "        st.plotly_chart(fig, use_container_width=True)\n",
    "\n",
    "        st.subheader(\"Detected Anomalies\")\n",
    "        st.dataframe(df[df[\"Anomaly\"] == \"Anomaly\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f7a57998-f9d7-4608-846d-6065e2939c6f",
   "metadata": {},
   "source": [
    "#  CREDIT FRAUD RISK"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "590944b4-8c69-48a7-8354-74880796009c",
   "metadata": {},
   "outputs": [],
   "source": [
    "if menu == \" Credit Fraud Risk\":\n",
    "    st.header(\" Credit Fraud Risk Model\")\n",
    "\n",
    "    st.info(\"Demo model – Replace with real banking data\")\n",
    "\n",
    "    amount = st.number_input(\"Transaction Amount\", 0.0, 100000.0, 5000.0)\n",
    "    frequency = st.slider(\"Transaction Frequency\", 1, 50, 5)\n",
    "    location_risk = st.selectbox(\"Location Risk\", [0, 1])\n",
    "\n",
    "    if st.button(\"Predict Risk\"):\n",
    "        X_demo = np.array([[amount, frequency, location_risk]])\n",
    "        y_demo = np.array([0, 1, 0, 1])\n",
    "\n",
    "        model = RandomForestClassifier()\n",
    "        model.fit(\n",
    "            np.random.rand(4, 3),\n",
    "            y_demo\n",
    "        )\n",
    "\n",
    "        prob = model.predict_proba(X_demo)[0][1]\n",
    "\n",
    "        if prob > 0.6:\n",
    "            st.error(f\" High Fraud Risk: {prob:.2f}\")\n",
    "        else:\n",
    "            st.success(f\" Low Fraud Risk: {prob:.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "ba4bbb4f-48ca-48d7-ad09-cb2c5af074ed",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in /opt/anaconda3/lib/python3.12/site-packages (1.37.1)\n",
      "Requirement already satisfied: pandas in /opt/anaconda3/lib/python3.12/site-packages (2.2.2)\n",
      "Requirement already satisfied: numpy in /opt/anaconda3/lib/python3.12/site-packages (1.26.4)\n",
      "Requirement already satisfied: scikit-learn in /opt/anaconda3/lib/python3.12/site-packages (1.5.1)\n",
      "Requirement already satisfied: matplotlib in /opt/anaconda3/lib/python3.12/site-packages (3.9.2)\n",
      "Requirement already satisfied: plotly in /opt/anaconda3/lib/python3.12/site-packages (5.24.1)\n",
      "Requirement already satisfied: altair<6,>=4.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (5.0.1)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (1.6.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (5.3.3)\n",
      "Requirement already satisfied: click<9,>=7.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: packaging<25,>=20 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (24.1)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (10.4.0)\n",
      "Requirement already satisfied: protobuf<6,>=3.20 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (4.25.3)\n",
      "Requirement already satisfied: pyarrow>=7.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (16.1.0)\n",
      "Requirement already satisfied: requests<3,>=2.27 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (2.32.3)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (13.7.1)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (8.2.3)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (4.11.0)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (3.1.43)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (0.8.0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in /opt/anaconda3/lib/python3.12/site-packages (from streamlit) (6.4.1)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/lib/python3.12/site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: scipy>=1.6.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.13.1)\n",
      "Requirement already satisfied: joblib>=1.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.4.2)\n",
      "Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/anaconda3/lib/python3.12/site-packages (from scikit-learn) (3.5.0)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (1.2.0)\n",
      "Requirement already satisfied: cycler>=0.10 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (4.51.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (1.4.4)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in /opt/anaconda3/lib/python3.12/site-packages (from matplotlib) (3.1.2)\n",
      "Requirement already satisfied: jinja2 in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
      "Requirement already satisfied: jsonschema>=3.0 in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
      "Requirement already satisfied: toolz in /opt/anaconda3/lib/python3.12/site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.7)\n",
      "Requirement already satisfied: six>=1.5 in /opt/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2.2.3)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2025.1.31)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/anaconda3/lib/python3.12/site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: smmap<5,>=3.0.1 in /opt/anaconda3/lib/python3.12/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/lib/python3.12/site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in /opt/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: mdurl~=0.1 in /opt/anaconda3/lib/python3.12/site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit pandas numpy scikit-learn matplotlib plotly\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23054e8b-cd2d-4e5d-8b81-915d6459f174",
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
