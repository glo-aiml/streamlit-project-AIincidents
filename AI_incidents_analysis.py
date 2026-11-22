import streamlit as st
import pandas as pd
import plotly.express as px
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Load Dataset
file_path2 = "ai_agent_incidents_clean.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "muhammadahmerkhan/ai-agent-security-and-real-world-incidents-2024-26",
    file_path2
)

# Preprocessing
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

severity_order = [
    "critical",
    "high",
    "medium",
    "low",
    "info"
]

df["severity"] = pd.Categorical(
    df["severity"],
    categories=severity_order,
    ordered=True
)

# Sidebar Filters
st.sidebar.title("Filters")

selected_region = st.sidebar.multiselect(
    "Region",
    sorted(df["region"].dropna().unique()),
    default=sorted(df["region"].dropna().unique())
)

selected_severity = st.sidebar.multiselect(
    "Severity",
    severity_order,
    default=severity_order
)

selected_years = st.sidebar.multiselect(
    "Year",
    sorted(df["year"].unique()),
    default=sorted(df["year"].unique())
)

filtered_df = df[
    (df["region"].isin(selected_region))
    & (df["severity"].isin(selected_severity))
    & (df["year"].isin(selected_years))
]

# Dashboard Header
st.title("AI Agent Incident Intelligence Dashboard")

st.markdown("""
Analyze trends in AI agent security incidents, severity levels,
regional exposure, and attack techniques.
""")

# Overview Metrics
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Incidents",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Critical Incidents",
        len(
            filtered_df[
                filtered_df["severity"] == "critical"
            ]
        )
    )

with col3:
    st.metric(
        "Regions Impacted",
        filtered_df["region"].nunique()
    )

with col4:
    st.metric(
        "Attack Types",
        filtered_df["attack_type_name"].nunique()
    )

# Severity Breakdown
st.subheader("Severity Distribution")

severity_counts = (
    filtered_df["severity"]
    .value_counts()
    .reindex(severity_order)
)

fig = px.pie(
    values=severity_counts.values,
    names=severity_counts.index,
    title="Incident Severity Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# Trend Analysis
st.subheader("Incident Trends Over Time")

monthly_counts = (
    filtered_df
    .groupby("month")
    .size()
    .reset_index(name="incidents")
)

fig = px.line(
    monthly_counts,
    x="month",
    y="incidents",
    markers=True,
    title="AI Agent Incidents Over Time"
)

st.plotly_chart(fig, use_container_width=True)

# Attack Type Frequency
st.subheader("Most Common Attack Types")

attack_labels = {
    "IPI;CRED": "Prompt Injection + Credential Theft",
    "INFRA;CRED": "Infrastructure Compromise + Credential Theft",
    "IPI;EXFIL": "Prompt Injection + Data Exfiltration",
    "SUPPLY;CRED": "Supply Chain Attack + Credential Theft",
    "Indirect Prompt Injection (IPI)": "Indirect Prompt Injection",
    "AI Infrastructure & Cluster Compromise": "Infrastructure Compromise",
    "AI Governance & Policy Failure": "Governance Failure",
    "AI Weaponization & Malicious Tooling": "Weaponziation & Malicious AI Use",
    "Credential & Secret Theft": "Credential Theft",
    "Model Context Protocol (MCP) Exploit": "MCP Exploit",
    "Autonomous Rogue Agent Behavior": "Rogue Agent Behavior",
    "Benchmark & Evaluation Poisoning": "Evaluation Poisoning",
    "Sandbox & Environment Escape": "Sandbox Escape",
    "AI Supply Chain & Dependency Attack": "Supply Chain Attack",
    "Unclassified AI Incident": "Unclassified Incident"
}

attack_counts = (
    filtered_df["attack_type_name"]
    .value_counts()
    .reset_index()
)

attack_counts.columns = [
    "Attack Type",
    "Count"
]

attack_counts["Attack Type"] = (
    attack_counts["Attack Type"]
    .replace(attack_labels)
)

fig = px.bar(
    attack_counts.head(15),
    x="Count",
    y="Attack Type",
    orientation="h",
    title="Top Attack Types"
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Attack Type Glossary"):
    st.markdown("""
    - **Prompt Injection**: Manipulating an AI system through crafted instructions.
    - **Indirect Prompt Injection**: Malicious instructions delivered through external content.
    - **Credential Theft**: Theft of passwords, API keys, or access tokens.
    - **Data Exfiltration**: Unauthorized extraction of sensitive information.
    - **MCP Exploit**: Abuse of the Model Context Protocol to access tools or data.
    - **Sandbox Escape**: Escaping an isolated execution environment.
    - **Supply Chain Attack**: Compromise through dependencies, plugins, or third-party services.
    - **Infrastructure Compromise**: Attack against underlying AI platforms, servers, or clusters.
    - **Rogue Agent Behavior**: Unintended or unauthorized autonomous actions by AI agents.
    - **Governance Failure**: Policy, oversight, or safety control failures.
    """)

# Region Exposure
st.subheader("AI Agent Incident Exposure by Region")

region_df = filtered_df[
    filtered_df["region"] != "Global"
].copy()

region_counts = (
    region_df["region"]
    .value_counts()
    .reset_index()
)

region_counts.columns = [
    "Region",
    "Incidents"
]

fig = px.bar(
    region_counts,
    x="Region",
    y="Incidents",
    color="Incidents",
    title="Incidents by Region"
)

st.plotly_chart(fig, use_container_width=True)
# Risk Score by Region
st.subheader("Risk Score by Region")

risk_weights = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "low": 2,
    "info": 1
}

risk_df = filtered_df[
    filtered_df["region"] != "Global"
].copy()

risk_df["risk_score"] = risk_df["severity"].astype(str).map(risk_weights)

region_risk = (
    risk_df
    .groupby("region", as_index=False)["risk_score"]
    .sum()
)

fig = px.bar(
    region_risk.sort_values(
        "risk_score",
        ascending=False
    ),
    x="region",
    y="risk_score",
    color="risk_score",
    title="Regional Risk Score"
)

st.plotly_chart(fig, use_container_width=True)

# Landmark Events
st.subheader("Landmark Incidents")

landmarks = filtered_df[
    filtered_df["is_landmark_event"] == True
]

st.dataframe(
    landmarks[
        [
            "date",
            "title",
            "severity",
            "region"
        ]
    ],
    use_container_width=True
)

# Raw Cleaned Data

st.subheader("Explore Incident Data")

st.dataframe(
    filtered_df,
    use_container_width=True
)
