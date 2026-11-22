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
    filtered_df["region"] != "GLOBAL"
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
st.subheader("Incident Frequency vs Severity Exposure")

st.caption("""
This chart compares how often incidents occur in each region against
their overall severity exposure. Regions appearing higher on the chart
are associated with more severe incidents, while regions farther right
have experienced more incidents overall.
""")

risk_weights = {
"critical": 5,
"high": 4,
"medium": 3,
"low": 2,
"info": 1
}
 
scatter_df = filtered_df[
filtered_df["region"] != "Global"
].copy()
 
scatter_df["risk_score"] = (
scatter_df["severity"]
.astype(str)
.map(risk_weights)
)
 
region_summary = (
scatter_df
.groupby("region")
.agg(
incident_count=("region", "size"),
risk_score=("risk_score", "sum")
)
.reset_index()
)
 
fig = px.scatter(
region_summary,
x="incident_count",
y="risk_score",
size="risk_score",
color="risk_score",
text="region",
title="Regional Incident Frequency vs Severity Exposure",
labels={
"incident_count": "Incident Count",
"risk_score": "Severity Exposure Score",
"region": "Region"
}
)
 
fig.update_traces(
textposition="top center"
)
 
st.plotly_chart(
fig,
use_container_width=True
)
How to interpret it

# Landmark Events
st.subheader("Landmark Incidents")

st.caption(
    "Highlights notable incidents that influenced AI security practices, governance discussions, or public understanding of AI-agent risks."
)

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
