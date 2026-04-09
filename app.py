import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="COVID-19 Social Media Analytics",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E2130;
        border-radius: 8px;
        padding: 8px 20px;
        color: #FAFAFA;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2A9D8F !important;
        color: white !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E2130, #2A2D3E);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2A9D8F33;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #2A9D8F; }
    .metric-label { font-size: 0.85rem; color: #888; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2A9D8F;
        border-left: 3px solid #2A9D8F;
        padding-left: 10px;
        margin: 20px 0 12px 0;
    }
    h1 { color: #FAFAFA !important; }
    h2, h3 { color: #E0E0E0 !important; }
    .finding-box {
        background: #1E2130;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #2A9D8F;
        margin: 8px 0;
        color: #CCCCCC;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("# 🦠 COVID-19 Social Media Analytics Dashboard")
st.markdown("##### Bhawna Patnaik & Aditya Bhilare · MSc Business Analytics · MS5131")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Sentiment & Model Comparison",
    "🔍 ABSA & Emotion Analysis",
    "🗺️ Topic Evolution"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — SENTIMENT & MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
with tab1:

    # Dataset summary cards
    st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("376,966", "Total Tweets Analysed"),
        ("411,887", "Raw Tweets Collected"),
        ("3", "Time Periods"),
        ("5", "Analytical Methods"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Dataset breakdown table
    st.markdown('<div class="section-header">Dataset Breakdown by Period</div>', unsafe_allow_html=True)
    dataset_df = pd.DataFrame({
        'Period': ['Apr-Jun 2020', 'Aug-Sep 2020', 'Apr-Jun 2021', 'Total'],
        'Raw Tweets': ['143,903', '120,509', '147,475', '411,887'],
        'Cleaned Tweets': ['135,723', '111,326', '129,917', '376,966'],
        'Removed': ['8,180', '9,183', '17,558', '34,921'],
        'Analytical Scope': [
            'Sentiment + ABSA + LDA + BERTopic',
            'ABSA + BERTopic + RoBERTa',
            'ABSA + BERTopic',
            ''
        ]
    })
    st.dataframe(dataset_df, use_container_width=True, hide_index=True)

    st.markdown("")
    col_left, col_right = st.columns(2)

    # Sentiment distribution
    with col_left:
        st.markdown('<div class="section-header">Sentiment Distribution — Apr-Jun 2020</div>', unsafe_allow_html=True)
        sent_df = pd.DataFrame({
            'Sentiment': ['Neutral', 'Positive', 'Negative'],
            'Count': [53083, 44427, 38213],
            'Percentage': [39.1, 32.7, 28.2]
        })
        fig_sent = px.bar(
            sent_df, x='Sentiment', y='Count',
            color='Sentiment',
            color_discrete_map={
                'Negative': '#E63946',
                'Neutral': '#6C757D',
                'Positive': '#2A9D8F'
            },
            text=sent_df['Percentage'].apply(lambda x: f'{x}%'),
            template='plotly_dark'
        )
        fig_sent.update_traces(textposition='outside', textfont_size=13)
        fig_sent.update_layout(
            showlegend=False,
            plot_bgcolor='#1E2130',
            paper_bgcolor='#1E2130',
            margin=dict(t=20, b=20),
            height=320
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    # Model comparison
    with col_right:
        st.markdown('<div class="section-header">Sentiment Model Performance Comparison</div>', unsafe_allow_html=True)
        model_df = pd.DataFrame({
            'Model': ['VADER', 'SVM + TF-IDF', 'BERT'],
            'Accuracy': [98.44, 97.55, 98.88],
            'Macro F1': [0.98, 0.97, 0.99],
            'Macro Precision': [0.99, 0.97, 0.99],
            'Macro Recall': [0.98, 0.97, 0.99]
        })
        fig_model = go.Figure()
        metrics = ['Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall']
        colors = ['#2A9D8F', '#E9C46A', '#E63946']
        for i, (_, row) in enumerate(model_df.iterrows()):
            vals = [row['Accuracy']/100, row['Macro F1'], row['Macro Precision'], row['Macro Recall']]
            fig_model.add_trace(go.Bar(
                name=row['Model'],
                x=metrics,
                y=vals,
                marker_color=colors[i],
                text=[f'{v:.2f}' for v in vals],
                textposition='outside'
            ))
        fig_model.update_layout(
            barmode='group',
            template='plotly_dark',
            plot_bgcolor='#1E2130',
            paper_bgcolor='#1E2130',
            legend=dict(bgcolor='#1E2130'),
            margin=dict(t=20, b=20),
            height=320,
            yaxis=dict(range=[0.94, 1.02])
        )
        st.plotly_chart(fig_model, use_container_width=True)

    # Key findings
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('<div class="finding-box">⚠️ <strong>Label Leakage:</strong> 98.4% agreement between fresh VADER predictions and dataset labels confirms labels are VADER-derived — accuracy reflects VADER approximation, not human-validated sentiment.</div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="finding-box">🏆 <strong>BERT Best Performer:</strong> 98.88% accuracy and macro F1 of 0.99, but only 1.33 percentage points above VADER — raising questions about the value of expensive transformer fine-tuning.</div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="finding-box">📊 <strong>Neutral Dominates:</strong> 39.1% neutral sentiment confirms Twitter functioned primarily as an information-sharing platform during COVID-19 rather than an emotional outlet.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — ABSA & EMOTION ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab2:

    st.markdown('<div class="section-header">Aspect-Based Sentiment Analysis — Temporal Trajectory</div>', unsafe_allow_html=True)

    absa_data = {
        'Aspect': ['Lockdown & Quarantine', 'Vaccines & Treatment', 'Government Response',
                   'Healthcare System', 'Economy & Jobs', 'Mental Health', 'Education', 'Misinformation'],
        'Apr-Jun 2020': [-0.051, 0.022, -0.054, -0.013, -0.000, -0.280, 0.063, -0.338],
        'Aug-Sep 2020': [-0.140, 0.022, -0.045, -0.018, 0.024, -0.273, 0.030, -0.299],
        'Apr-Jun 2021': [-0.134, 0.050, -0.015, -0.009, 0.054, -0.270, 0.043, -0.288]
    }
    absa_df = pd.DataFrame(absa_data)

    # Heatmap
    heat_data = absa_df.set_index('Aspect')[['Apr-Jun 2020', 'Aug-Sep 2020', 'Apr-Jun 2021']]
    fig_heat = px.imshow(
        heat_data,
        color_continuous_scale='RdYlGn',
        zmin=-0.4, zmax=0.4,
        text_auto='.3f',
        template='plotly_dark',
        aspect='auto'
    )
    fig_heat.update_layout(
        plot_bgcolor='#1E2130',
        paper_bgcolor='#1E2130',
        margin=dict(t=20, b=20),
        height=380,
        coloraxis_colorbar=dict(title='Avg Compound Score')
    )
    fig_heat.update_traces(textfont_size=12)
    st.plotly_chart(fig_heat, use_container_width=True)

    col_l, col_r = st.columns(2)

    # Trajectory line chart
    with col_l:
        st.markdown('<div class="section-header">Sentiment Trajectory by Aspect</div>', unsafe_allow_html=True)
        periods = ['Apr-Jun 2020', 'Aug-Sep 2020', 'Apr-Jun 2021']
        highlight = {
            'Lockdown & Quarantine': '#E63946',
            'Vaccines & Treatment': '#2A9D8F',
            'Mental Health': '#9B59B6',
            'Misinformation': '#E67E22',
            'Education': '#27AE60',
            'Government Response': '#3498DB'
        }
        fig_traj = go.Figure()
        for _, row in absa_df.iterrows():
            aspect = row['Aspect']
            if aspect in highlight:
                fig_traj.add_trace(go.Scatter(
                    x=periods,
                    y=[row['Apr-Jun 2020'], row['Aug-Sep 2020'], row['Apr-Jun 2021']],
                    mode='lines+markers',
                    name=aspect,
                    line=dict(color=highlight[aspect], width=2.5),
                    marker=dict(size=8)
                ))
        fig_traj.add_hline(y=0.05, line_dash='dash', line_color='#2A9D8F',
                            annotation_text='Positive threshold', opacity=0.5)
        fig_traj.add_hline(y=-0.05, line_dash='dash', line_color='#E63946',
                            annotation_text='Negative threshold', opacity=0.5)
        fig_traj.add_hline(y=0, line_color='white', opacity=0.2)
        fig_traj.update_layout(
            template='plotly_dark',
            plot_bgcolor='#1E2130',
            paper_bgcolor='#1E2130',
            legend=dict(bgcolor='#1E2130', font=dict(size=10)),
            margin=dict(t=20, b=20),
            height=350,
            yaxis_title='Avg Compound Score'
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    # RoBERTa emotion
    with col_r:
        st.markdown('<div class="section-header">RoBERTa Emotion Analysis — Aug-Sep 2020</div>', unsafe_allow_html=True)
        emotion_df = pd.DataFrame({
            'Emotion': ['Sadness', 'Optimism', 'Joy', 'Anger'],
            'Count': [6030, 4536, 3115, 1319],
            'Percentage': [40.2, 30.2, 20.8, 8.8]
        })
        fig_emo = px.bar(
            emotion_df, x='Emotion', y='Count',
            color='Emotion',
            color_discrete_map={
                'Sadness': '#457B9D',
                'Optimism': '#2A9D8F',
                'Joy': '#E9C46A',
                'Anger': '#E63946'
            },
            text=emotion_df['Percentage'].apply(lambda x: f'{x}%'),
            template='plotly_dark'
        )
        fig_emo.update_traces(textposition='outside', textfont_size=13)
        fig_emo.update_layout(
            showlegend=False,
            plot_bgcolor='#1E2130',
            paper_bgcolor='#1E2130',
            margin=dict(t=20, b=20),
            height=250
        )
        st.plotly_chart(fig_emo, use_container_width=True)

        # Irony gauge
        st.markdown('<div class="section-header">Sarcasm/Irony Detection Rate</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=20.4,
            delta={'reference': 10, 'valueformat': '.1f', 'suffix': '%'},
            number={'suffix': '%', 'font': {'size': 36, 'color': '#E63946'}},
            gauge={
                'axis': {'range': [0, 40], 'tickcolor': 'white'},
                'bar': {'color': '#E63946'},
                'bgcolor': '#1E2130',
                'steps': [
                    {'range': [0, 10], 'color': '#2A9D8F33'},
                    {'range': [10, 25], 'color': '#E9C46A33'},
                    {'range': [25, 40], 'color': '#E6394633'}
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 2},
                    'thickness': 0.75,
                    'value': 20.4
                }
            },
            title={'text': "Irony Rate (Sample of 15,000 tweets)", 'font': {'color': '#888', 'size': 12}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor='#1E2130',
            font={'color': 'white'},
            height=200,
            margin=dict(t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ABSA key findings
    st.markdown('<div class="section-header">Key ABSA Findings</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    findings = [
        ("😰", "Mental Health", "Most negative across ALL periods (-0.28)", "#9B59B6"),
        ("🦠", "Misinformation", "Most negative overall (-0.34 → -0.29)", "#E67E22"),
        ("💉", "Vaccines", "Steadily improving (+0.02 → +0.05)", "#2A9D8F"),
        ("🔒", "Lockdown", "Got more negative (-0.05 → -0.14)", "#E63946"),
    ]
    for col, (icon, title, desc, color) in zip([a1, a2, a3, a4], findings):
        with col:
            st.markdown(f"""
            <div style="background:#1E2130; border-radius:10px; padding:16px;
                        border-top: 3px solid {color}; text-align:center;">
                <div style="font-size:1.8rem">{icon}</div>
                <div style="font-weight:700; color:{color}; margin:6px 0">{title}</div>
                <div style="font-size:0.82rem; color:#AAA">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — TOPIC EVOLUTION
# ══════════════════════════════════════════════════════════════════
with tab3:

    st.markdown('<div class="section-header">BERTopic — Topic Distribution Across Three Periods</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    topics_2020_apr = {
        'Healthcare': 1566, 'Masks': 1563, 'Reopening': 1379,
        'Food Insecurity': 1317, 'Education': 871, 'Mental Health': 680,
        'Protests': 667, 'Treatment/Drugs': 643, 'Lockdown Policy': 597, 'Testing': 489
    }
    topics_2020_aug = {
        'Testing': 4899, 'Vaccine Trials': 3743, 'Masks': 2667,
        'Healthcare': 1726, 'Case Counts': 1497, 'Politics/Election': 1121,
        'China/Origins': 876, 'Food Insecurity': 853, 'Contact Tracing': 824,
        'Treatment/Drugs': 740
    }
    topics_2021_apr = {
        'Vaccine Discourse': 28000, 'Vaccine Access': 734,
        'Lockdown Policy': 384, 'Prevention': 251,
        'UK Govt/Policy': 191, 'Politics/Election': 191,
        'Protests': 205, 'Treatment/Drugs': 170
    }

    def make_topic_chart(topics, title, color):
        df = pd.DataFrame(list(topics.items()), columns=['Topic', 'Tweets'])
        df = df.sort_values('Tweets', ascending=True)
        fig = px.bar(df, x='Tweets', y='Topic', orientation='h',
                     color_discrete_sequence=[color],
                     template='plotly_dark',
                     text='Tweets')
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(
            title=dict(text=title, font=dict(size=13, color='white')),
            plot_bgcolor='#1E2130',
            paper_bgcolor='#1E2130',
            showlegend=False,
            margin=dict(t=40, b=10, l=10, r=60),
            height=380,
            xaxis_title='Tweet Count',
            yaxis_title=''
        )
        return fig

    with col1:
        st.plotly_chart(make_topic_chart(topics_2020_apr, 'Apr-Jun 2020', '#4ECDC4'),
                        use_container_width=True)
        st.markdown('<div style="text-align:center;font-size:0.8rem;color:#888;font-style:italic">Early pandemic — broad fragmented discourse</div>', unsafe_allow_html=True)

    with col2:
        st.plotly_chart(make_topic_chart(topics_2020_aug, 'Aug-Sep 2020', '#E9C46A'),
                        use_container_width=True)
        st.markdown('<div style="text-align:center;font-size:0.8rem;color:#888;font-style:italic">Mid pandemic — testing, politics, vaccine trials emerge</div>', unsafe_allow_html=True)

    with col3:
        st.plotly_chart(make_topic_chart(topics_2021_apr, 'Apr-Jun 2021', '#FF6B6B'),
                        use_container_width=True)
        st.markdown('<div style="text-align:center;font-size:0.8rem;color:#888;font-style:italic">Late period — vaccine consolidates all discourse</div>', unsafe_allow_html=True)

    # LDA vs BERTopic
    st.markdown('<div class="section-header">LDA vs BERTopic — Method Comparison</div>', unsafe_allow_html=True)

    lda_col, bert_col = st.columns(2)

    with lda_col:
        st.markdown("**LDA Results — Apr-Jun 2020**")
        lda_metrics = pd.DataFrame({
            'Metric': ['Topics Found', 'Avg Topic Size', 'Topic Diversity',
                       'Perplexity', 'Avg Coherence', 'Outlier %'],
            'Value': ['8 (fixed)', '~16,965 tweets', '0.863',
                      '3,850.93', '-137.76', '0%']
        })
        st.dataframe(lda_metrics, use_container_width=True, hide_index=True)

    with bert_col:
        st.markdown("**BERTopic Results — Apr-Jun 2020**")
        bert_metrics = pd.DataFrame({
            'Metric': ['Topics Found', 'Avg Topic Size', 'Topic Diversity',
                       'Perplexity', 'Avg Coherence', 'Outlier %'],
            'Value': ['14 (automatic)', '1,097 tweets', '0.775',
                      'N/A', 'N/A (embedding-based)', '53.9%']
        })
        st.dataframe(bert_metrics, use_container_width=True, hide_index=True)

    # Three phases narrative
    st.markdown('<div class="section-header">The Three-Phase COVID-19 Discourse Evolution</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    phases = [
        ("🌱", "Apr-Jun 2020", "Early Pandemic",
         "Fragmented discourse across 14 topics — masks, education, mental health, food insecurity, religion. Broad public uncertainty with no dominant narrative.",
         "#4ECDC4"),
        ("⚡", "Aug-Sep 2020", "Mid Pandemic",
         "Discourse sharpened around specific developments — vaccine trials, US election, contact tracing apps, Remdesivir. A more informed public engaging with concrete pandemic responses.",
         "#E9C46A"),
        ("💉", "Apr-Jun 2021", "Late Period / Vaccine Rollout",
         "Vaccine discourse became overwhelmingly dominant. Previously distinct topics (masks, food insecurity) disappeared entirely — collective attention consolidated around vaccination.",
         "#FF6B6B"),
    ]
    for col, (icon, period, phase, desc, color) in zip([p1, p2, p3], phases):
        with col:
            st.markdown(f"""
            <div style="background:#1E2130; border-radius:12px; padding:20px;
                        border-top: 4px solid {color}; height: 220px;">
                <div style="font-size:2rem; text-align:center">{icon}</div>
                <div style="font-weight:700; color:{color}; font-size:0.95rem;
                             text-align:center; margin:8px 0">{period}<br>
                     <span style="color:#888; font-size:0.8rem; font-weight:400">{phase}</span>
                </div>
                <div style="font-size:0.82rem; color:#AAA; line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    <div style="background:#1E2130; border-radius:10px; padding:16px 20px;
                border-left: 4px solid #2A9D8F; margin-top:10px;">
        <span style="color:#2A9D8F; font-weight:700">Key Finding: </span>
        <span style="color:#CCCCCC; font-size:0.9rem">
        This temporal evolution — from fragmented early pandemic uncertainty to focused vaccine discourse —
        provides empirical evidence of how collective public attention consolidates around dominant narratives
        during extended crises. BERTopic (diversity: 0.775) captured finer semantic granularity than LDA
        (diversity: 0.863, 8 fixed topics), supporting its use as the primary temporal analysis tool.
        </span>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555; font-size:0.8rem; padding:10px">
    Social Media Analytics of COVID-19 Discourse · MS5131 Major Business Analytics Project ·
    Bhawna Patnaik & Aditya Bhilare · 1MBY1 MSc Business Analytics
</div>""", unsafe_allow_html=True)
