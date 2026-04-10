import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import streamlit.components.v1 as components

st.set_page_config(
    page_title="st.markdown("# 🦠 COVID-19 Social Media Analytics VERSION 2")s",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
  /* ── Global ── */
  /* ── Remove Streamlit top padding ── */
  .main .block-container { padding-top: 0.5rem !important; }
  header[data-testid="stHeader"] { display: none !important; }
  section[data-testid="stSidebar"] { display: none; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
  .stTabs [data-baseweb="tab"] {
    background: #1E2130; border-radius: 8px;
    padding: 6px 16px; color: #FAFAFA; font-weight: 600; font-size: 0.85rem;
  }
  .stTabs [aria-selected="true"] { background: #2A9D8F !important; color: white !important; }

  /* ── Cards ── */
  .metric-card {
    background: linear-gradient(135deg,#1E2130,#2A2D3E);
    border-radius: 12px; padding: 16px 10px;
    border: 1px solid #2A9D8F33; text-align: center;
  }
  .metric-value { font-size: clamp(1.3rem,3vw,2rem); font-weight:700; color:#2A9D8F; }
  .metric-label { font-size: clamp(0.7rem,1.5vw,0.85rem); color:#888; margin-top:4px; }

  /* ── Section headers ── */
  .section-header {
    font-size: clamp(0.85rem,1.8vw,1.05rem); font-weight:600; color:#2A9D8F;
    border-left:3px solid #2A9D8F; padding-left:10px; margin:18px 0 10px;
  }

  /* ── Finding boxes ── */
  .finding-box {
    background:#1E2130; border-radius:8px; padding:12px 14px;
    border-left:4px solid #2A9D8F; margin:6px 0;
    color:#CCCCCC; font-size:clamp(0.78rem,1.5vw,0.9rem); line-height:1.5;
  }

  /* ── Responsive tables ── */
  .stDataFrame { font-size: clamp(0.7rem,1.5vw,0.9rem) !important; }

  /* ── Radio buttons ── */
  .stRadio label { font-size: clamp(0.78rem,1.5vw,0.9rem) !important; }

  /* ── Plotly charts responsive ── */
  .js-plotly-plot { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🦠 COVID-19 Social Media Analytics")
st.markdown("##### Bhawna Patnaik & Aditya Bhilare · MSc Business Analytics · MS5131")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sentiment & Models",
    "🔍 ABSA & Emotions",
    "🗺️ Topic Evolution",
    "☁️ Word Clouds"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,(val,label) in zip([c1,c2,c3,c4],[
        ("376,966","Total Tweets"),("411,887","Raw Collected"),
        ("3","Time Periods"),("5","Methods Used")
    ]):
        with col:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">Dataset Breakdown</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Period":["Apr-Jun 2020","Aug-Sep 2020","Apr-Jun 2021","Total"],
        "Raw":["143,903","120,509","147,475","411,887"],
        "Cleaned":["135,723","111,326","129,917","376,966"],
        "Removed":["8,180","9,183","17,558","34,921"],
        "Scope":["Sentiment+ABSA+LDA+BERTopic","ABSA+BERTopic+RoBERTa","ABSA+BERTopic",""]
    }), use_container_width=True, hide_index=True)

    st.markdown("")
    col_l, col_r = st.columns([1,1])

    with col_l:
        st.markdown('<div class="section-header">Sentiment Distribution — Apr-Jun 2020</div>', unsafe_allow_html=True)
        fig = px.bar(pd.DataFrame({
            "Sentiment":["Negative","Neutral","Positive"],
            "Count":[38213,53083,44427],
            "Pct":["28.2%","39.1%","32.7%"]
        }), x="Sentiment", y="Count", color="Sentiment",
            color_discrete_map={"Negative":"#E63946","Neutral":"#6C757D","Positive":"#2A9D8F"},
            text="Pct", template="plotly_dark")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="#1E2130",
                          paper_bgcolor="#1E2130", margin=dict(t=10,b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        models = ["VADER","SVM+TF-IDF","BERT"]
        accs   = [98.44, 97.55, 98.88]
        f1s    = [0.98,  0.97,  0.99 ]
        colors = ["#2A9D8F","#E9C46A","#E63946"]
        for m,a,f,c in zip(models,accs,f1s,colors):
            fig2.add_trace(go.Bar(name=m, x=["Accuracy","Macro F1"],
                y=[a/100, f], marker_color=c,
                text=[f"{a}%",str(f)], textposition="outside"))
        fig2.update_layout(barmode="group", template="plotly_dark",
            plot_bgcolor="#1E2130", paper_bgcolor="#1E2130",
            yaxis=dict(range=[0.93,1.02]), height=300,
            margin=dict(t=10,b=10), legend=dict(bgcolor="#1E2130"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    f1c,f2c,f3c = st.columns(3)
    findings = [
        ("⚠️","Label Leakage","98.4% agreement confirms VADER-derived labels — accuracy reflects VADER approximation, not human-validated sentiment."),
        ("🏆","BERT Best","98.88% accuracy — only 1.33pp above VADER, questioning the value of expensive transformer fine-tuning."),
        ("📊","Neutral Dominates","39.1% neutral confirms Twitter as information-sharing platform, not primarily emotional outlet."),
    ]
    for col,(icon,title,desc) in zip([f1c,f2c,f3c],findings):
        with col:
            st.markdown(f'''<div class="finding-box">
                {icon} <b>{title}:</b> {desc}</div>''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">ABSA Temporal Heatmap — Click cells to explore</div>', unsafe_allow_html=True)

    absa_data = {
        'Aspect':['Lockdown & Quarantine','Vaccines & Treatment','Government Response',
                  'Healthcare System','Economy & Jobs','Mental Health','Education','Misinformation'],
        'Apr-Jun 2020':[-0.051,0.022,-0.054,-0.013,-0.000,-0.280,0.063,-0.338],
        'Aug-Sep 2020':[-0.140,0.022,-0.045,-0.018,0.024,-0.273,0.030,-0.299],
        'Apr-Jun 2021':[-0.134,0.050,-0.015,-0.009,0.054,-0.270,0.043,-0.288]
    }
    absa_df = pd.DataFrame(absa_data)
    heat_data = absa_df.set_index('Aspect')[['Apr-Jun 2020','Aug-Sep 2020','Apr-Jun 2021']]

    fig_heat = px.imshow(
        heat_data, color_continuous_scale='RdYlGn',
        zmin=-0.4, zmax=0.4, text_auto='.3f',
        template='plotly_dark', aspect='auto'
    )
    fig_heat.update_layout(
        plot_bgcolor='#1E2130', paper_bgcolor='#1E2130',
        margin=dict(t=20,b=20),
            height=560,
        coloraxis_colorbar=dict(title='Avg Compound Score'),
        hoverlabel=dict(bgcolor='#1E2130', font_size=13)
    )
    fig_heat.update_traces(
        hovertemplate='<b>%{y}</b><br>Period: %{x}<br>Score: %{z:.3f}<extra></extra>',
        textfont_size=12
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">3D Sentiment Trajectory</div>', unsafe_allow_html=True)
        periods_idx = [0, 1, 2]
        period_labels = ['Apr-Jun 2020','Aug-Sep 2020','Apr-Jun 2021']
        colors_3d = ['#E63946','#2A9D8F','#9B59B6','#3498DB','#27AE60','#E74C3C','#F39C12','#E67E22']

        fig_3d = go.Figure()
        for i, row in absa_df.iterrows():
            vals = [row['Apr-Jun 2020'], row['Aug-Sep 2020'], row['Apr-Jun 2021']]
            fig_3d.add_trace(go.Scatter3d(
                x=periods_idx,
                y=[i]*3,
                z=vals,
                mode='lines+markers',
                name=row['Aspect'],
                line=dict(color=colors_3d[i % len(colors_3d)], width=4),
                marker=dict(size=6, color=colors_3d[i % len(colors_3d)]),
                hovertemplate='<b>%{text}</b><br>Score: %{z:.3f}<extra></extra>',
                text=[row['Aspect']]*3
            ))

        fig_3d.add_trace(go.Surface(
            x=[[0,1,2],[0,1,2]],
            y=[[-0.5,-0.5],[ 7.5, 7.5]],
            z=[[0,0],[0,0]],
            opacity=0.15,
            colorscale=[[0,'#888888'],[1,'#888888']],
            showscale=False,
            hoverinfo='skip'
        ))

        fig_3d.update_layout(
            template='plotly_dark',
            paper_bgcolor='#1E2130',
            scene=dict(
                xaxis=dict(tickvals=[0,1,2], ticktext=period_labels,
                           title='Time Period', gridcolor='#333', backgroundcolor='#1E2130'),
                yaxis=dict(tickvals=list(range(8)),
                           ticktext=absa_df['Aspect'].tolist(),
                           title='Aspect', gridcolor='#333', backgroundcolor='#1E2130'),
                zaxis=dict(title='Sentiment Score', gridcolor='#333', backgroundcolor='#1E2130',
                           range=[-0.4, 0.15]),
                bgcolor='#1E2130',
                camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2))
            ),
            legend=dict(bgcolor='#1E2130', font=dict(size=9)),
            margin=dict(t=20,b=20),
            height=420
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown('<div style="color:#666;font-size:0.78rem;margin-top:-10px">💡 Drag to rotate · Scroll to zoom · Click legend to toggle aspects</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">RoBERTa Emotion Analysis — Aug-Sep 2020</div>', unsafe_allow_html=True)
        emotion_df = pd.DataFrame({
            'Emotion':['Sadness','Optimism','Joy','Anger'],
            'Count':[6030,4536,3115,1319],
            'Pct':[40.2,30.2,20.8,8.8]
        })
        fig_emo = px.bar(emotion_df, x='Emotion', y='Count',
            color='Emotion',
            color_discrete_map={'Sadness':'#457B9D','Optimism':'#2A9D8F','Joy':'#E9C46A','Anger':'#E63946'},
            text=emotion_df['Pct'].apply(lambda x: f'{x}%'),
            template='plotly_dark')
        fig_emo.update_traces(textposition='outside', textfont_size=13,
            hovertemplate='<b>%{x}</b><br>Tweets: %{y:,}<extra></extra>')
        fig_emo.update_layout(showlegend=False, plot_bgcolor='#1E2130',
            paper_bgcolor='#1E2130', margin=dict(t=20,b=10), height=240)
        st.plotly_chart(fig_emo, use_container_width=True)

        st.markdown('<div class="section-header">Sarcasm/Irony Detection Rate</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=20.4,
            number={'suffix':'%', 'font':{'size':40, 'color':'#E63946'}},
            gauge={
                'axis':{'range':[0,40], 'tickcolor':'white'},
                'bar':{'color':'#E63946', 'thickness':0.3},
                'bgcolor':'#1E2130',
                'bordercolor':'#333',
                'steps':[
                    {'range':[0,10], 'color':'#1a3a2a'},
                    {'range':[10,25], 'color':'#3a3a1a'},
                    {'range':[25,40], 'color':'#3a1a1a'}
                ],
                'threshold':{
                    'line':{'color':'white','width':3},
                    'thickness':0.75,
                    'value':20.4
                }
            },
            title={'text':"of tweets are ironic/sarcastic",'font':{'color':'#888','size':12}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor='#1E2130', font={'color':'white'},
            height=220, margin=dict(t=30,b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("""
        <div class="finding-box">
        ⚠️ <b>Why this matters:</b> VADER classifies ironic tweets as <i>positive</i>
        despite negative intent — 20.4% sarcasm rate means ~1 in 5 tweets
        is systematically misclassified by lexicon-based approaches.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">ABSA Key Findings</div>', unsafe_allow_html=True)
    a1,a2,a3,a4 = st.columns(4)
    for col, (icon,title,desc,color) in zip([a1,a2,a3,a4],[
        ("😰","Mental Health","Most negative across ALL periods (-0.28)","#9B59B6"),
        ("🦠","Misinformation","Most negative overall (-0.34 → -0.29)","#E67E22"),
        ("💉","Vaccines","Steadily improving (+0.02 → +0.05)","#2A9D8F"),
        ("🔒","Lockdown","Got more negative (-0.05 → -0.14)","#E63946"),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:#1E2130;border-radius:10px;padding:16px;
                        border-top:3px solid {color};text-align:center;">
                <div style="font-size:1.8rem">{icon}</div>
                <div style="font-weight:700;color:{color};margin:6px 0">{title}</div>
                <div style="font-size:0.82rem;color:#AAA">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">BERTopic — Topic Distribution Across Periods</div>', unsafe_allow_html=True)

    t2020a = {"Healthcare":1566,"Masks":1563,"Reopening":1379,"Food":1317,
              "Education":871,"Mental Health":680,"Protests":667,
              "Treatment":643,"Lockdown":597,"Testing":489}
    t2020b = {"Testing":4899,"Vaccine Trials":3743,"Masks":2667,"Healthcare":1726,
              "Case Counts":1497,"Politics/Election":1121,"China/Origins":876,
              "Food":853,"Contact Tracing":824,"Treatment":740}
    t2021  = {"Vaccine Discourse":28000,"Vaccine Access":734,"Lockdown":384,
              "Prevention":251,"UK Govt":191,"Politics":191,"Protests":205,"Treatment":170}

    tc1,tc2,tc3 = st.columns(3)
    def topic_chart(data, title, color):
        df = pd.DataFrame(list(data.items()),columns=["Topic","Tweets"]).sort_values("Tweets")
        fig = px.bar(df,x="Tweets",y="Topic",orientation="h",
                     color_discrete_sequence=[color],template="plotly_dark",text="Tweets")
        fig.update_traces(textposition="outside", textfont_size=9)
        fig.update_layout(title=dict(text=title,font=dict(size=12,color="white")),
            plot_bgcolor="#1E2130",paper_bgcolor="#1E2130",showlegend=False,
            margin=dict(t=35,b=5,l=5,r=50),height=340,
            xaxis_title="",yaxis_title="",yaxis=dict(tickfont=dict(size=9)))
        return fig

    with tc1:
        st.plotly_chart(topic_chart(t2020a,"Apr-Jun 2020","#4ECDC4"), use_container_width=True)
        st.markdown('<div style="text-align:center;font-size:0.75rem;color:#888;font-style:italic">Early pandemic — broad fragmented discourse</div>', unsafe_allow_html=True)
    with tc2:
        st.plotly_chart(topic_chart(t2020b,"Aug-Sep 2020","#E9C46A"), use_container_width=True)
        st.markdown('<div style="text-align:center;font-size:0.75rem;color:#888;font-style:italic">Mid pandemic — testing, politics, vaccine trials</div>', unsafe_allow_html=True)
    with tc3:
        st.plotly_chart(topic_chart(t2021,"Apr-Jun 2021","#FF6B6B"), use_container_width=True)
        st.markdown('<div style="text-align:center;font-size:0.75rem;color:#888;font-style:italic">Late period — vaccine consolidates all discourse</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">LDA vs BERTopic</div>', unsafe_allow_html=True)
    lc,bc = st.columns(2)
    with lc:
        st.markdown("**LDA — Apr-Jun 2020**")
        st.dataframe(pd.DataFrame({
            "Metric":["Topics","Avg Size","Diversity","Perplexity","Coherence","Outliers"],
            "Value":["8 (fixed)","~16,965","0.863","3,850.93","-137.76","0%"]
        }), use_container_width=True, hide_index=True)
    with bc:
        st.markdown("**BERTopic — Apr-Jun 2020**")
        st.dataframe(pd.DataFrame({
            "Metric":["Topics","Avg Size","Diversity","Perplexity","Coherence","Outliers"],
            "Value":["14 (auto)","1,097","0.775","N/A","N/A","53.9%"]
        }), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Three-Phase Discourse Evolution</div>', unsafe_allow_html=True)
    p1,p2,p3 = st.columns(3)
    for col,(icon,period,phase,desc,color) in zip([p1,p2,p3],[
        ("🌱","Apr-Jun 2020","Early Pandemic",
         "Fragmented across 14 topics — masks, education, mental health, food, religion. Broad uncertainty.","#4ECDC4"),
        ("⚡","Aug-Sep 2020","Mid Pandemic",
         "Vaccine trials, US election, contact tracing apps, Remdesivir. A more informed public engaging with concrete events.","#E9C46A"),
        ("💉","Apr-Jun 2021","Vaccine Rollout",
         "Vaccine discourse overwhelmingly dominant. Masks and food insecurity disappeared. One narrative absorbs all others.","#FF6B6B"),
    ]):
        with col:
            st.markdown(f'''<div style="background:#1E2130;border-radius:10px;
                padding:16px;border-top:3px solid {color};">
                <div style="text-align:center;font-size:1.8rem">{icon}</div>
                <div style="font-weight:700;color:{color};text-align:center;
                     font-size:0.9rem;margin:6px 0">{period}<br>
                     <span style="color:#888;font-size:0.78rem;font-weight:400">{phase}</span></div>
                <div style="font-size:0.8rem;color:#AAA;line-height:1.5">{desc}</div>
            </div>''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — WORDCLOUD
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Dominant COVID-19 Terms — Interactive Word Clouds</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#AAA;font-size:0.85rem;margin-bottom:10px">Word size = topic frequency from BERTopic. Hover for tweet count. Switch periods to see discourse narrowing.</div>', unsafe_allow_html=True)

    selected = st.radio("", ["Apr-Jun 2020","Aug-Sep 2020","Apr-Jun 2021"],
                        horizontal=True, key="wc_radio")

    wc_raw = {"Apr-Jun 2020": [["covid19", 40829], ["case", 40829], ["vaccine", 40829], ["death", 40829], ["new", 40829], ["china", 40829], ["coronavirus", 40829], ["trump", 40829], ["test", 40829], ["covid cases", 40829], ["health", 1566], ["medic", 1566], ["healthcare", 1566], ["public health", 1566], ["care", 1566], ["public", 1566], ["doctor", 1566], ["worker", 1566], ["health worker", 1566], ["mask", 1563], ["wear", 1563], ["face", 1563], ["cover", 1563], ["protect", 1563], ["cloth", 1563], ["spread", 1563], ["prevent", 1563], ["use", 1563], ["reopen", 1379], ["school", 1379], ["state", 1379], ["phase", 1379], ["economy", 1379], ["plan", 1379], ["business", 1379], ["open", 1379], ["texas", 1379], ["restaurant", 1379], ["food", 1317], ["distribute", 1317], ["eat", 1317], ["supply", 1317], ["donate", 1317], ["help", 1317], ["provide", 1317], ["family", 1317], ["agriculture", 1317], ["deliver", 1317], ["tune", 1015], ["live", 1015], ["perform", 1015], ["tonight", 1015], ["rais", 1015], ["sing", 1015], ["covid19 relief", 1015]], "Aug-Sep 2020": [["covid test", 4899], ["test", 4899], ["death", 4899], ["positive", 4899], ["covid19", 4899], ["test positive", 4899], ["free", 4899], ["patient", 4899], ["vaccine", 3743], ["covid vaccine", 3743], ["trump", 3743], ["trial", 3743], ["trump covid", 3743], ["donald trump", 3743], ["donald", 3743], ["president trump", 3743], ["president", 3743], ["mask", 2667], ["wear", 2667], ["wear mask", 2667], ["face", 2667], ["distancing", 2667], ["cover", 2667], ["social distancing", 2667], ["wash", 2667], ["protect", 2667], ["social", 2667], ["health", 1726], ["medic", 1726], ["public health", 1726], ["care", 1726], ["healthcare", 1726], ["public", 1726], ["doctor", 1726], ["worker", 1726], ["case", 1497], ["covid cases", 1497], ["new cases", 1497], ["new", 1497], ["report", 1497], ["confirm", 1497], ["counti", 1497], ["record", 1497], ["ontario", 1258], ["canada", 1258], ["canadian", 1258], ["quebec", 1258], ["report new", 1258], ["ford", 1258], ["toronto", 1258], ["biden", 1121], ["joe biden", 1121], ["joe", 1121], ["debat", 1121], ["campaign", 1121]], "Apr-Jun 2021": [["covid19", 52698], ["vaccine", 52698], ["covid vaccine", 52698], ["case", 52698], ["new", 52698], ["india", 52698], ["health", 52698], ["report", 52698], ["people", 52698], ["death", 52698], ["provide", 734], ["sign", 734], ["access", 734], ["citi", 734], ["number", 734], ["park", 734], ["link", 734], ["watch", 637], ["live", 637], ["talk", 637], ["pleas", 637], ["share", 637], ["releas", 637], ["love", 637], ["pandemic", 384], ["time", 384], ["end", 384], ["impact", 384], ["year", 384], ["start", 384], ["since", 384], ["hope", 384], ["thing", 384], ["far", 384], ["june", 294], ["day", 294], ["join", 294], ["week", 294], ["meet", 294], ["open", 294], ["author", 293], ["test", 293], ["million", 293], ["surge", 293], ["medic", 293], ["government", 276], ["level", 276], ["care", 276], ["home", 276], ["data", 276], ["die", 276], ["use", 251], ["prevent", 251], ["import", 251], ["distancing", 251]]}

    color_schemes = {
        "Apr-Jun 2020": ["#FF6B6B","#FF8E53","#FF6B35","#FF9F43","#EE5A24",
                         "#FFC312","#FDA7DF","#FF7979","#F9CA24","#BADC58",
                         "#FF6B6B","#FF8E53","#FF6B35","#FF9F43","#EE5A24",
                         "#FFC312","#FDA7DF","#FF7979","#F9CA24","#BADC58",
                         "#FF6B6B","#FF8E53","#FF6B35","#FF9F43","#EE5A24",
                         "#FFC312","#FDA7DF","#FF7979","#F9CA24","#BADC58",
                         "#FF6B6B","#FF8E53","#FF6B35","#FF9F43","#EE5A24",
                         "#FFC312","#FDA7DF","#FF7979","#F9CA24","#BADC58",
                         "#FF6B6B","#FF8E53","#FF6B35","#FF9F43","#EE5A24",
                         "#FFC312","#FDA7DF","#FF7979","#F9CA24","#BADC58",
                         "#FF6B6B","#FF8E53","#FF6B35","#FF9F43","#EE5A24"],
        "Aug-Sep 2020": ["#4ECDC4","#45B7D1","#00D2D3","#54A0FF","#2ED573",
                         "#1E90FF","#74B9FF","#0984E3","#00CEC9","#81ECEC",
                         "#6C5CE7","#55EFC4","#4ECDC4","#45B7D1","#00D2D3",
                         "#54A0FF","#2ED573","#1E90FF","#74B9FF","#0984E3",
                         "#00CEC9","#81ECEC","#6C5CE7","#55EFC4","#4ECDC4",
                         "#45B7D1","#00D2D3","#54A0FF","#2ED573","#1E90FF",
                         "#74B9FF","#0984E3","#00CEC9","#81ECEC","#6C5CE7",
                         "#55EFC4","#4ECDC4","#45B7D1","#00D2D3","#54A0FF",
                         "#2ED573","#1E90FF","#74B9FF","#0984E3","#00CEC9",
                         "#81ECEC","#6C5CE7","#55EFC4","#4ECDC4","#45B7D1",
                         "#00D2D3","#54A0FF","#2ED573","#1E90FF","#74B9FF"],
        "Apr-Jun 2021": ["#A8E6CF","#3D9970","#27AE60","#2ECC71","#1ABC9C",
                         "#16A085","#6FCF97","#219653","#A8E6CF","#3D9970",
                         "#27AE60","#2ECC71","#1ABC9C","#16A085","#6FCF97",
                         "#219653","#A8E6CF","#3D9970","#27AE60","#2ECC71",
                         "#1ABC9C","#16A085","#6FCF97","#219653","#A8E6CF",
                         "#3D9970","#27AE60","#2ECC71","#1ABC9C","#16A085",
                         "#6FCF97","#219653","#A8E6CF","#3D9970","#27AE60",
                         "#2ECC71","#1ABC9C","#16A085","#6FCF97","#219653",
                         "#A8E6CF","#3D9970","#27AE60","#2ECC71","#1ABC9C",
                         "#16A085","#6FCF97","#219653","#A8E6CF","#3D9970",
                         "#27AE60","#2ECC71","#1ABC9C","#16A085","#6FCF97"],
    }

    border_colors = {"Apr-Jun 2020":"#FF6B6B","Aug-Sep 2020":"#4ECDC4","Apr-Jun 2021":"#A8E6CF"}
    narratives = {
        "Apr-Jun 2020": ("🌱","Early pandemic — scattered across masks, schools, mental health, food and faith. No dominant narrative."),
        "Aug-Sep 2020": ("⚡","Mid pandemic — testing surges, Trump/Biden election, vaccine trials. Discourse sharpens."),
        "Apr-Jun 2021": ("💉","Late period — vaccine completely dominates. One narrative absorbs all others."),
    }

    words_data = wc_raw[selected]
    colors     = color_schemes[selected]
    border     = border_colors[selected]
    icon, narr = narratives[selected]

    mn = min(c for _,c in words_data)
    mx = max(c for _,c in words_data)
    def fsize(v, lo=13, hi=78):
        if mx == mn: return (lo+hi)//2
        return int(lo + (v-mn)/(mx-mn)*(hi-lo))

    word_objects = json.dumps([
        {"text": str(w), "size": fsize(int(c)), "color": colors[i % len(colors)], "count": int(c)}
        for i, (w, c) in enumerate(words_data)
    ])

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3-cloud/1.2.5/d3.layout.cloud.min.js"></script>
<style>
  body{{margin:0;background:#0D1117;overflow:hidden}}
  #wc{{width:100%;height:460px;display:flex;align-items:center;justify-content:center}}
  text{{cursor:pointer;font-family:"Arial Black",Impact,sans-serif;font-weight:900;
        transition:opacity .15s ease}}
  text:hover{{opacity:.7}}
  #tip{{position:fixed;pointer-events:none;background:#1E2130;
        border:1px solid {border};color:#fff;padding:7px 13px;border-radius:8px;
        font:13px Arial,sans-serif;display:none;z-index:999;
        box-shadow:0 4px 12px rgba(0,0,0,.5)}}
</style></head><body>
<div id="wc"><svg id="svg"></svg></div>
<div id="tip"></div>
<script>
const words = {word_objects};
function render(){{
  const W = document.getElementById("wc").offsetWidth || 900;
  const H = 460;
  d3.select("#svg").attr("width",W).attr("height",H);
  d3.select("#svg").selectAll("*").remove();
  const g = d3.select("#svg").append("g").attr("transform","translate("+W/2+","+H/2+")");
  const tip = document.getElementById("tip");
  // Scale font sizes to canvas width
  const scale = Math.min(1, W/900);
  d3.layout.cloud()
    .size([W-10, H-10])
    .words(words.map(d=>({{...d}})))
    .padding(5)
    .rotate(()=>(Math.random()>.75?90:0))
    .font("Arial Black")
    .fontWeight("900")
    .fontSize(d=>Math.max(10, d.size*scale))
    .on("end", ws=>{{
      g.selectAll("text").data(ws).enter().append("text")
        .style("font-size",d=>d.size*scale+"px")
        .style("font-family","Arial Black,Impact,sans-serif")
        .style("font-weight","900")
        .style("fill",d=>d.color)
        .attr("text-anchor","middle")
        .attr("transform",d=>"translate("+[d.x,d.y]+")rotate("+d.rotate+")")
        .text(d=>d.text)
        .on("mousemove",function(e,d){{
          tip.style.display="block";
          tip.style.left=(e.clientX+14)+"px";
          tip.style.top=(e.clientY-10)+"px";
          tip.innerHTML="<b>"+d.text+"</b><br>"+d.count.toLocaleString()+" tweets";
        }})
        .on("mouseleave",()=>tip.style.display="none")
        .on("mouseover",function(e,d){{
          d3.select(this).transition().duration(120)
            .style("font-size",(d.size*scale*1.2)+"px").style("opacity","0.85");
        }})
        .on("mouseout",function(e,d){{
          d3.select(this).transition().duration(120)
            .style("font-size",(d.size*scale)+"px").style("opacity","1");
        }});
    }}).start();
}}
render();
window.addEventListener("resize", render);
</script></body></html>"""

    components.html(html, height=470, scrolling=False)

    st.markdown(f'''<div style="background:#1E2130;border-radius:10px;
        padding:12px 16px;border-left:4px solid {border};margin-top:6px;">
        <span style="font-size:1.1rem">{icon}</span>
        <span style="color:{border};font-weight:700;margin-left:8px">{selected}</span>
        <br><span style="color:#CCCCCC;font-size:0.86rem;line-height:1.5">{narr}</span>
    </div>''', unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('''<div style="text-align:center;color:#555;font-size:0.78rem;padding:8px">
    Social Media Analytics of COVID-19 Discourse · MS5131 · Bhawna Patnaik & Aditya Bhilare · 1MBY1 MSc Business Analytics
</div>''', unsafe_allow_html=True)
