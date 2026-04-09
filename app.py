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
        border: 1px solid #2A9D8F55;
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

st.markdown("# 🦠 COVID-19 Social Media Analytics Dashboard")
st.markdown("##### Bhawna Patnaik & Aditya Bhilare · MSc Business Analytics · MS5131")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📊 Sentiment & Models",
    "🔍 ABSA & Emotions",
    "🗺️ Topic Evolution"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, (val, label) in zip([c1,c2,c3,c4], [
        ("376,966","Total Tweets Analysed"),
        ("411,887","Raw Tweets Collected"),
        ("3","Time Periods"),
        ("5","Analytical Methods")
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">Dataset Breakdown</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        'Period':['Apr-Jun 2020','Aug-Sep 2020','Apr-Jun 2021','Total'],
        'Raw Tweets':['143,903','120,509','147,475','411,887'],
        'Cleaned Tweets':['135,723','111,326','129,917','376,966'],
        'Removed':['8,180','9,183','17,558','34,921'],
        'Analytical Scope':['Sentiment + ABSA + LDA + BERTopic','ABSA + BERTopic + RoBERTa','ABSA + BERTopic','']
    }), use_container_width=True, hide_index=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Sentiment Distribution — Apr-Jun 2020</div>', unsafe_allow_html=True)
        sent_df = pd.DataFrame({
            'Sentiment':['Neutral','Positive','Negative'],
            'Count':[53083,44427,38213],
            'Pct':[39.1,32.7,28.2]
        })
        fig_sent = px.bar(sent_df, x='Sentiment', y='Count',
            color='Sentiment',
            color_discrete_map={'Negative':'#E63946','Neutral':'#6C757D','Positive':'#2A9D8F'},
            text=sent_df['Pct'].apply(lambda x: f'{x}%'),
            template='plotly_dark',
            hover_data={'Count':True,'Pct':True,'Sentiment':False})
        fig_sent.update_traces(textposition='outside', textfont_size=13,
            hovertemplate='<b>%{x}</b><br>Count: %{y:,}<br>Percentage: %{customdata[1]}%<extra></extra>')
        fig_sent.update_layout(showlegend=False, plot_bgcolor='#1E2130',
            paper_bgcolor='#1E2130', margin=dict(t=20,b=20), height=320)
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
        model_df = pd.DataFrame({
            'Model':['VADER','SVM + TF-IDF','BERT'],
            'Accuracy':[98.44,97.55,98.88],
            'Macro F1':[0.98,0.97,0.99],
            'Precision':[0.99,0.97,0.99],
            'Recall':[0.98,0.97,0.99]
        })
        metric_sel = st.selectbox('Select metric to display:', ['Accuracy','Macro F1','Precision','Recall'])
        yvals = model_df[metric_sel].tolist()
        if metric_sel == 'Accuracy':
            yvals_norm = [v/100 for v in yvals]
            yrange = [0.96, 1.01]
        else:
            yvals_norm = yvals
            yrange = [0.95, 1.01]
        fig_mod = go.Figure()
        colors = ['#2A9D8F','#E9C46A','#E63946']
        for i, (model, val) in enumerate(zip(model_df['Model'], yvals_norm)):
            fig_mod.add_trace(go.Bar(
                x=[model], y=[val],
                name=model,
                marker_color=colors[i],
                text=[f'{val:.2f}' if metric_sel != 'Accuracy' else f'{yvals[i]:.2f}%'],
                textposition='outside',
                hovertemplate=f'<b>{model}</b><br>{metric_sel}: {yvals[i]}<extra></extra>'
            ))
        fig_mod.update_layout(
            template='plotly_dark', plot_bgcolor='#1E2130', paper_bgcolor='#1E2130',
            showlegend=False, margin=dict(t=20,b=20), height=280,
            yaxis=dict(range=yrange)
        )
        st.plotly_chart(fig_mod, use_container_width=True)

    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    with f1:
        st.markdown('<div class="finding-box">⚠️ <b>Label Leakage:</b> 98.4% agreement between fresh VADER predictions and dataset labels — accuracy reflects VADER approximation, not human-validated sentiment.</div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="finding-box">🏆 <b>BERT Best:</b> 98.88% accuracy, but only 1.33 points above VADER — questioning the value of expensive transformer fine-tuning on automatically labelled data.</div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="finding-box">📊 <b>Neutral Dominates:</b> 39.1% neutral confirms Twitter was an information-sharing platform during COVID-19 rather than primarily emotional outlet.</div>', unsafe_allow_html=True)

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
        margin=dict(t=20,b=20), height=380,
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
        st.caption("💡 Drag to rotate · Scroll to zoom · Click legend to toggle aspects")

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
    st.markdown('<div class="section-header">3D Topic Landscape Across Three Periods</div>', unsafe_allow_html=True)

    all_topics = {
        'Masks':            [1563, 2667, 0],
        'Testing':          [489,  4899, 0],
        'Vaccine Discourse':[0,    3743, 28000],
        'Healthcare':       [1566, 1726, 0],
        'Food Insecurity':  [1317, 853,  0],
        'Education':        [871,  0,    0],
        'Mental Health':    [680,  0,    0],
        'Protests':         [667,  0,    205],
        'Treatment/Drugs':  [643,  740,  170],
        'Lockdown Policy':  [597,  0,    384],
        'Politics/Election':[0,    1121, 191],
        'China/Origins':    [0,    876,  0],
        'Contact Tracing':  [0,    824,  0],
        'Vaccine Access':   [0,    0,    734],
    }

    topic_colors_3d = px.colors.qualitative.Vivid

    fig_3d_topics = go.Figure()
    period_labels = ['Apr-Jun 2020','Aug-Sep 2020','Apr-Jun 2021']

    for i, (topic, counts) in enumerate(all_topics.items()):
        color = topic_colors_3d[i % len(topic_colors_3d)]
        non_zero = [(j, c) for j, c in enumerate(counts) if c > 0]
        if len(non_zero) < 2:
            if non_zero:
                j, c = non_zero[0]
                fig_3d_topics.add_trace(go.Scatter3d(
                    x=[period_labels[j]],
                    y=[topic],
                    z=[c],
                    mode='markers',
                    name=topic,
                    marker=dict(size=8, color=color),
                    hovertemplate=f'<b>{topic}</b><br>Period: %{{x}}<br>Tweets: %{{z:,}}<extra></extra>'
                ))
        else:
            fig_3d_topics.add_trace(go.Scatter3d(
                x=[period_labels[j] for j, c in enumerate(counts)],
                y=[topic]*3,
                z=counts,
                mode='lines+markers',
                name=topic,
                line=dict(color=color, width=5),
                marker=dict(size=[8 if c > 0 else 3 for c in counts], color=color),
                hovertemplate=f'<b>{topic}</b><br>Period: %{{x}}<br>Tweets: %{{z:,}}<extra></extra>'
            ))

    fig_3d_topics.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1E2130',
        scene=dict(
            xaxis=dict(title='Time Period', gridcolor='#333', backgroundcolor='#1E2130'),
            yaxis=dict(title='Topic', gridcolor='#333', backgroundcolor='#1E2130',
                       tickfont=dict(size=9)),
            zaxis=dict(title='Tweet Count', gridcolor='#333', backgroundcolor='#1E2130'),
            bgcolor='#1E2130',
            camera=dict(eye=dict(x=2.0, y=-2.0, z=1.5))
        ),
        legend=dict(bgcolor='#1E2130', font=dict(size=9), x=1.0),
        margin=dict(t=20, b=20),
        height=520
    )
    st.plotly_chart(fig_3d_topics, use_container_width=True)
    st.caption("💡 Drag to rotate · Scroll to zoom · Click legend items to show/hide topics · Hover for exact counts")

    st.markdown('<div class="section-header">BERTopic Topic Distribution Per Period</div>', unsafe_allow_html=True)
    period_sel = st.selectbox('Select time period:', ['Apr-Jun 2020','Aug-Sep 2020','Apr-Jun 2021'])

    topic_data = {
        'Apr-Jun 2020': {'Healthcare':1566,'Masks':1563,'Reopening':1379,'Food Insecurity':1317,
                          'Education':871,'Mental Health':680,'Protests':667,'Treatment/Drugs':643,
                          'Lockdown Policy':597,'Testing':489},
        'Aug-Sep 2020': {'Testing':4899,'Vaccine Trials':3743,'Masks':2667,'Healthcare':1726,
                          'Case Counts':1497,'Politics/Election':1121,'China/Origins':876,
                          'Food Insecurity':853,'Contact Tracing':824,'Treatment/Drugs':740},
        'Apr-Jun 2021': {'Vaccine Discourse':28000,'Vaccine Access':734,'Lockdown Policy':384,
                          'Prevention':251,'UK Govt/Policy':191,'Politics/Election':191,
                          'Protests':205,'Treatment/Drugs':170}
    }
    period_colors = {'Apr-Jun 2020':'#4ECDC4','Aug-Sep 2020':'#E9C46A','Apr-Jun 2021':'#FF6B6B'}

    df_p = pd.DataFrame(list(topic_data[period_sel].items()), columns=['Topic','Tweets'])
    df_p = df_p.sort_values('Tweets', ascending=True)
    fig_bar = px.bar(df_p, x='Tweets', y='Topic', orientation='h',
                     color_discrete_sequence=[period_colors[period_sel]],
                     template='plotly_dark', text='Tweets')
    fig_bar.update_traces(textposition='outside', textfont_size=11,
        hovertemplate='<b>%{y}</b><br>Tweets: %{x:,}<extra></extra>')
    fig_bar.update_layout(
        plot_bgcolor='#1E2130', paper_bgcolor='#1E2130',
        showlegend=False, margin=dict(t=10,b=10,l=10,r=60), height=350
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-header">The Three-Phase COVID-19 Discourse Evolution</div>', unsafe_allow_html=True)
    p1,p2,p3 = st.columns(3)
    for col,(icon,period,phase,desc,color) in zip([p1,p2,p3],[
        ("🌱","Apr-Jun 2020","Early Pandemic",
         "Fragmented discourse across 14 topics — masks, education, mental health, food insecurity. Broad uncertainty with no dominant narrative.","#4ECDC4"),
        ("⚡","Aug-Sep 2020","Mid Pandemic",
         "Discourse sharpened — vaccine trials, US election, contact tracing, Remdesivir. A more informed public engaging with concrete developments.","#E9C46A"),
        ("💉","Apr-Jun 2021","Vaccine Rollout",
         "Vaccine discourse overwhelmingly dominant. Previously distinct topics disappeared entirely — collective attention consolidated around vaccination.","#FF6B6B"),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:#1E2130;border-radius:12px;padding:20px;
                        border-top:4px solid {color};min-height:200px;">
                <div style="font-size:2rem;text-align:center">{icon}</div>
                <div style="font-weight:700;color:{color};font-size:0.95rem;
                             text-align:center;margin:8px 0">{period}<br>
                     <span style="color:#888;font-size:0.8rem;font-weight:400">{phase}</span></div>
                <div style="font-size:0.82rem;color:#AAA;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    <div style="background:#1E2130;border-radius:10px;padding:16px 20px;
                border-left:4px solid #2A9D8F;margin-top:10px;">
        <span style="color:#2A9D8F;font-weight:700">Key Finding: </span>
        <span style="color:#CCCCCC;font-size:0.9rem">
        BERTopic (diversity: 0.775, 14 automatic topics) captured finer semantic granularity than LDA
        (diversity: 0.863, 8 fixed topics). This temporal evolution from fragmented early pandemic
        uncertainty to vaccine-dominated discourse provides empirical evidence of how collective
        public attention consolidates around dominant narratives during extended crises.
        </span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#555;font-size:0.8rem;padding:10px">
    Social Media Analytics of COVID-19 Discourse · MS5131 Major Business Analytics Project ·
    Bhawna Patnaik & Aditya Bhilare · 1MBY1 MSc Business Analytics
</div>""", unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-header">Dominant COVID-19 Terms — Interactive Word Bubbles</div>',
                unsafe_allow_html=True)
    
    period_words = {
        'Apr-Jun 2020': {
            'covid19': 50, 'lockdown': 30, 'mask': 28, 'death': 25,
            'school': 22, 'health': 20, 'food': 18, 'mental': 16,
            'teacher': 15, 'protest': 14, 'faith': 12, 'economy': 11,
            'hospital': 18, 'reopen': 16, 'quarantine': 14
        },
        'Aug-Sep 2020': {
            'covid19': 45, 'test': 48, 'trump': 40, 'positive': 35,
            'vaccine': 38, 'biden': 28, 'death': 25, 'canada': 22,
            'election': 20, 'mask': 26, 'china': 18, 'remdesivir': 15,
            'contact': 16, 'trace': 14, 'wuhan': 12
        },
        'Apr-Jun 2021': {
            'vaccine': 70, 'vaccination': 65, 'covid19': 45, 'pfizer': 38,
            'moderna': 35, 'dose': 32, 'delta': 25, 'variant': 22,
            'lockdown': 20, 'boris': 15, 'johnson': 14, 'case': 18,
            'death': 16, 'government': 14, 'ease': 12
        }
    }
    
    colors = {
        'Apr-Jun 2020': '#4ECDC4',
        'Aug-Sep 2020': '#E9C46A', 
        'Apr-Jun 2021': '#FF6B6B'
    }
    
    selected = st.radio(
        "Select time period:",
        list(period_words.keys()),
        horizontal=True
    )
    
    words = period_words[selected]
    np.random.seed(42)
    
    # Generate bubble positions
    n = len(words)
    labels = list(words.keys())
    sizes = list(words.values())
    
    # Spiral layout
    angles = np.linspace(0, 4*np.pi, n)
    radii = np.linspace(0.1, 1.0, n)
    x = radii * np.cos(angles) + np.random.uniform(-0.1, 0.1, n)
    y = radii * np.sin(angles) + np.random.uniform(-0.1, 0.1, n)
    
    fig_wc = go.Figure()
    
    fig_wc.add_trace(go.Scatter(
        x=x, y=y,
        mode='text+markers',
        text=labels,
        marker=dict(
            size=[s*1.5 for s in sizes],
            color=colors[selected],
            opacity=0.15,
            line=dict(width=0)
        ),
        textfont=dict(
            size=[max(10, s//2) for s in sizes],
            color=colors[selected]
        ),
        hovertemplate='<b>%{text}</b><br>Relative frequency: %{marker.size}<extra></extra>'
    ))
    
    fig_wc.update_layout(
        template='plotly_dark',
        plot_bgcolor='#1E2130',
        paper_bgcolor='#1E2130',
        height=500,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=20, b=20)
    )
    
    st.plotly_chart(fig_wc, use_container_width=True)
    
    # Narrative below
    narratives = {
        'Apr-Jun 2020': '🌱 Early pandemic — discourse scattered across masks, schools, mental health, food and faith. No single dominant narrative.',
        'Aug-Sep 2020': '⚡ Mid pandemic — testing surges, Trump/Biden election heats up, vaccine trials announced. Discourse sharpens around specific events.',
        'Apr-Jun 2021': '💉 Late period — vaccine completely dominates. The magnifying glass has zoomed in on a single topic. Pfizer, Moderna, dose, delta — one narrative.'
    }
    
    st.markdown(f"""
    <div style="background:#1E2130; border-radius:10px; padding:16px 20px;
                border-left: 4px solid {colors[selected]}; margin-top:10px;">
        <span style="color:{colors[selected]}; font-size:1rem">{narratives[selected]}</span>
    </div>""", unsafe_allow_html=True)
