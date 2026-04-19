import streamlit as st


DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #08090F;
    --bg2: #0D1117;
    --card: rgba(255,255,255,0.04);
    --card-border: rgba(0, 212, 255, 0.12);
    --primary: #00D4FF;
    --secondary: #7B2FBE;
    --accent: #FF6B35;
    --text: #CDD6E0;
    --text-dim: #64748B;
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-head) !important;
}

/* Hide SVG icons bleeding into expander labels and sidebar nav */
[data-testid="stExpander"] summary svg,
[data-testid="stSidebarNav"] svg,
[data-testid="stSidebarNavItems"] svg,
section[data-testid="stSidebar"] nav svg {
    display: none !important;
}

/* Expander label font fix */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    font-family: var(--font-head) !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    overflow: visible !important;
}

/* Sidebar nav text */
[data-testid="stSidebarNavItems"] a span,
[data-testid="stSidebarNavLink"] span {
    font-family: var(--font-head) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--card-border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: var(--font-head) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,255,0.3) !important;
}

.stSlider [data-baseweb="slider"] {
    color: var(--primary) !important;
}
.stSlider [data-testid="stThumbValue"] { color: var(--primary) !important; }

.stSelectbox [data-baseweb="select"] {
    background: var(--card) !important;
    border-color: var(--card-border) !important;
}

.stDataFrame { background: var(--card) !important; }

div[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    backdrop-filter: blur(10px) !important;
}
div[data-testid="metric-container"] label {
    color: var(--text-dim) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-family: var(--font-head) !important;
    font-weight: 800 !important;
}

h1, h2, h3, h4 {
    font-family: var(--font-head) !important;
    color: var(--text) !important;
}

p, li, span { font-family: var(--font-head) !important; }

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-family: var(--font-head) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
}

[data-testid="stFileUploader"] {
    border: 1px dashed var(--card-border) !important;
    background: var(--card) !important;
    border-radius: 12px !important;
}

.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    border-color: rgba(0,212,255,0.3);
    box-shadow: 0 8px 32px rgba(0,212,255,0.08);
    transform: translateY(-2px);
}

.rec-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(123,47,190,0.05));
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.rec-card:hover {
    border-color: rgba(0,212,255,0.4);
    box-shadow: 0 4px 24px rgba(0,212,255,0.1);
}
.rec-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #00D4FF;
    margin-bottom: 0.3rem;
}
.rec-reason {
    font-size: 0.82rem;
    color: #94A3B8;
    margin-bottom: 0.5rem;
}
.rec-badge {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    color: #10B981;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-top: 4px;
}
.rec-model-tag {
    display: inline-block;
    background: rgba(123,47,190,0.15);
    border: 1px solid rgba(123,47,190,0.3);
    color: #A78BFA;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-top: 4px;
}

.page-header {
    border-bottom: 1px solid rgba(0,212,255,0.12);
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    background: linear-gradient(135deg, #00D4FF, #7B2FBE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.page-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
}

.stat-pill {
    display: inline-block;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    color: #00D4FF;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 2px;
}

.best-model-banner {
    background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,47,190,0.1));
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.best-model-name {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    color: #00D4FF;
}
.best-model-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #A78BFA;
    margin-top: 0.3rem;
}

.fl-stat {
    text-align: center;
    padding: 1rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 12px;
}
.fl-stat-num {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    color: #00D4FF;
}
.fl-stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.pred-result {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(16,185,129,0.05));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.pred-value {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    color: #00D4FF;
}
.pred-unit {
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    color: #64748B;
}
</style>
"""


def inject_css():
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def page_header(title, subtitle=""):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">{title}</div>
        {"<div class='page-sub'>" + subtitle + "</div>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def glass_card(content_html):
    st.markdown(f'<div class="glass-card">{content_html}</div>', unsafe_allow_html=True)


def fl_stat(num, label):
    return f"""<div class="fl-stat"><div class="fl-stat-num">{num}</div><div class="fl-stat-label">{label}</div></div>"""


def sidebar_info():
    with st.sidebar:
        st.markdown("""
        <div style='padding:1rem 0; border-bottom:1px solid rgba(0,212,255,0.1); margin-bottom:1rem;'>
            <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.1rem;
                background:linear-gradient(135deg,#00D4FF,#7B2FBE);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;'>
                Energy Savings AI
            </div>
            <div style='font-family:"DM Mono",monospace;font-size:0.65rem;color:#64748B;margin-top:2px;'>
                FedProx Optimized System
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:"DM Mono",monospace;font-size:0.68rem;color:#64748B;
            padding:0.8rem;background:rgba(0,212,255,0.03);
            border:1px solid rgba(0,212,255,0.08);border-radius:8px;'>
            <div style='color:#00D4FF;margin-bottom:4px;font-weight:500;'>System Info</div>
            Samples: ~50,000<br>
            Clients: 10<br>
            Strategy: FedProx<br>
            Model: Random Forest
        </div>
        """, unsafe_allow_html=True)
