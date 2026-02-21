# app.py
# ------------------------------------------------------------
# HVAC Maintenance & Inefficiency Agent (Streamlit, single-file)
# Copy-paste into app.py, then:
#   pip install -r requirements.txt
#   streamlit run app.py
# Optional: put MP4s in ./videos/*.mp4 for the left feed panel.
# ------------------------------------------------------------

import time
from pathlib import Path
from datetime import datetime, timedelta
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Page setup + CSS tightening
# ============================================================
st.set_page_config(page_title="HVAC Maintenance & Inefficiency Agent", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.0rem;}
      .stMetric {padding: 6px 10px;}
      div[data-testid="stVerticalBlockBorderWrapper"] {padding: 10px;}
      .tight-card {padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(49,51,63,0.15);}
      .muted {opacity: 0.75;}
      .pill {display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(49,51,63,0.18);}
      .small {font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏢🧠 HVAC Maintenance & Inefficiency Agent")
st.caption("Left: MP4 feed (optional) | Right: Chiller + AHU telemetry, inefficiency flags, maintenance prediction, GenBI (offline demo).")


# ============================================================
# Video loading (optional)
# ============================================================
VIDEO_DIR = Path("videos")
video_files = sorted([p for p in VIDEO_DIR.glob("*.mp4")]) if VIDEO_DIR.exists() else []
HAS_VIDEO = len(video_files) > 0


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.header("Controls")

    autoplay = st.toggle("Autoplay telemetry", value=True)
    tick_ms = st.slider("Refresh speed (ms)", 150, 1500, 350, 10)

    st.divider()
    st.subheader("Signal realism (demo)")
    noise = st.slider("Sensor noise", 0.0, 3.0, 0.6, 0.1)
    drift = st.slider("Degradation drift", 0.0, 2.0, 0.7, 0.1)
    stress = st.slider("Load stress", 0.0, 2.0, 0.9, 0.1)

    st.divider()
    st.subheader("Asset selection")
    asset = st.selectbox(
        "Pick an asset",
        ["Plant-A | Chiller-01 + AHU-07", "Plant-A | Chiller-02 + AHU-03", "Plant-B | Chiller-01 + AHU-02"],
        index=0,
    )

    st.divider()
    st.subheader("Design baseline (reference)")
    design_kw_per_tr = st.slider("Design kW/TR", 0.45, 1.10, 0.62, 0.01)
    design_delta_t_chw = st.slider("Design CHW ΔT (°C)", 3.0, 9.0, 5.5, 0.1)
    design_supply_air = st.slider("Design Supply Air Temp (°C)", 10.0, 18.0, 13.0, 0.1)
    design_ahu_cfm = st.slider("Design Airflow (CFM)", 5000, 60000, 22000, 500)

    st.divider()
    st.subheader("Video")
    if HAS_VIDEO:
        if "video_idx" not in st.session_state:
            st.session_state.video_idx = 0
        chosen = st.selectbox(
            "Pick a video",
            options=list(range(len(video_files))),
            format_func=lambda i: f"{i+1}. {video_files[i].name}",
            index=st.session_state.video_idx
        )
        st.session_state.video_idx = chosen

        cA, cB, cC = st.columns(3)
        with cA:
            if st.button("⏮ Prev"):
                st.session_state.video_idx = (st.session_state.video_idx - 1) % len(video_files)
        with cB:
            if st.button("▶ Next"):
                st.session_state.video_idx = (st.session_state.video_idx + 1) % len(video_files)
        with cC:
            if st.button("🔁 Reset"):
                st.session_state.video_idx = 0
    else:
        st.caption("No videos found in ./videos. App runs fine without video.")


# ============================================================
# Layout
# ============================================================
left, right = st.columns([1.2, 1.0], gap="large")


# ============================================================
# Telemetry generation (synthetic but physically-inspired)
# ============================================================
def make_hvac_series(seed: int, n: int = 320, noise: float = 0.6, drift: float = 0.7, stress: float = 0.9):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Load profile: daily-ish + short cycles + stress noise
    load_frac = 0.55 + 0.18 * np.sin(2 * np.pi * t / 95) + 0.10 * np.sin(2 * np.pi * t / 31)
    load_frac += stress * 0.06 * rng.normal(0, 1, size=n)
    load_frac = np.clip(load_frac, 0.25, 0.98)

    # Cooling load (TR)
    tr = 220 * load_frac + rng.normal(0, noise * 2.5, size=n)
    tr = np.clip(tr, 30, 240)

    # Degradation proxies
    condenser_fouling = (drift * 0.0018) * t
    coil_fouling = (drift * 0.0012) * t
    filter_loading = (drift * 0.0022) * t

    # Occasional maintenance "resets"
    for k in range(90, n, 110):
        condenser_fouling[k:] -= 0.10
        coil_fouling[k:] -= 0.07
        filter_loading[k:] -= 0.12

    condenser_fouling = np.clip(condenser_fouling, 0, 0.9)
    coil_fouling = np.clip(coil_fouling, 0, 0.9)
    filter_loading = np.clip(filter_loading, 0, 0.9)

    # Part-load penalty (inefficiency) when load is low
    part_load_penalty = 0.22 * np.clip(0.45 - load_frac, 0, 1)

    # kW/TR increases with condenser fouling + part-load penalty
    kw_per_tr = 0.58 + 0.25 * condenser_fouling + part_load_penalty + rng.normal(0, noise * 0.02, size=n)
    kw_per_tr = np.clip(kw_per_tr, 0.45, 1.30)
    chiller_kw = tr * kw_per_tr

    # CHW temps
    chw_supply = 6.5 + 0.4 * rng.normal(0, noise, size=n)
    chw_return = chw_supply + (4.5 + 2.5 * load_frac - 1.8 * coil_fouling) + rng.normal(0, noise * 0.3, size=n)
    chw_delta_t = chw_return - chw_supply

    # AHU airflow & fan power
    cfm = 18000 + 14000 * load_frac - 7000 * filter_loading + rng.normal(0, noise * 300, size=n)
    cfm = np.clip(cfm, 7000, 42000)

    filter_dp = 0.35 + 1.25 * filter_loading + rng.normal(0, noise * 0.04, size=n)  # proxy
    filter_dp = np.clip(filter_dp, 0.2, 2.2)

    fan_kw = 6.0 + 0.00022 * (cfm ** 1.15) + 2.2 * filter_loading + rng.normal(0, noise * 0.3, size=n)
    fan_kw = np.clip(fan_kw, 3.0, 35.0)

    # Air temps
    return_air = 24.5 + 1.8 * load_frac + rng.normal(0, noise * 0.4, size=n)
    supply_air = 13.5 + 2.2 * coil_fouling + rng.normal(0, noise * 0.3, size=n)  # coil fouling worsens SAT
    sat_delta = return_air - supply_air

    # A crude "COP proxy" (not thermodynamically perfect, but stable for demo)
    # COP roughly inversely proportional to kW/TR (scaled)
    cop_proxy = np.clip(3.5 * (0.75 / kw_per_tr), 1.0, 7.0)

    return {
        "t": t,
        "tr": tr,
        "chiller_kw": chiller_kw,
        "kw_per_tr": kw_per_tr,
        "cop_proxy": cop_proxy,
        "chw_supply": chw_supply,
        "chw_return": chw_return,
        "chw_delta_t": chw_delta_t,
        "cfm": cfm,
        "fan_kw": fan_kw,
        "filter_dp": filter_dp,
        "return_air": return_air,
        "supply_air": supply_air,
        "sat_delta": sat_delta,
        "condenser_fouling": condenser_fouling,
        "coil_fouling": coil_fouling,
        "filter_loading": filter_loading,
        "load_frac": load_frac,
    }


def status_from_score(x: float):
    if x >= 70:
        return "ALERT"
    if x >= 40:
        return "WATCH"
    return "NORMAL"


def compute_scores(x, cursor, design_kw_per_tr, design_delta_t_chw, design_supply_air, design_ahu_cfm):
    kwtr = float(x["kw_per_tr"][cursor])
    dT = float(x["chw_delta_t"][cursor])
    sat = float(x["supply_air"][cursor])
    cfm = float(x["cfm"][cursor])
    fdp = float(x["filter_dp"][cursor])
    fouling_c = float(x["condenser_fouling"][cursor])
    fouling_a = float(x["coil_fouling"][cursor])
    filt = float(x["filter_loading"][cursor])
    load = float(x["load_frac"][cursor])

    # Normalize to 0..1 "badness"
    kwtr_norm = np.clip((kwtr - design_kw_per_tr) / 0.35, 0, 1)             # higher than design = bad
    dT_norm = np.clip((design_delta_t_chw - dT) / 3.5, 0, 1)                # lower than design = bad
    sat_norm = np.clip((sat - design_supply_air) / 4.0, 0, 1)               # higher SAT than design = bad
    cfm_norm = np.clip((design_ahu_cfm - cfm) / (0.6 * design_ahu_cfm), 0, 1)
    fdp_norm = np.clip((fdp - 0.4) / 1.4, 0, 1)

    # Inefficiency (energy leakage)
    inefficiency = (
        0.52 * kwtr_norm +
        0.18 * dT_norm +
        0.15 * fdp_norm +
        0.15 * (0.6 * sat_norm + 0.4 * cfm_norm)
    ) * 100
    inefficiency = float(np.clip(inefficiency, 0, 100))

    # Maintenance risk (service urgency)
    maintenance = (
        0.35 * (0.65 * fouling_c + 0.35 * kwtr_norm) +
        0.25 * (0.70 * filt + 0.30 * fdp_norm) +
        0.25 * (0.70 * fouling_a + 0.30 * sat_norm) +
        0.15 * (0.60 * dT_norm + 0.40 * (load > 0.85))
    ) * 100
    maintenance = float(np.clip(maintenance, 0, 100))

    # RUL (hours) — decreases nonlinearly with risk
    rul = float(np.clip(240 * (1 - (maintenance / 100) ** 1.35), 6, 240))

    # Findings (root-cause hints)
    findings = []
    if kwtr_norm > 0.55 and fouling_c > 0.35:
        findings.append("Condenser fouling likely (kW/TR drift + fouling proxy high).")
    if fdp_norm > 0.55 or filt > 0.55:
        findings.append("Filter loading high (ΔP rising) → airflow restriction & fan energy waste.")
    if sat_norm > 0.55 and fouling_a > 0.30:
        findings.append("Cooling coil fouling likely (SAT higher than design) → comfort/latent issues.")
    if dT_norm > 0.55:
        findings.append("Low CHW ΔT syndrome → bypass/overpumping/valve issues (energy waste).")
    if load < 0.45 and kwtr_norm > 0.35:
        findings.append("Part-load inefficiency detected (cycling / staging mismatch).")

    if not findings:
        findings.append("No dominant fault signature detected (within baseline band for this demo).")

    return inefficiency, maintenance, rul, findings


def make_line(fig_h, x, y, name, yaxis=None):
    tr = go.Scatter(x=x, y=y, mode="lines", name=name)
    if yaxis:
        tr.update(yaxis=yaxis)
    fig_h.add_trace(tr)


def genbi_answer(q: str, x, cursor, ineff, maint, rul, next_maint_str, status_eff, status_maint, findings):
    ql = q.strip().lower()
    if not ql:
        return None, None

    # Simple intents
    if ("current" in ql or "now" in ql) and ("risk" in ql or "maintenance" in ql):
        return f"Current **maintenance risk** is **{maint:.0f}/100** (**{status_maint}**) and predicted RUL is **{rul:.0f} hrs**.", None

    if "ineff" in ql or "energy" in ql or "kw/tr" in ql:
        return f"Current **inefficiency score** is **{ineff:.0f}/100** (**{status_eff}**). kW/TR now = **{x['kw_per_tr'][cursor]:.2f}**.", None

    if "next" in ql and ("service" in ql or "maintenance" in ql):
        return f"Predicted next maintenance window by **{next_maint_str}** (demo).", None

    if "root" in ql or "cause" in ql or "why" in ql:
        bullet = "\n".join([f"- {f}" for f in findings[:5]])
        return f"Top suspected drivers (demo):\n{bullet}", None

    if "anomal" in ql or "spike" in ql:
        w = 90
        s = max(0, cursor - w)
        series = x["kw_per_tr"][s:cursor + 1]
        z = (series - series.mean()) / (series.std() + 1e-6)
        spikes = int((np.abs(z) > 2.2).sum())
        return f"Detected **{spikes}** kW/TR anomaly candidates in last {len(series)} ticks (demo z-score > 2.2).", None

    # Trend requests: "show last N ticks <metric> trend"
    m = re.search(r"last\s+(\d+)\s+ticks", ql)
    n = int(m.group(1)) if m else 60
    n = int(np.clip(n, 20, 200))
    s = max(0, cursor - n)

    metric_map = {
        "kw/tr": ("kW/TR", x["kw_per_tr"], "kW/TR"),
        "cop": ("COP (proxy)", x["cop_proxy"], "COP"),
        "dt": ("CHW ΔT", x["chw_delta_t"], "°C"),
        "delta t": ("CHW ΔT", x["chw_delta_t"], "°C"),
        "sat": ("Supply Air Temp", x["supply_air"], "°C"),
        "airflow": ("Airflow (CFM)", x["cfm"], "CFM"),
        "cfm": ("Airflow (CFM)", x["cfm"], "CFM"),
        "filter": ("Filter ΔP", x["filter_dp"], "ΔP (proxy)"),
        "fan": ("Fan kW", x["fan_kw"], "kW"),
        "chiller kw": ("Chiller kW", x["chiller_kw"], "kW"),
        "tr": ("Cooling Load (TR)", x["tr"], "TR"),
    }

    # Find which metric user asked for
    chosen_key = None
    for k in metric_map.keys():
        if k in ql:
            chosen_key = k
            break

    if chosen_key:
        title, series, unit = metric_map[chosen_key]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x["t"][s:cursor + 1], y=series[s:cursor + 1], mode="lines", name=title))
        fig.update_layout(height=270, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Tick", yaxis_title=unit)
        return f"Showing last **{cursor - s}** ticks of **{title}**.", fig

    return "Try: current risk, current inefficiency, next maintenance, root cause, anomalies, or 'show last 60 ticks kw/tr trend'.", None


# ============================================================
# Build telemetry for selected asset
# ============================================================
seed = abs(hash(asset)) % (10**6)
x = make_hvac_series(seed, noise=noise, drift=drift, stress=stress)


# ============================================================
# Cursor + autoplay (session-state)
# ============================================================
if "cursor" not in st.session_state:
    st.session_state.cursor = 0
if "last_asset" not in st.session_state:
    st.session_state.last_asset = None

if st.session_state.last_asset != asset:
    st.session_state.last_asset = asset
    st.session_state.cursor = 0

cursor = int(np.clip(st.session_state.cursor, 0, len(x["t"]) - 1))
st.session_state.cursor = cursor

if autoplay:
    st.session_state.cursor = min(st.session_state.cursor + 2, len(x["t"]) - 1)
    time.sleep(tick_ms / 1000.0)
    st.rerun()


# ============================================================
# Current computations
# ============================================================
ineff, maint, rul_hours, findings = compute_scores(
    x, cursor,
    design_kw_per_tr=design_kw_per_tr,
    design_delta_t_chw=design_delta_t_chw,
    design_supply_air=design_supply_air,
    design_ahu_cfm=design_ahu_cfm
)

status_eff = status_from_score(ineff)
status_maint = status_from_score(maint)

next_maint_dt = datetime.now() + timedelta(hours=rul_hours)
next_maint_str = next_maint_dt.strftime("%d %b %Y, %I:%M %p")
confidence = float(np.clip(70 + (x["condenser_fouling"][cursor] * 18) + (x["filter_loading"][cursor] * 10) - (noise * 3), 55, 93))


# ============================================================
# LEFT: video (optional) + summary + quick GenBI
# ============================================================
with left:
    st.subheader("🎥 Live Asset Feed (optional)")
    st.write(f"**Asset:** {asset}")

    if HAS_VIDEO:
        current_video = video_files[st.session_state.video_idx]
        st.write(f"**Now playing:** {current_video.name}")
        st.video(str(current_video))
    else:
        st.markdown(
            "<div class='tight-card'><span class='pill'>No video found</span> "
            "<span class='muted small'>Add MP4s to ./videos to enable live feed.</span></div>",
            unsafe_allow_html=True
        )

    st.markdown('<div class="tight-card">', unsafe_allow_html=True)
    st.markdown("### 📌 Executive Snapshot")
    a, b, c = st.columns(3)
    a.metric("Maintenance State", status_maint)
    b.metric("Maintenance Risk", f"{maint:.0f}/100")
    c.metric("RUL", f"{rul_hours:.0f} hrs")

    d, e, f = st.columns(3)
    d.metric("Efficiency State", status_eff)
    e.metric("Inefficiency", f"{ineff:.0f}/100")
    f.metric("kW/TR", f"{x['kw_per_tr'][cursor]:.2f}")

    st.markdown(f"**Next maintenance window:** {next_maint_str}")
    st.markdown(f"<span class='muted'>Confidence (demo): {confidence:.0f}%</span>", unsafe_allow_html=True)

    st.markdown("#### 🧾 Top Findings (demo)")
    for item in findings[:5]:
        st.write(f"- {item}")

    st.markdown("#### 🔎 GenBI Quick Query")
    quick_q = st.text_input("Ask about risk, inefficiency, trends, anomalies…", placeholder="e.g., show last 60 ticks kw/tr trend")
    st.markdown("</div>", unsafe_allow_html=True)

    quick_answer, quick_fig = genbi_answer(
        quick_q, x, cursor, ineff, maint, rul_hours, next_maint_str, status_eff, status_maint, findings
    ) if quick_q else (None, None)

    if quick_q and quick_answer:
        st.info(quick_answer)
        if quick_fig is not None:
            st.plotly_chart(quick_fig, use_container_width=True)


# ============================================================
# RIGHT: Always-visible KPI rows + Tabs
# ============================================================
with right:
    st.subheader("📟 Chiller + AHU Dashboard")

    r1, r2, r3 = st.columns(3)
    r1.metric("Cooling Load (TR)", f"{x['tr'][cursor]:.0f}")
    r2.metric("Chiller kW", f"{x['chiller_kw'][cursor]:.0f}")
    r3.metric("COP (proxy)", f"{x['cop_proxy'][cursor]:.2f}")

    r4, r5, r6 = st.columns(3)
    r4.metric("kW/TR", f"{x['kw_per_tr'][cursor]:.2f}")
    r5.metric("CHW ΔT (°C)", f"{x['chw_delta_t'][cursor]:.2f}")
    r6.metric("CHW Supply (°C)", f"{x['chw_supply'][cursor]:.1f}")

    r7, r8, r9 = st.columns(3)
    r7.metric("SAT (°C)", f"{x['supply_air'][cursor]:.1f}")
    r8.metric("Airflow (CFM)", f"{x['cfm'][cursor]:.0f}")
    r9.metric("Filter ΔP (proxy)", f"{x['filter_dp'][cursor]:.2f}")

    tabs = st.tabs(["📈 Live Telemetry", "🧠 Agent", "💬 GenBI Query"])

    # ---------------- Tab 1: Live telemetry ----------------
    with tabs[0]:
        window = 140
        start = max(0, cursor - window)

        fig = go.Figure()
        make_line(fig, x["t"][start:cursor + 1], x["kw_per_tr"][start:cursor + 1], "kW/TR", None)
        make_line(fig, x["t"][start:cursor + 1], x["chw_delta_t"][start:cursor + 1], "CHW ΔT (°C)", "y2")
        make_line(fig, x["t"][start:cursor + 1], x["filter_dp"][start:cursor + 1], "Filter ΔP (proxy)", "y3")
        fig.add_vline(x=x["t"][cursor], line_width=2)

        fig.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Telemetry Tick",
            yaxis=dict(title="kW/TR"),
            yaxis2=dict(title="ΔT (°C)", overlaying="y", side="right"),
            yaxis3=dict(title="Filter ΔP", overlaying="y", side="right", position=0.97, showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        colX, colY = st.columns([1, 2])
        with colX:
            if st.button("⏩ Advance telemetry"):
                st.session_state.cursor = min(st.session_state.cursor + 10, len(x["t"]) - 1)
                st.rerun()
        with colY:
            st.progress(int((cursor / (len(x["t"]) - 1)) * 100))

    # ---------------- Tab 2: Agent (risk, RUL, recommendations) ----------------
    with tabs[1]:
        c1, c2 = st.columns(2)

        with c1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rul_hours,
                number={"suffix": " hrs"},
                gauge={"axis": {"range": [0, 240]}, "bar": {"thickness": 0.35}},
                title={"text": "Remaining Useful Life (RUL)"}
            ))
            gauge.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)

        with c2:
            # Risk trend window
            w = 140
            s = max(0, cursor - w)
            maint_series = []
            ineff_series = []
            for i in range(s, cursor + 1):
                ie, ma, _, _ = compute_scores(
                    x, i,
                    design_kw_per_tr=design_kw_per_tr,
                    design_delta_t_chw=design_delta_t_chw,
                    design_supply_air=design_supply_air,
                    design_ahu_cfm=design_ahu_cfm
                )
                maint_series.append(ma)
                ineff_series.append(ie)

            risk_fig = go.Figure()
            risk_fig.add_trace(go.Scatter(x=x["t"][s:cursor + 1], y=maint_series, mode="lines", name="Maintenance Risk"))
            risk_fig.add_trace(go.Scatter(x=x["t"][s:cursor + 1], y=ineff_series, mode="lines", name="Inefficiency"))
            risk_fig.add_hline(y=40, line_width=1)
            risk_fig.add_hline(y=70, line_width=1)
            risk_fig.add_vline(x=x["t"][cursor], line_width=2)
            risk_fig.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Tick",
                yaxis_title="Score (0-100)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(risk_fig, use_container_width=True)

        st.markdown("### 📅 Predicted maintenance window")
        m1, m2, m3 = st.columns(3)
        m1.metric("Next Service Due By", next_maint_str)
        m2.metric("Confidence (demo)", f"{confidence:.0f}%")
        m3.metric("Most Likely Issue (demo)", (
            "Condenser fouling" if x["condenser_fouling"][cursor] > 0.45 else
            "Filter clogging" if x["filter_loading"][cursor] > 0.55 else
            "Cooling coil fouling" if x["coil_fouling"][cursor] > 0.45 else
            "Part-load staging"
        ))

        if status_maint == "ALERT":
            st.error("🚨 Recommendation: Schedule service immediately. Maintenance risk is high; expect escalating inefficiency / downtime risk.")
        elif status_maint == "WATCH":
            st.warning("⚠️ Recommendation: Monitor closely. Plan servicing in the next available window; verify coils/filters/ΔT behavior.")
        else:
            st.success("✅ Recommendation: Operate normally. No near-term service intervention required (within baseline band).")

    # ---------------- Tab 3: Full GenBI ----------------
    with tabs[2]:
        st.markdown("### 💬 GenBI Query")
        st.caption("Ask in plain English. (Rule-based/offline demo; can be upgraded to LLM later.)")

        q = st.text_input("Your question", placeholder="e.g., What is the current risk and why? Or show last 80 ticks filter trend")
        ans, fig = genbi_answer(
            q, x, cursor, ineff, maint, rul_hours, next_maint_str, status_eff, status_maint, findings
        ) if q else (None, None)

        if ans:
            st.info(ans)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
