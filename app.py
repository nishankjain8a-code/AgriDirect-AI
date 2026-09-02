import streamlit as st
from PIL import Image
import numpy as np
import hashlib, time

st.set_page_config(page_title='AgriDirect AI', page_icon='🌾', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
:root{--bg:#07120d;--panel:#0d1c14;--panel2:#10251a;--text:#eaf5ee;--muted:#9ab2a2;--lime:#b7ff5a;--line:#20372a}
.stApp{background:linear-gradient(135deg,#06100b 0%,#091810 55%,#0c1b13 100%);color:var(--text)}
.block-container{padding-top:1.2rem;max-width:1450px}.brand{font-size:28px;font-weight:800;letter-spacing:-1px}.brand span{color:var(--lime)}
.badge{display:inline-block;padding:5px 10px;border:1px solid #31543c;border-radius:999px;color:#cfe9d7;font-size:12px;background:#0b1911}.hero{padding:20px 22px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(120deg,#0d2015,#0b1811);margin-bottom:18px}
.card{background:rgba(13,28,20,.92);border:1px solid var(--line);border-radius:16px;padding:18px;height:100%;box-shadow:0 8px 30px rgba(0,0,0,.16)}
.kpi{font-size:30px;font-weight:800;margin:4px 0}.muted{color:var(--muted);font-size:13px}.good{color:var(--lime)}.warn{color:#ffd166}.danger{color:#ff8b8b}
.decision{border:1px solid #496f35;background:linear-gradient(120deg,#142a17,#0d1b12);border-radius:18px;padding:22px}.decision h1{font-size:40px;margin:0;color:var(--lime)}
.receipt{border-left:3px solid var(--lime);padding:12px 16px;background:#0a1710;border-radius:10px}.small{font-size:12px;color:var(--muted)}
.stButton>button{border-radius:10px;font-weight:700}.stTabs [data-baseweb="tab"]{font-weight:700}
</style>''', unsafe_allow_html=True)

if 'analysis' not in st.session_state: st.session_state.analysis=None
if 'scenario' not in st.session_state: st.session_state.scenario='Tomato — Nashik'

with st.sidebar:
    st.markdown('<div class="brand">Agri<span>Direct</span> AI</div>', unsafe_allow_html=True)
    st.caption('Decision intelligence for Indian farmers')
    st.divider()
    st.radio('Navigation',['Command Center','Crop Scanner','Market Intelligence','Decision Receipt','Model Monitor'], label_visibility='collapsed')
    st.divider()
    st.markdown('**SIH Demo Mode**')
    demo = st.toggle('Simulation enabled', True)
    st.info('Demo values are simulated unless connected to live market/weather/buyer APIs.')

st.markdown('<div class="hero"><span class="badge">● SIH LIVE SIMULATOR</span><h1 style="margin:10px 0 4px">From crop image to a money decision.</h1><div class="muted">AI crop intelligence + market signals + logistics + storage + buyer/FPO options in one farmer-first workflow.</div></div>', unsafe_allow_html=True)

# farmer context
c1,c2,c3,c4 = st.columns(4)
with c1: crop = st.selectbox('Crop',['Tomato','Onion','Potato','Cotton','Chilli','Wheat'], index=0)
with c2: qty = st.number_input('Quantity (quintal)', 1, 1000, 50)
with c3: location = st.selectbox('Location',['Nashik, Maharashtra','Pune, Maharashtra','Nagpur, Maharashtra','Indore, Madhya Pradesh','Jaipur, Rajasthan'])
with c4: network = st.selectbox('Network',['Good','Low bandwidth','Offline / sync later'])

st.subheader('🌿 Crop Scanner')
left,right = st.columns([1.05,1])
with left:
    source = st.radio('Input',['Camera','Upload','Demo scenario'], horizontal=True)
    img = None
    if source=='Camera':
        cam = st.camera_input('Capture crop image')
        if cam: img=Image.open(cam)
    elif source=='Upload':
        up=st.file_uploader('Upload crop image', type=['jpg','jpeg','png','webp'])
        if up: img=Image.open(up)
    else:
        st.success('Demo scenario ready — use the Analyse button to simulate the complete pipeline.')
    if img: st.image(img, caption='Farmer-provided crop frame', use_container_width=True)
    analyse = st.button('🔎 Analyse frame', type='primary', use_container_width=True)

with right:
    st.markdown('<div class="card"><div class="muted">MODEL PIPELINE</div><h3>Vision → Evidence → Decision</h3><p>Image quality → crop identification → condition → severity → visual quality → maturity → shelf-life → market & logistics decision.</p><div class="small">Production note: the SIH simulator uses deterministic demo inference when a field-trained checkpoint/API is not connected.</div></div>', unsafe_allow_html=True)

if analyse:
    seed_text = f'{crop}|{location}|{qty}' + (str(img.size) if img else 'demo')
    seed=int(hashlib.sha256(seed_text.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    crop_conf=int(rng.integers(91,98)); severity=int(rng.integers(5,22)); quality=int(rng.integers(82,94)); health=100-severity
    readiness=int(rng.integers(78,94)); shelf=int(rng.integers(2,6)); imageq=int(rng.integers(88,98))
    price=int(rng.integers(2200,2700)); transport=int(qty*rng.integers(22,36)); spoil=int(qty*rng.integers(0,18)); gross=qty*price; net=gross-transport-spoil
    hold_price=price+int(rng.integers(-100,180)); hold_spoil=spoil+int(qty*rng.integers(8,25)); hold_net=qty*hold_price-transport-hold_spoil
    decision='SELL NOW' if net>=hold_net else 'HOLD 2 DAYS'
    if network!='Good': decision='SELL NOW' if net>=hold_net else 'HOLD 2 DAYS'
    st.session_state.analysis=dict(crop=crop,crop_conf=crop_conf,severity=severity,quality=quality,health=health,readiness=readiness,shelf=shelf,imageq=imageq,price=price,transport=transport,spoil=spoil,gross=gross,net=net,hold_price=hold_price,hold_net=hold_net,decision=decision,location=location,qty=qty)

if st.session_state.analysis:
    a=st.session_state.analysis
    st.subheader('🤖 AI Analysis')
    cols=st.columns(6)
    vals=[('Crop confidence',f"{a['crop_conf']}%"),('Health',f"{a['health']}%"),('Visual quality',f"{a['quality']}%"),('Severity',f"{a['severity']}%"),('Harvest readiness',f"{a['readiness']}%"),('Shelf life',f"{a['shelf']} days")]
    for col,(label,val) in zip(cols,vals):
        with col: st.markdown(f'<div class="card"><div class="muted">{label}</div><div class="kpi">{val}</div></div>',unsafe_allow_html=True)

    st.subheader('🧠 AgriDirect Decision Engine')
    st.markdown(f'''<div class="decision"><div class="muted">RECOMMENDATION FOR {a['qty']} q • {a['location']}</div><h1>{a['decision']}</h1><p>Expected net realization: <b>₹{a['net']:,}</b> &nbsp; | &nbsp; Simulated market: <b>₹{a['price']:,}/q</b></p><div class="muted">Alternative: Hold 2 days → estimated ₹{a['hold_net']:,}</div></div>''', unsafe_allow_html=True)

    t1,t2,t3,t4=st.tabs(['Why this decision','Market intelligence','Execution options','Decision receipt'])
    with t1:
        x,y=st.columns(2)
        with x:
            st.markdown('**Visual evidence**')
            st.progress(a['quality']/100, text=f"Visual quality {a['quality']}%")
            st.progress(a['readiness']/100, text=f"Harvest readiness {a['readiness']}%")
            st.markdown('• Size uniformity indicates sale-ready lot\n\n• Surface damage is limited\n\n• Shelf-life window is short enough to penalize waiting')
        with y:
            st.markdown('**Economic evidence**')
            st.write(f"Gross realization: ₹{a['gross']:,}")
            st.write(f"Transport & handling: −₹{a['transport']:,}")
            st.write(f"Estimated spoilage risk: −₹{a['spoil']:,}")
            st.write(f"**Expected net: ₹{a['net']:,}**")
    with t2:
        st.metric('Simulated current price',f"₹{a['price']:,}/q", 'Demo')
        st.metric('2-day scenario',f"₹{a['hold_price']:,}/q", 'Simulation')
        st.warning('These values are intentionally labelled simulated. Connect mandi/eNAM/APMC sources before claiming live market intelligence.')
    with t3:
        st.write('**1. Direct buyer** — fastest realization, buyer reliability 94% (demo).')
        st.write('**2. FPO aggregation** — combine volume for better negotiation.')
        st.write('**3. Hold** — only when expected price gain exceeds storage/spoilage risk.')
        st.write('**4. Logistics** — compare transport cost before accepting a buyer.')
    with t4:
        st.markdown(f'''<div class="receipt"><b>AGRIDIRECT DECISION RECEIPT</b><br><br>Crop: {a['crop']}<br>Location: {a['location']}<br>Quantity: {a['qty']} q<br>AI crop confidence: {a['crop_conf']}%<br>Visual quality: {a['quality']}%<br>Shelf life: {a['shelf']} days<br><br><b>Recommendation: {a['decision']}</b><br>Expected net realization: ₹{a['net']:,}<br><span class="small">Generated in SIH simulation mode • {time.strftime('%Y-%m-%d %H:%M')}</span></div>''', unsafe_allow_html=True)
else:
    st.info('Choose Camera / Upload / Demo scenario and click **Analyse frame** to run the simulation.')

st.divider()
st.caption('AgriDirect AI — SIH demonstration prototype. AI, market, logistics and buyer figures shown in simulation mode are not guarantees or live quotations.')
