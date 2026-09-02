import streamlit as st
from PIL import Image
import numpy as np
import hashlib, time
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

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

# Real-time frame gate: separates people from crop imagery before crop analysis.
def human_crop_gate(pil_img):
    if pil_img is None:
        return {'status':'NO FRAME', 'confidence':0, 'reason':'Capture or upload a frame first.', 'detections':[]}
    if YOLO is None:
        return {'status':'VISION MODEL OFFLINE', 'confidence':0, 'reason':'Object detector package is not installed. Demo inference remains available.', 'detections':[]}
    try:
        model = st.session_state.get('yolo_model')
        if model is None:
            model = YOLO('yolo11n.pt')
            st.session_state.yolo_model = model
        result = model.predict(source=np.array(pil_img.convert('RGB')), conf=0.35, verbose=False)[0]
        names = result.names
        detections=[]
        h,w = pil_img.height,pil_img.width
        for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            label=names[int(cls)]
            x1,y1,x2,y2=box
            area=max(0,x2-x1)*max(0,y2-y1)/(w*h)
            detections.append((label,float(conf),float(area),[int(x1),int(y1),int(x2),int(y2)]))
        persons=[d for d in detections if d[0]=='person']
        strongest=max(persons,key=lambda x:x[1],default=None)
        if strongest and (strongest[1] >= 0.50 or strongest[2] >= 0.12):
            return {'status':'HUMAN DETECTED','confidence':round(strongest[1]*100), 'reason':'A person is visible in the frame. Ask the farmer to point the camera only at the crop.', 'detections':detections}
        return {'status':'CROP FRAME','confidence':round((1-max([d[1] for d in persons],default=0))*100), 'reason':'No significant human detected. Safe to continue crop analysis.', 'detections':detections}
    except Exception as e:
        return {'status':'UNCERTAIN','confidence':0,'reason':f'Vision gate could not verify the frame: {type(e).__name__}.', 'detections':[]}

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
    if img:
        st.image(img, caption='Farmer-provided frame', use_container_width=True)
        gate = human_crop_gate(img)
        if gate['status']=='HUMAN DETECTED':
            st.error(f"🚫 {gate['status']} · {gate['confidence']}% confidence\n\n{gate['reason']}")
        elif gate['status']=='CROP FRAME':
            st.success(f"✅ {gate['status']} · {gate['confidence']}% frame confidence\n\n{gate['reason']}")
        elif gate['status']=='VISION MODEL OFFLINE':
            st.warning(f"⚠️ {gate['status']}\n\n{gate['reason']}")
        else:
            st.warning(f"⚠️ {gate['status']}\n\n{gate['reason']}")
    else:
        gate = {'status':'NO FRAME','confidence':0}
    analyse = st.button('🔎 Analyse frame', type='primary', use_container_width=True)

with right:
    st.markdown('<div class="card"><div class="muted">MODEL PIPELINE</div><h3>Vision → Evidence → Decision</h3><p>Image quality → crop identification → condition → severity → visual quality → maturity → shelf-life → market & logistics decision.</p><div class="small">Frame gate uses a real object detector to reject person-heavy frames. Crop/disease scores remain demo inference until a field-trained crop model is connected.</div></div>', unsafe_allow_html=True)

def visual_crop_quality(pil_img):
    """Estimate visible crop condition from actual pixels. This is a screening score, not a disease diagnosis."""
    arr=np.array(pil_img.convert('RGB'))
    if arr.size==0:
        return {'score':0,'grade':'REJECT','reason':'Empty image'}
    gray=cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    brightness=float(gray.mean())
    contrast=float(gray.std())
    blur=float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sat=float(hsv[:,:,1].mean())
    green=((hsv[:,:,0]>30)&(hsv[:,:,0]<95)&(hsv[:,:,1]>45)&(hsv[:,:,2]>45)).mean()
    dark=((gray<55)).mean()
    brown=((hsv[:,:,0]>5)&(hsv[:,:,0]<30)&(hsv[:,:,1]>55)&(hsv[:,:,2]>45)).mean()
    # Image-quality factors derived from the uploaded frame.
    sharp=np.clip((blur-25)/220,0,1)
    light=1-np.clip(abs(brightness-135)/135,0,1)
    contrast_s=np.clip((contrast-18)/55,0,1)
    vegetation=np.clip((green*2.2 + sat/255*0.35),0,1)
    damage=np.clip(brown*1.5 + dark*0.35,0,1)
    score=int(np.clip(100*(0.30*sharp+0.20*light+0.15*contrast_s+0.35*vegetation-0.18*damage),0,100))
    if green < 0.035 and sat < 48:
        grade='NOT A CROP / LOW VEGETATION SIGNAL'
        reason='The frame has very little vegetation/leaf-colour signal. Capture the crop closer and fill the frame.'
    elif score>=78:
        grade='GOOD CROP'
        reason='Strong vegetation signal with usable lighting, contrast and sharpness.'
    elif score>=52:
        grade='FAIR / DAMAGED CROP'
        reason='Crop-like frame detected, but visible damage, darkness, blur or weak vegetation signal reduces quality.'
    else:
        grade='BAD CROP / REJECT'
        reason='The crop frame appears low quality or visibly stressed/damaged. Re-capture a clear close-up before making a sale decision.'
    return {'score':score,'grade':grade,'reason':reason,'metrics':{'brightness':brightness,'blur':blur,'green':green,'brown':brown}}

if analyse:
    if not img:
        st.session_state.analysis=None
        st.warning('Capture or upload a crop image first.')
        st.stop()
    if gate.get('status') == 'HUMAN DETECTED':
        st.session_state.analysis=None
        st.error('Analysis blocked: human detected. Please capture a crop-only frame.')
        st.stop()
    if gate.get('status') in ('VISION MODEL OFFLINE','UNCERTAIN'):
        st.session_state.analysis=None
        st.warning('Frame could not be safely verified. Please retry with a clear crop-only image.')
        st.stop()

    visual=visual_crop_quality(img)
    if visual['grade'] in ('NOT A CROP / LOW VEGETATION SIGNAL','BAD CROP / REJECT'):
        st.session_state.analysis=None
        st.error(f"🚫 {visual['grade']} · {visual['score']}% visual score\n\n{visual['reason']}")
        st.stop()

    # The actual uploaded pixels now drive the visible quality score; deterministic demo economics remain separate.
    seed_text = f'{crop}|{location}|{qty}|{visual["score"]}|{img.size}'
    seed=int(hashlib.sha256(seed_text.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    quality=int(np.clip(visual['score'] + rng.integers(-4,5), 0, 100))
    severity=int(np.clip(100-quality + rng.integers(-5,6), 0, 65))
    health=max(0,100-severity)
    crop_conf=int(np.clip(86 + visual['score']//8 + rng.integers(-2,3), 0, 99))
    readiness=int(np.clip(62 + quality//3 + rng.integers(-4,5), 0, 98))
    shelf=max(1,int(np.clip(1 + quality/22 - severity/35 + rng.integers(-1,2),1,7)))
    imageq=int(np.clip(55 + visual['score']*0.45,0,99))
    price=int(rng.integers(2200,2700)); transport=int(qty*rng.integers(22,36))
    spoil=int(qty*max(1,rng.integers(2,18)+severity//8)); gross=qty*price; net=gross-transport-spoil
    hold_price=price+int(rng.integers(-100,180)); hold_spoil=spoil+int(qty*rng.integers(8,25)); hold_net=qty*hold_price-transport-hold_spoil
    decision='SELL NOW' if net>=hold_net or severity>=35 else 'HOLD 2 DAYS'
    st.session_state.analysis=dict(crop=crop,crop_conf=crop_conf,severity=severity,quality=quality,health=health,readiness=readiness,shelf=shelf,imageq=imageq,price=price,transport=transport,spoil=spoil,gross=gross,net=net,hold_price=hold_price,hold_net=hold_net,decision=decision,location=location,qty=qty,visual_grade=visual['grade'],visual_reason=visual['reason'])

if st.session_state.analysis:
    a=st.session_state.analysis
    st.subheader('🤖 AI Analysis')
    cols=st.columns(6)
    vals=[('Crop confidence',f"{a['crop_conf']}%"),('Health',f"{a['health']}%"),('Visual quality',f"{a['quality']}%"),('Severity',f"{a['severity']}%"),('Harvest readiness',f"{a['readiness']}%"),('Shelf life',f"{a['shelf']} days")]
    for col,(label,val) in zip(cols,vals):
        with col: st.markdown(f'<div class="card"><div class="muted">{label}</div><div class="kpi">{val}</div></div>',unsafe_allow_html=True)

    st.markdown(f"<div class='card'><div class='muted'>REAL IMAGE QUALITY GATE</div><h3>{a['visual_grade']} · {a['quality']}%</h3><p>{a['visual_reason']}</p><div class='small'>Quality score is computed from the uploaded frame's sharpness, lighting, contrast, vegetation signal and visible damage-like colour regions. It is a screening signal, not a medical/agronomic diagnosis.</div></div>", unsafe_allow_html=True)

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
