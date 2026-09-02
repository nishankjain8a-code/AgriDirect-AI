
import io, time, math, hashlib
from datetime import datetime
import numpy as np
import cv2, requests, os
import streamlit as st
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

st.set_page_config(page_title="AgriDirect AI • ORBIT X", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--bg:#020503;--p:#061009;--p2:#09150c;--line:#18321f;--lime:#caff3d;--mint:#62f5c5;--amber:#ffd166;--red:#ff6570;--txt:#edfff2;--muted:#718678}
.stApp{background:#020503;color:var(--txt);background-image:radial-gradient(circle at 80% 8%,rgba(202,255,61,.10),transparent 27%),radial-gradient(circle at 15% 80%,rgba(98,245,197,.055),transparent 25%),linear-gradient(rgba(255,255,255,.012) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.012) 1px,transparent 1px);background-size:auto,auto,40px 40px}
.block-container{max-width:1550px;padding:1rem 1.4rem 4rem} header[data-testid="stHeader"],section[data-testid="stSidebar"]{display:none}*{font-family:'Space Grotesk',sans-serif}
.top{display:flex;justify-content:space-between;align-items:center;padding:8px 2px 14px;border-bottom:1px solid var(--line)}.logo{font-weight:700;font-size:22px}.logo em{color:var(--lime);font-style:normal}.sys{font:500 9px 'DM Mono';letter-spacing:1.5px;color:var(--muted)}.dot{color:var(--lime);text-shadow:0 0 15px var(--lime)}
.hero{position:relative;min-height:350px;border:1px solid #23432a;border-radius:32px;overflow:hidden;padding:34px;background:linear-gradient(135deg,#0a180d,#030704 62%,#0a160c);box-shadow:0 30px 100px rgba(0,0,0,.35)}.hero:before{content:"";position:absolute;inset:-120px -100px auto auto;width:480px;height:480px;border:1px solid rgba(202,255,61,.18);border-radius:50%;box-shadow:0 0 0 45px rgba(202,255,61,.025),0 0 0 100px rgba(202,255,61,.018),0 0 0 170px rgba(202,255,61,.012)}.hero:after{content:"";position:absolute;left:0;right:0;top:0;height:1px;background:linear-gradient(90deg,transparent,var(--lime),transparent);box-shadow:0 0 30px var(--lime);animation:beam 4s linear infinite}@keyframes beam{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.hero h1{font-size:58px;line-height:.96;letter-spacing:-4px;margin:10px 0 15px;max-width:900px}.hero h1 span{color:var(--lime)}.hero p{max-width:820px;color:#9ab0a0;font-size:15px}.pill{display:inline-block;border:1px solid #294d33;background:#07120a;border-radius:999px;padding:6px 10px;margin-right:5px;font:500 9px 'DM Mono';letter-spacing:.6px}.live{color:var(--lime);box-shadow:0 0 20px rgba(202,255,61,.05)}
.card{background:linear-gradient(145deg,rgba(8,18,11,.97),rgba(3,8,5,.98));border:1px solid var(--line);border-radius:20px;padding:18px;box-shadow:0 18px 60px rgba(0,0,0,.17);height:100%}.card:hover{border-color:#335b3d}.label{font:500 9px 'DM Mono';letter-spacing:1.7px;color:var(--muted);text-transform:uppercase}.big{font-size:36px;font-weight:700;letter-spacing:-2px}.mega{font-size:70px;line-height:.9;font-weight:700;letter-spacing:-5px}.lime{color:var(--lime)}.mint{color:var(--mint)}.amber{color:var(--amber)}.red{color:var(--red)}.muted{color:var(--muted);font-size:11px}
.command{border:1px solid #557d3c;border-radius:25px;padding:23px;background:radial-gradient(circle at 80% 20%,rgba(202,255,61,.09),transparent 30%),linear-gradient(135deg,#0e2112,#061009);box-shadow:0 0 70px rgba(202,255,61,.06)}.command h2{font-size:44px;margin:4px 0;color:var(--lime)}
.radar{height:240px;border-radius:20px;border:1px solid #183521;background:radial-gradient(circle,#0e2413 0 1px,transparent 1px);background-size:25px 25px;position:relative;overflow:hidden}.radar:before{content:"";position:absolute;inset:15%;border:1px solid #345b3c;border-radius:50%;box-shadow:0 0 0 35px rgba(202,255,61,.018),0 0 0 70px rgba(202,255,61,.012)}.sweep{position:absolute;left:50%;top:50%;width:46%;height:2px;transform-origin:left center;background:linear-gradient(90deg,var(--lime),transparent);animation:sweep 2.7s linear infinite;box-shadow:0 0 12px var(--lime)}@keyframes sweep{to{transform:rotate(360deg)}}.blip{position:absolute;width:9px;height:9px;border-radius:50%;background:var(--lime);box-shadow:0 0 20px var(--lime);animation:pulse 1.6s infinite}@keyframes pulse{50%{transform:scale(1.8);opacity:.4}}
.trace{display:flex;gap:6px;flex-wrap:wrap;align-items:center}.node{border:1px solid #24452e;background:#061009;padding:7px 9px;border-radius:9px;font:500 8px 'DM Mono';color:#b8cbbd}.arr{color:#48684f}.micro{background:#061009;border:1px solid #173421;border-radius:13px;padding:11px}.micro b{font-size:17px}.bar{height:5px;background:#112018;border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--mint),var(--lime))}
.agent{border-left:3px solid var(--lime);background:#061009;border-radius:12px;padding:11px;margin:7px 0}.agent small{color:var(--muted);font:9px 'DM Mono'}.agent b{display:block;margin:3px 0}.stButton>button{border-radius:13px!important;background:#09160c!important;border:1px solid #284c32!important;min-height:44px;font-weight:700!important}.stButton>button:hover{border-color:var(--lime)!important;box-shadow:0 0 25px rgba(202,255,61,.12)!important}.stProgress>div>div>div>div{background:var(--lime)}.stTabs [aria-selected="true"]{color:var(--lime)!important}
div[data-baseweb="select"]>div,.stNumberInput input{background:#061009!important;border-color:#183521!important}.footer{text-align:center;color:#425747;font:9px 'DM Mono';padding:18px}
</style>
""", unsafe_allow_html=True)

CROPS={"Tomato":("🍅",2400,5,.08),"Onion":("🧅",2900,18,.035),"Potato":("🥔",2100,15,.04),"Cotton":("🌿",6900,25,.02),"Chilli":("🌶️",7200,12,.05),"Wheat":("🌾",2550,45,.015)}
LOCS=["Nashik, Maharashtra","Pune, Maharashtra","Nagpur, Maharashtra","Indore, Madhya Pradesh","Jaipur, Rajasthan","Lucknow, Uttar Pradesh","Karnal, Haryana"]

def money(x): return f"₹{int(round(x)):,}"
def geocode(name):
    try:
        z=requests.get("https://geocoding-api.open-meteo.com/v1/search",params={"name":name.split(",")[0],"count":1},timeout=5).json().get("results",[])
        return z[0] if z else None
    except:return None
def weather(loc):
    g=geocode(loc)
    if not g:return None
    try:
        z=requests.get("https://api.open-meteo.com/v1/forecast",params={"latitude":g["latitude"],"longitude":g["longitude"],"current":"temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m","hourly":"precipitation_probability,temperature_2m","forecast_days":2,"timezone":"auto"},timeout=6).json()
        return {"geo":g,**z}
    except:return None
def wname(code): return {0:"Clear",1:"Mostly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",51:"Drizzle",61:"Rain",63:"Rain",65:"Heavy rain",80:"Showers",81:"Showers",82:"Heavy showers",95:"Thunderstorm"}.get(int(code),"Mixed")

def demo():
    a=np.zeros((720,1100,3),np.uint8);a[:]=[23,54,29];rng=np.random.default_rng(8)
    for x,y,rx,ry in [(220,360,230,170),(550,290,290,175),(870,390,210,150)]:cv2.ellipse(a,(x,y),(rx,ry),-8,0,360,(43,130,50),-1)
    for x,y,r in [(200,320,44),(295,400,52),(510,270,46),(600,350,50),(820,370,43),(910,320,38)]:cv2.circle(a,(x,y),r,(180,212,58),-1)
    a=np.uint8(np.clip(a.astype(np.int16)+rng.normal(0,5,a.shape),0,255));return Image.fromarray(a)

def stats(im):
    a=np.array(im.convert("RGB"));g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
    bright=float(g.mean());contrast=float(g.std());sharp=float(cv2.Laplacian(g,cv2.CV_64F).var());sat=float(hsv[:,:,1].mean())
    green=float(((hsv[:,:,0]>30)&(hsv[:,:,0]<95)&(hsv[:,:,1]>45)&(hsv[:,:,2]>45)).mean())
    brown=float(((hsv[:,:,0]>5)&(hsv[:,:,0]<32)&(hsv[:,:,1]>55)&(hsv[:,:,2]>40)).mean())
    face=0
    try:
        cc=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml");face=len(cc.detectMultiScale(cv2.equalizeHist(g),1.15,5,minSize=(50,50)))
    except:pass
    light=1-np.clip(abs(bright-135)/135,0,1);sh=np.clip((sharp-12)/250,0,1);co=np.clip((contrast-10)/75,0,1);veg=np.clip(green*2.8+sat/255*.2,0,1);dam=np.clip(brown*1.45,0,1)
    score=int(np.clip(100*(.25*sh+.22*light+.18*co+.35*veg-.12*dam),0,100))
    gate="PERSON / NON-CROP" if face>=2 else ("NOT A CROP" if green<.018 and sat<38 else ("GOOD CROP" if score>=78 else ("FAIR CROP" if score>=52 else "RECAPTURE")))
    return dict(score=score,bright=bright,contrast=contrast,sharp=sharp,sat=sat,green=green,brown=brown,faces=face,gate=gate)

def analyze(im,crop,qty,loc,w, buyer_adj=0):
    v=stats(im);seed=int(hashlib.sha256(f"{crop}{qty}{loc}{v['score']}".encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed)
    q=v["score"];severity=int(np.clip(100-q*.55+v["brown"]*80+rng.uniform(-5,5),0,100));health=int(np.clip(100-severity*.72+rng.uniform(-4,4),0,100));ready=int(np.clip(48+q*.45-severity*.22+rng.uniform(-3,3),0,100));shelf=max(1,int(CROPS[crop][2]*(.55+.45*health/100)))
    base=CROPS[crop][1];now=max(1,int(base*rng.uniform(.94,1.09)*(.94+q/900))*(1+buyer_adj/100));d2=max(1,int(now*rng.uniform(.97,1.075)));d5=max(1,int(d2*rng.uniform(.97,1.06)))
    transport=max(250,int(qty*rng.uniform(18,42)));handling=int(qty*rng.uniform(5,12));spoil=int(qty*now*CROPS[crop][3]*(1+(100-health)/180)*.06);net=qty*now-transport-handling-spoil;holdloss=int(qty*now*CROPS[crop][3]*(2+shelf/20));net2=qty*d2-transport-handling-holdloss
    rain=max((w or {}).get("hourly",{}).get("precipitation_probability",[0])[:12] or [0]);risk=int(np.clip(rain*.65+((w or {}).get("current",{}).get("relative_humidity_2m",55)-55)*.4,0,100))
    action="RECAPTURE" if v["gate"] in ("NOT A CROP","PERSON / NON-CROP","RECAPTURE") else ("SELL NOW" if health<48 or shelf<=2 else ("AGGREGATE" if qty>=150 and net2>net*.98 else ("HOLD 2 DAYS" if net2>net*1.025 and risk<70 else "SELL NOW")))
    return {"crop":crop,"qty":qty,"loc":loc,"v":v,"quality":q,"severity":severity,"health":health,"ready":ready,"shelf":shelf,"conf":int(np.clip(78+q*.2,55,99)),"m":{"now":now,"d2":d2,"d5":d5,"net":net,"net2":net2,"transport":transport,"handling":handling,"spoil":spoil},"risk":risk,"rain":int(rain),"action":action,"time":datetime.now().strftime("%d %b %Y • %H:%M"),"image":im}

def heatmap(im):
    a=np.array(im.convert("RGB"));g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);edge=cv2.GaussianBlur(cv2.Laplacian(g,cv2.CV_32F),(0,0),5);h=np.uint8(np.clip(abs(edge)/max(float(np.max(abs(edge))),1)*255,0,255));hm=cv2.applyColorMap(h,cv2.COLORMAP_TURBO);return Image.fromarray(cv2.cvtColor(cv2.addWeighted(a,.58,hm,.42,0),cv2.COLOR_BGR2RGB))
def evidence(im):
    x=im.copy().convert("RGB");d=ImageDraw.Draw(x);w,h=x.size
    for box in [(.10,.12,.44,.72),(.53,.18,.88,.78)]:d.rectangle((int(w*box[0]),int(h*box[1]),int(w*box[2]),int(h*box[3])),outline=(202,255,61),width=max(2,w//180))
    d.text((16,16),"AI EVIDENCE LAYER",(202,255,61));return x
def reason(a):
    return {"RECAPTURE":"Frame gate failed. Better framing gives the engine cleaner evidence.","SELL NOW":"Quality/shelf risk is strong enough that waiting can erase upside.","HOLD 2 DAYS":"The modeled 2-day net realization is higher while weather risk stays manageable.","AGGREGATE":"Lot size makes pooled transport and stronger buyer negotiation economically attractive."}.get(a["action"],"Compare the scenarios before committing.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_detector.pt")

@st.cache_resource(show_spinner=False)
def load_crop_model():
    if YOLO is None or not os.path.exists(MODEL_PATH):
        return None
    try:
        return YOLO(MODEL_PATH)
    except Exception:
        return None

def detect_crop(im):
    model=load_crop_model()
    if model is None:
        return {"available":False,"detections":[],"status":"CUSTOM MODEL NOT LOADED"}
    try:
        arr=np.array(im.convert("RGB"))
        results=model.predict(source=arr, conf=0.35, iou=0.5, verbose=False)
        det=[]
        for r in results:
            names=r.names
            if r.boxes is None: continue
            for b in r.boxes:
                cls=int(b.cls[0]); conf=float(b.conf[0])
                name=names.get(cls,str(cls)) if isinstance(names,dict) else str(cls)
                det.append({"class":name,"confidence":round(conf*100,1),
                            "box":[int(x) for x in b.xyxy[0].tolist()]})
        det.sort(key=lambda x:x["confidence"],reverse=True)
        return {"available":True,"detections":det,"status":"CUSTOM YOLO MODEL ACTIVE"}
    except Exception as e:
        return {"available":True,"detections":[],"status":"MODEL ERROR — FALLBACK ACTIVE","error":str(e)}

def scan_guide(v, det):
    if v["faces"]>=2: return ("PERSON DETECTED","Keep faces/persons out of the frame. Point only at the crop.")
    if v["sharp"]<35: return ("HOLD STEADY","Image looks soft. Hold the phone steady and move slightly closer.")
    if v["bright"]<55: return ("MORE LIGHT","Move into better natural light and avoid strong backlight.")
    if v["bright"]>225: return ("REDUCE GLARE","Tilt the phone slightly to reduce overexposure and reflections.")
    if v["score"]<45: return ("FRAME THE CROP","Center one clear plant/fruit and make it occupy more of the frame.")
    if det["available"] and not det["detections"]: return ("FIND THE CROP","Detector found no crop above threshold. Reframe and scan again.")
    if det["detections"] and det["detections"][0]["confidence"]<55:
        return ("GET CLOSER",f"Possible {det['detections'][0]['class']} detected, but confidence is low. Fill more of the frame.")
    if det["detections"]:
        d=det["detections"][0]
        return ("CROP LOCKED",f"{d['class']} detected at {d['confidence']:.1f}% confidence. Launch the full mission.")
    return ("READY TO SCAN","Frame is usable. Launch the mission for quality, shelf-life and decision analysis.")

def draw_detections(im, det):
    x=im.copy().convert("RGB")
    if not det["detections"]: return x
    d=ImageDraw.Draw(x); w,h=x.size
    for z in det["detections"][:8]:
        x1,y1,x2,y2=z["box"]
        d.rectangle((x1,y1,x2,y2),outline=(202,255,61),width=max(3,w//220))
        top=max(0,y1-26)
        d.rounded_rectangle((x1,top,min(w,x1+220),y1),radius=6,fill=(3,8,5),outline=(202,255,61))
        d.text((x1+8,max(2,top+4)),f"{z['class']} {z['confidence']:.1f}%",fill=(202,255,61))
    return x

# session
for k,v in {"a":None,"w":None,"buyer":0}.items(): st.session_state.setdefault(k,v)
crop=st.selectbox("CROP",list(CROPS),format_func=lambda x:f"{CROPS[x][0]} {x}",key="crop")
q1,q2,q3=st.columns([1,1,1])
with q1:qty=st.number_input("LOT • QUINTAL",1,10000,50)
with q2:loc=st.selectbox("FIELD LOCATION",LOCS)
with q3:mode=st.selectbox("NETWORK",["GOOD","LOW BANDWIDTH","OFFLINE SYNC"])
if st.session_state.w is None or st.session_state.w and st.session_state.w["geo"]["name"]!=loc.split(",")[0]:st.session_state.w=weather(loc)
w=st.session_state.w

st.markdown('<div class="top"><div class="logo">Agri<em>Direct</em> <span style="color:#5b6d60">/ ORBIT X</span></div><div class="sys"><span class="dot">●</span> 7 AGENTS ONLINE &nbsp; • &nbsp; FIELD DECISION CORE</div></div>',unsafe_allow_html=True)

tabs=st.tabs(["🌐 COMMAND","📷 VISION","🤖 AGENT WAR ROOM","⏳ TIME MACHINE","💰 NEGOTIATOR","📡 MARKET TWIN","⚠ RISK","🚚 EXECUTION","🧾 PASSPORT"])

with tabs[0]:
    st.markdown('<div class="hero"><span class="pill live">● LIVE FIELD SYSTEM</span><span class="pill">MULTI-AGENT</span><span class="pill">COUNTERFACTUAL AI</span><span class="pill">FARMER DIGITAL TWIN</span><h1>One photo.<br><span>Seven minds. One move.</span></h1><p>AgriDirect ORBIT X turns crop evidence into a transparent action by combining vision, quality, shelf-life, weather, market scenarios, logistics and negotiation logic.</p></div>',unsafe_allow_html=True)
    st.write("")
    if st.session_state.a:
        a=st.session_state.a;m=a["m"];l,r=st.columns([1.1,.9])
        with l:st.markdown(f'<div class="command"><div class="label">FARMER DIGITAL TWIN • RECOMMENDATION</div><h2>{a["action"]}</h2><div class="mega">{money(m["net"])}</div><div class="muted">expected net • {qty} quintal • {a["shelf"]}-day shelf window</div><hr><b>WHY?</b><p class="muted">{reason(a)}</p></div>',unsafe_allow_html=True)
        with r:st.markdown(f'<div class="radar"><div class="sweep"></div><div class="blip" style="left:32%;top:34%"></div><div class="blip" style="left:69%;top:59%"></div><div class="blip" style="left:54%;top:25%"></div><div style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);text-align:center"><div class="label">FIELD PULSE</div><div class="big lime">{a["health"]}%</div><div class="muted">health signal</div></div></div>',unsafe_allow_html=True)
        st.markdown('<div class="trace"><span class="node">CAMERA</span><span class="arr">›</span><span class="node">VISION</span><span class="arr">›</span><span class="node">QUALITY</span><span class="arr">›</span><span class="node">SHELF</span><span class="arr">›</span><span class="node">WEATHER</span><span class="arr">›</span><span class="node">MARKET</span><span class="arr">›</span><span class="node">NETBACK</span><span class="arr">›</span><span class="node">ACTION</span></div>',unsafe_allow_html=True)
    else:st.markdown('<div class="card"><div class="label">MISSION READY</div><h3>Start a field mission.</h3><p class="muted">Open Vision → Camera/Upload/Synthetic Demo → launch the inspection.</p></div>',unsafe_allow_html=True)
    st.write("")
    if w:
        cur=w["current"];cs=st.columns(5)
        for col,(lab,val,sub) in zip(cs,[("TEMP",f'{cur.get("temperature_2m","—")}°C',wname(cur.get("weather_code",0))),("HUMIDITY",f'{cur.get("relative_humidity_2m","—")}%',"live"),("RAIN",f'{cur.get("precipitation","—")} mm',"live"),("WIND",f'{cur.get("wind_speed_10m","—")} km/h',"10m"),("LOCATION",w["geo"]["name"],"weather adapter")]):
            with col:st.markdown(f'<div class="card"><div class="label">{lab}</div><div class="big">{val}</div><div class="muted">{sub}</div></div>',unsafe_allow_html=True)

with tabs[1]:
    st.markdown("## Vision Lab <span class='muted'>/ evidence first</span>",unsafe_allow_html=True)
    l,r=st.columns([1.1,.9]);im=None
    with l:
        src=st.radio("INPUT",["📷 Camera","📁 Upload","🧪 Synthetic Demo"],horizontal=True)
        if src=="📷 Camera":
            f=st.camera_input("Frame the crop — avoid faces, keep produce large")
            if f:im=Image.open(f)
        elif src=="📁 Upload":
            f=st.file_uploader("JPG / PNG / WEBP",type=["jpg","jpeg","png","webp"])
            if f:im=Image.open(f)
        else:im=demo()
        if im:
            preview=im.copy(); preview.thumbnail((1200,800))
            v=stats(im); det=detect_crop(im)
            guide_title,guide_text=scan_guide(v,det)
            st.image(draw_detections(preview,det),use_container_width=True)
            cs=st.columns(4)
            for col,(lab,val) in zip(cs,[("FRAME",v["score"]),("SHARPNESS",int(v["sharp"])),("GREEN SIGNAL",f'{v["green"]*100:.1f}%'),("FACE GATE",v["faces"])]):
                with col:st.markdown(f'<div class="micro"><div class="label">{lab}</div><b>{val}</b></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="agent"><small>ORBIT SCAN AGENT • LIVE GUIDANCE</small><b>{guide_title}</b><span class="muted">{guide_text}</span></div>',unsafe_allow_html=True)
            if det["available"] and det["detections"]:
                d0=det["detections"][0]
                st.caption(f"Model: {det['status']} • Top crop: {d0['class']} • confidence {d0['confidence']:.1f}%")
            elif det["available"]:
                st.caption(f"Model: {det['status']}")
            else:
                st.caption("Model: training-ready adapter • add models/crop_detector.pt to activate custom crop detection.")
        if st.button("⚡ LAUNCH 8-STEP MISSION",type="primary",use_container_width=True):
            if not im:st.warning("Capture/upload an image first.")
            else:
                bar=st.progress(0,text="Booting…")
                stages=["Frame integrity","Human / non-crop gate","Pixel feature extraction","Quality evidence","Shelf-life estimate","Weather fusion","Market counterfactuals","Action synthesis"]
                for i,s in enumerate(stages,1):time.sleep(.07);bar.progress(i/8,text=f"{i:02d}/08  {s}")
                st.session_state.a=analyze(im,crop,qty,loc,w);st.session_state.a['detector']=detect_crop(im);st.toast("ORBIT DECISION READY",icon="🌱")
    with r:
        st.markdown('<div class="card"><div class="label">VISION CORE</div><h3>Observable evidence</h3><p class="muted">The system measures image pixels for framing, brightness, contrast, sharpness, vegetation and visible color stress. This is a screening pipeline, not a production disease diagnosis.</p></div>',unsafe_allow_html=True)
        if st.session_state.a:
            a=st.session_state.a
            e1,e2=st.columns(2)
            with e1:st.image(evidence(a["image"]),caption="Region evidence")
            with e2:st.image(heatmap(a["image"]),caption="Texture / edge stress map")
            st.progress(a["quality"]/100,text=f'Visual quality • {a["quality"]}%')
            st.progress(a["health"]/100,text=f'Health signal • {a["health"]}%')
            st.progress(a["ready"]/100,text=f'Readiness • {a["ready"]}%')

with tabs[2]:
    st.markdown("## 🤖 Agent War Room <span class='muted'>/ independent opinions → final vote</span>",unsafe_allow_html=True)
    a=st.session_state.a
    if not a:st.info("Run Vision first.")
    else:
        m=a["m"];agents=[
            ("👁 VISION AGENT","SELL" if a["quality"]<70 else "HOLD","Frame quality + visible stress"),
            ("📈 MARKET AGENT","HOLD" if m["d2"]>m["now"]*1.025 else "SELL","2-day price scenario"),
            ("🌦 WEATHER AGENT","SELL" if a["risk"]>65 else "HOLD","Rain / humidity risk"),
            ("📦 SHELF AGENT","SELL" if a["shelf"]<=2 else "HOLD","Shelf window"),
            ("🚚 LOGISTICS AGENT","AGGREGATE" if qty>=150 else "SELL","Lot-scale economics"),
            ("🤝 NEGOTIATION AGENT","NEGOTIATE","Buyer alternative / floor"),
            ("🧠 DECISION AGENT",a["action"],"Weighted final synthesis")]
        for name,vote,why in agents:st.markdown(f'<div class="agent"><small>{name}</small><b>{vote}</b><span class="muted">{why}</span></div>',unsafe_allow_html=True)
        st.write("");st.markdown(f'<div class="command"><div class="label">CONSENSUS</div><h2>{a["action"]}</h2><p class="muted">Agents are advisory modules in this prototype; their outputs are derived from the same transparent evidence and scenario engine.</p></div>',unsafe_allow_html=True)

with tabs[3]:
    st.markdown("## ⏳ Time Machine <span class='muted'>/ replay the future</span>",unsafe_allow_html=True)
    a=st.session_state.a
    if not a:st.info("Run Vision first.")
    else:
        m=a["m"];days=st.slider("JUMP FORWARD",0,7,2);loss=min(35,days*(3+(100-a["health"])/25));future=m["now"]*(1+.018*days)*(1-loss/100-a["risk"]*days*.004);net=max(0,a["qty"]*future-m["transport"]-m["handling"]-m["spoil"]*(1+days*.35));delta=net-m["net"]
        c=st.columns(4)
        for col,(lab,val,sub) in zip(c,[("NOW",money(m["net"]),"net"),(f"+{days}D",money(net),"counterfactual"),("DELTA",money(delta),"gain / loss"),("QUALITY DRIFT",f"-{loss:.1f}%","modeled")]):
            with col:st.markdown(f'<div class="card"><div class="label">{lab}</div><div class="big">{val}</div><div class="muted">{sub}</div></div>',unsafe_allow_html=True)
        st.progress(max(0,min(1,(a["health"]-loss)/100)),text=f'Projected quality • {max(0,a["health"]-loss):.0f}%')
        (st.success if delta>0 else st.warning)(f'{("Waiting could add " if delta>0 else "Waiting could cost ")}{money(abs(delta))} in modeled net realization.')

with tabs[4]:
    st.markdown("## 🤝 Buyer Negotiation Copilot <span class='muted'>/ protect farmer realization</span>",unsafe_allow_html=True)
    a=st.session_state.a
    if not a:st.info("Run Vision first.")
    else:
        offer=st.number_input("BUYER OFFER • ₹ / QUINTAL",100,100000,int(a["m"]["now"]*.95),100)
        base=a["m"]["now"];floor=max(1,base*.93);counter=max(floor,base*1.04)
        netoffer=a["qty"]*offer-a["m"]["transport"]-a["m"]["handling"]-a["m"]["spoil"]
        cols=st.columns(3)
        for col,(lab,val,sub) in zip(cols,[("OFFER",money(offer),"buyer"),("AI COUNTER",money(counter),"suggested"),("WALK-AWAY FLOOR",money(floor),"modeled")]):
            with col:st.markdown(f'<div class="card"><div class="label">{lab}</div><div class="mega">{val}</div><div class="muted">{sub}</div></div>',unsafe_allow_html=True)
        verdict="ACCEPT" if offer>=base*1.03 else ("NEGOTIATE" if offer>=floor else "WALK AWAY")
        st.markdown(f'<div class="command"><div class="label">COPILOT VERDICT</div><h2>{verdict}</h2><p class="muted">Suggested counter: {money(counter)}/q • modeled net at offer: {money(netoffer)}</p></div>',unsafe_allow_html=True)
        st.text_area("COPY COUNTER-OFFER",f"Current lot: {a['qty']} quintal. Based on the modeled quality and alternatives, I can offer at {money(counter)}/quintal. Please confirm pickup terms and payment timing.",height=90)

with tabs[5]:
    st.markdown("## 📡 Market Twin <span class='muted'>/ transparent sandbox</span>",unsafe_allow_html=True)
    a=st.session_state.a
    if not a:st.info("Run Vision first.")
    else:
        m=a["m"];c=st.columns(4)
        for col,(lab,val,sub) in zip(c,[("SPOT",money(m["now"]),"scenario/q"),("+2D",money(m["d2"]),"scenario/q"),("+5D",money(m["d5"]),"scenario/q"),("NETBACK",money(m["net"]),"after modeled costs")]):
            with col:st.markdown(f'<div class="card"><div class="label">{lab}</div><div class="big">{val}</div><div class="muted">{sub}</div></div>',unsafe_allow_html=True)
        st.warning("Market values are deterministic demonstration scenarios, not live mandi quotations. Replace the adapter with a verified mandi/market source for production.")
        adj=st.slider("BUYER PRICE SHOCK", -15,15,0);offer=m["now"]*(1+adj/100);net=a["qty"]*offer-m["transport"]-m["handling"]-m["spoil"]
        st.markdown(f'<div class="command"><div class="label">SCENARIO NETBACK</div><div class="mega">{money(net)}</div><div class="muted">at {money(offer)}/quintal</div></div>',unsafe_allow_html=True)

with tabs[6]:
    st.markdown("## ⚠ Risk Radar <span class='muted'>/ weather shock + visual stress</span>",unsafe_allow_html=True)
    a=st.session_state.a
    if not a:st.info("Run Vision first.")
    else:
        c=st.columns(3)
        for col,lab,val,sub in [(c[0],"WEATHER",a["risk"],f'rain next 12h {a["rain"]}%'),(c[1],"VISIBLE STRESS",a["severity"],"image-derived"),(c[2],"SHELF WINDOW",a["shelf"],"days")]:
            with col:st.markdown(f'<div class="card"><div class="label">{lab}</div><div class="mega">{val}{"%" if lab!="SHELF WINDOW" else ""}</div><div class="muted">{sub}</div></div>',unsafe_allow_html=True)
        shock=st.toggle("🌧 SIMULATE WEATHER SHOCK")
        if shock:st.error("Shock mode: assume a severe rain event. Re-check net realization and transport before dispatch.")
        if w:
            hp=w.get("hourly",{}).get("precipitation_probability",[])[:12]
            if hp:st.area_chart(hp,use_container_width=True)

with tabs[7]:
    st.markdown("## 🚚 Execution Graph <span class='muted'>/ from decision to movement</span>",unsafe_allow_html=True)
    a=st.session_state.a
    if not a:st.info("Run Vision first.")
    else:
        routes=[("🚜","LOCAL PICKUP",a["qty"]*30,"fast"),("🚚","POOLED / FPO",a["qty"]*20,"best at scale"),("🏪","BUYER COLLECTION",a["qty"]*23,"balanced")]
        cs=st.columns(3)
        for col,(ic,name,cost,tag) in zip(cs,routes):
            with col:st.markdown(f'<div class="card"><div style="font-size:35px">{ic}</div><div class="label">{name}</div><div class="big">{money(cost)}</div><span class="pill">{tag}</span><p class="muted">Modeled transport cost.</p></div>',unsafe_allow_html=True)
        st.write("");st.markdown(f'<div class="command"><div class="label">EXECUTION PLAN</div><h2>{"AGGREGATE + NEGOTIATE" if a["qty"]>=150 else "COMPARE 3 OFFERS"}</h2><p class="muted">{a["qty"]} quintal • {a["loc"]} • compare net realization before dispatch.</p></div>',unsafe_allow_html=True)

with tabs[8]:
    st.markdown("## 🧾 Decision Passport <span class='muted'>/ auditable share card</span>",unsafe_allow_html=True)
    a=st.session_state.a
    if not a:st.info("No completed mission.")
    else:
        m=a["m"]
        report=f"""AGRIDIRECT AI • ORBIT X
DECISION PASSPORT
{a["time"]}

FARMER DIGITAL TWIN
Crop: {a["crop"]}
Lot: {a["qty"]} quintal
Location: {a["loc"]}

VISION
Gate: {a["v"]["gate"]}
Visual quality: {a["quality"]}%
Confidence: {a["conf"]}%
Health signal: {a["health"]}%
Visible stress: {a["severity"]}%
Readiness: {a["ready"]}%
Shelf window: {a["shelf"]} days

DECISION
Action: {a["action"]}
Reason: {reason(a)}

ECONOMICS
Spot scenario: {money(m["now"])}/q
Expected net now: {money(m["net"])}
2-day net scenario: {money(m["net2"])}
Transport: {money(m["transport"])}
Handling: {money(m["handling"])}
Estimated loss: {money(m["spoil"])}

RISK
Weather risk: {a["risk"]}%
Rain probability next 12h: {a["rain"]}%

TRANSPARENCY
Weather: Open-Meteo adapter when available.
Market: deterministic demonstration scenarios.
Vision: OpenCV image-pixel screening.
Disease diagnosis: NOT CLAIMED.
"""
        st.code(report);st.download_button("⬇ EXPORT DECISION PASSPORT",report.encode(),file_name="AgriDirect_ORBIT_X_Passport.txt",mime="text/plain",use_container_width=True)

st.markdown('<div class="footer">AGRIDIRECT AI • ORBIT X • MULTI-AGENT DECISION INTELLIGENCE • REAL IMAGE PROCESSING • LIVE WEATHER ADAPTER • TRANSPARENT MARKET SIMULATION</div>',unsafe_allow_html=True)
