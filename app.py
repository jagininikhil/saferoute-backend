"""
app.py v3 - SafeRoute AI Backend
Auto-injects time-of-day + precipitation features on every predict call
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, os, datetime, random, requests

app  = Flask(__name__)
CORS(app, origins="*")
BASE = os.path.dirname(__file__)

model, feature_names, encoder = None, None, None
try:
    model = joblib.load(os.path.join(BASE,'accident_model.pkl'))
    print("✅ Model loaded")
except: print("⚠  Run train_model_v3.py first")

try:
    feature_names = joblib.load(os.path.join(BASE,'model_features.pkl'))
    print(f"✅ Features: {feature_names}")
except:
    feature_names = ['temperature','humidity','visibility','wind_speed','weather','junction','traffic_signal']
    print("⚠  Using 7 base features")

try:
    encoder = joblib.load(os.path.join(BASE,'weather_encoder.pkl'))
    print("✅ Encoder loaded")
except: print("⚠  Encoder not found")

RISK = {
    1:("Low",      "#34A853","Road conditions safe.",         "Normal speed"),
    2:("Moderate", "#FBBC04","Drive carefully.",              "Reduce 10-15%"),
    3:("High",     "#EA4335","High risk. Reduce speed.",      "Reduce 25-30%"),
    4:("Severe",   "#9C27B0","Severe risk. Alternate route.", "Avoid if possible"),
}

def get_time_features():
    now=datetime.datetime.now()
    h=now.hour; dow=now.weekday()
    return {
        'hour':        h,
        'is_night':    1 if (h>=22 or h<=5) else 0,
        'is_peak':     1 if ((7<=h<=9 or 17<=h<=20) and dow<5) else 0,
        'is_weekend':  1 if dow>=5 else 0,
        'is_dawn_dusk':1 if (5<=h<=7 or 18<=h<=20) else 0,
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data=request.get_json(force=True)
        tf=get_time_features()
        all_vals={
            'temperature':   float(data.get('temperature',72)),
            'humidity':      float(data.get('humidity',60)),
            'visibility':    float(data.get('visibility',8)),
            'wind_speed':    float(data.get('wind_speed',5)),
            'weather':       float(data.get('weather',0)),
            'junction':      float(data.get('junction',0)),
            'traffic_signal':float(data.get('traffic_signal',0)),
            'precipitation': float(data.get('precipitation',0.0)),
            'pressure':      float(data.get('pressure',29.9)),
            'wind_chill':    float(data.get('wind_chill',data.get('temperature',72))),
            'crossing':      float(data.get('crossing',0)),
            'stop':          float(data.get('stop',0)),
            'give_way':      float(data.get('give_way',0)),
            'railway':       float(data.get('railway',0)),
            'roundabout':    float(data.get('roundabout',0)),
            'turning_loop':  float(data.get('turning_loop',0)),
            'amenity':       float(data.get('amenity',0)),
            'traffic_density':float(data.get('traffic_density',0.3)),
            **tf,
        }
        if model and feature_names:
            row=[all_vals.get(f,0.0) for f in feature_names]
            sv=int(model.predict(np.array([row]))[0])
            sv=max(1,min(4,sv))
            if all_vals['traffic_density']>0.8 and sv<4: sv=min(4,sv+1)
            if all_vals['is_night']==1 and sv==2: sv=3
        else:
            sv=_rule(all_vals)
        lv,color,msg,adv=RISK[sv]
        reasons=[]
        if all_vals['precipitation']>0.1: reasons.append(f"Rain {all_vals['precipitation']:.1f}in")
        if all_vals['visibility']<5:      reasons.append(f"Low visibility {all_vals['visibility']}mi")
        if all_vals['wind_speed']>20:     reasons.append(f"High wind {all_vals['wind_speed']}mph")
        if all_vals['is_night']:          reasons.append("Night driving")
        if all_vals['is_peak']:           reasons.append("Rush hour")
        if all_vals['is_dawn_dusk']:      reasons.append("Sun glare risk")
        if all_vals['traffic_density']>0.65: reasons.append("Heavy traffic")
        return jsonify({"severity":sv,"risk_level":lv,"color":color,
                        "message":msg,"speed_advisory":adv,"reasons":reasons,
                        "time_context":{"hour":tf['hour'],"is_night":bool(tf['is_night']),
                                        "is_peak":bool(tf['is_peak']),"is_weekend":bool(tf['is_weekend'])}})
    except Exception as e:
        return jsonify({"error":str(e)}),400

def _rule(v):
    sc=0
    t=v['temperature']; vis=v['visibility']; w=v['wind_speed']
    p=v['precipitation']; pr=v['pressure']
    if t>95 or t<15: sc+=2
    elif t>85 or t<32: sc+=1
    if vis<1: sc+=4
    elif vis<3: sc+=3
    elif vis<6: sc+=1.5
    if w>40: sc+=3
    elif w>25: sc+=2
    elif w>15: sc+=1
    if p>0.5: sc+=2.5
    elif p>0.1: sc+=1.5
    elif p>0: sc+=0.5
    if pr<29.0: sc+=1.5
    elif pr<29.5: sc+=0.5
    if v['weather']==1: sc+=1.5
    if v['junction']==1: sc+=1
    if v['traffic_signal']==1: sc+=0.5
    if v['is_night']==1: sc+=1.5
    if v['is_peak']==1: sc+=1
    if v['is_dawn_dusk']==1: sc+=0.8
    td=v.get('traffic_density',0.3)
    if td>0.8: sc+=2
    elif td>0.6: sc+=1
    elif td>0.4: sc+=0.5
    return 1 if sc<2 else 2 if sc<4 else 3 if sc<6 else 4

@app.route('/traffic')
def traffic():
    tf=get_time_features()
    base=0.72 if tf['is_peak'] else 0.22 if tf['is_night'] else 0.45
    density=min(1.0,max(0.0,base+(random.random()-.5)*.18))
    level="Heavy" if density>0.65 else "Moderate" if density>0.4 else "Free"
    color="#EA4335" if density>0.65 else "#FBBC04" if density>0.4 else "#34A853"
    return jsonify({"density":round(density,2),"level":level,"color":color,
                    "is_peak":bool(tf['is_peak']),"is_night":bool(tf['is_night'])})

@app.route('/routes')
def get_routes():
    lat=request.args.get('lat',type=float)
    lng=request.args.get('lng',type=float)
    dest=request.args.get('dest','')
    if not lat or not lng or not dest:
        return jsonify({"routes":[],"error":"Missing params"}),400
    try:
        geo=requests.get(f"https://nominatim.openstreetmap.org/search?q={dest}&format=json&limit=1",
            headers={"User-Agent":"SafeRouteAI/1.0"},timeout=6).json()
        if not geo: return jsonify({"routes":[],"error":"Location not found"})
        dlat,dlng=float(geo[0]['lat']),float(geo[0]['lon'])
    except Exception as e: return jsonify({"routes":[],"error":str(e)})
    try:
        osrm=requests.get(
            f"https://router.project-osrm.org/route/v1/driving/{lng},{lat};{dlng},{dlat}"
            f"?overview=full&alternatives=true&geometries=geojson",timeout=8).json()
        raw=osrm.get('routes',[])
    except Exception as e: return jsonify({"routes":[],"error":str(e)})
    tf=get_time_features()
    result=[]
    for i,rt in enumerate(raw[:2]):
        km=round(rt['distance']/1000,1); eta=round(rt['duration']/60)
        risk=random.choice(["Low","Moderate"]) if not tf['is_peak'] else random.choice(["Moderate","High"])
        result.append({"name":["Fastest","Alternative"][i],"distance":km,"eta":eta,"risk":risk,
                        "geometry":rt['geometry'],
                        "traffic":"Heavy" if tf['is_peak'] and i==0 else "Light"})
    if result:
        result.sort(key=lambda x:["Low","Moderate","High","Severe"].index(x['risk']))
        result[0]['name']="Safest · "+result[0]['name']
    return jsonify({"routes":result,"destination":{"lat":dlat,"lng":dlng}})

@app.route('/roadtype')
def roadtype():
    lat=request.args.get('lat',type=float); lng=request.args.get('lng',type=float)
    if not lat or not lng: return jsonify({"road_type":"Local Road"})
    try:
        q=f"[out:json];way(around:60,{lat},{lng})[highway];out 1;"
        d=requests.get("https://overpass-api.de/api/interpreter",params={"data":q},timeout=5).json()
        hw=d.get("elements",[{}])[0].get("tags",{}).get("highway","")
        m={"motorway":"Highway","trunk":"Highway","primary":"City Road",
           "secondary":"City Road","tertiary":"City Road",
           "residential":"Residential","service":"Service Road","unclassified":"Local Road"}
        return jsonify({"road_type":m.get(hw,"Local Road")})
    except: return jsonify({"road_type":"Local Road"})

@app.route('/health')
def health():
    return jsonify({"status":"running","model_loaded":model is not None,
                    "features":feature_names,"time_now":get_time_features()})

if __name__=='__main__':
    print("\n🚗 SafeRoute AI Backend v3 — http://localhost:5000")
    app.run(port=5000,debug=True)