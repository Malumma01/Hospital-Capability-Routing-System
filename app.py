# app.py
import streamlit as st
import pandas as pd
import joblib
import random
import time
from math import radians, sin, cos, atan2, sqrt
import streamlit.components.v1 as components
from streamlit_geolocation import streamlit_geolocation

# Configure page
st.set_page_config(
    page_title="Malụmma - Hospital Capability Routing System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("Malụmma")
st.subheader("Hospital Capability Routing System:)

# Custom styling with colors

st.markdown("""
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }

    /* MAIN APP BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #0F766E 0%, #2563EB 100%);
        color: #111827;
    }

    /* MAIN PAGE HEADINGS ON BLUE BACKGROUND */
    h1 {
        color: #FFFFFF;
        text-align: center;
        font-size: 3em;
        font-weight: 800;
        margin: 20px 0;
        text-shadow: 0 2px 6px rgba(0,0,0,0.35);
    }

    h2 {
        color: #FFFFFF;
        font-size: 1.9em;
        font-weight: 700;
        margin: 14px 0;
    }

    h3 {
        color: #FFFFFF;
        font-size: 1.45em;
        font-weight: 700;
        margin: 12px 0;
    }

    h4 {
        color: #ffffff;
        font-size: 1.15em;
        font-weight: 700;
    }

    p, li, label, span, div {
        font-size: 1rem;
        line-height: 1.6;
    }

    /* WHITE CONTENT CARDS */
    .stContainer {
        background: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.18);
        color: #111827;
    }

    .stForm {
        background: rgba(255,255,255,0.97);
        border-radius: 15px;
        padding: 20px;
        color: #111827;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #0F766E 0%, #2563EB 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 1.08em;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(0,0,0,0.20);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }

    /* INPUTS */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        border: 2px solid #0F766E !important;
        border-radius: 8px !important;
        padding: 10px !important;
        background: #FFFFFF !important;
        color: #111827 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }

    /* SUCCESS BOXES (FIRST AID) */
    .stSuccess {
        background: #ECFDF5 !important;
        color: #065F46 !important;
        border: 2px solid #10B981 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 12px !important;
    }

    /* ERROR BOXES (DO NOT) */
    .stError {
        background: #dc2626 !important;
        color: #991B1B !important;
        border: 2px solid #EF4444 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 12px !important;
    }

    /* WARNING BOXES */
    .stWarning {
        background: #FFFBEB !important;
        color: #92400E !important;
        border: 2px solid #F59E0B !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 12px !important;
    }

    /* INFO BOXES */
    .stInfo {
        background: #EFF6FF !important;
        color: #000000 !important;
        border: 2px solid #60A5FA !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 12px !important;
    }

    /* TOP FEATURE / METRIC CARDS */
    .metric-card {
        background: linear-gradient(135deg, #0F766E 0%, #2563EB 100%);
        color: #FFFFFF;
        padding: 22px;
        border-radius: 12px;
        margin: 10px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.20);
    }

    .metric-card h3,
    .metric-card p {
        color: #FFFFFF !important;
    }

    /* STREAMLIT METRICS */
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.12);
    }

    /* LINKS */
    a {
        color: #FFFFFF !important;
        font-weight: 700;
        text-decoration: underline;
    }

    /* MOBILE */
    @media (max-width: 768px) {
        h1 { font-size: 2.2em; }
        h2 { font-size: 1.5em; }
        h3 { font-size: 1.2em; }
        .stButton > button {
            width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = ""
if 'city' not in st.session_state:
    st.session_state.city = "Abuja"
if 'lat' not in st.session_state:
    st.session_state.lat = 9.0579
if 'lon' not in st.session_state:
    st.session_state.lon = 7.4891
if 'triage_result' not in st.session_state:
    st.session_state.triage_result = None
if 'best_hospitals' not in st.session_state:
    st.session_state.best_hospitals = None

# Load models
@st.cache_resource
def load_triage_model():
    vectorizer = joblib.load("triage_vectorizer.joblib")
    model = joblib.load("triage_model.joblib")
    return vectorizer, model

vectorizer, triage_model = load_triage_model()

# Load hospitals
@st.cache_data
def load_hospitals():
    return pd.read_csv("hospitals.csv")

hospitals_df = load_hospitals()

# Helper functions
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def simulate_status(row):
    random.seed(int(time.time() // 60) + int(row["lat"] * 1000))
    return random.choices(["Available", "Busy", "Offline"], weights=[0.6, 0.3, 0.1], k=1)[0]

def estimate_ambulance_eta(distance_km):
    if distance_km <= 0.5:
        return 5
    return int((distance_km / 30) * 60) + 5

def google_maps_embed_url(lat, lon, zoom=15):
    return f"https://www.google.com/maps?q={lat},{lon}&z={zoom}&output=embed"

def google_maps_route_url(origin_lat, origin_lon, dest_lat, dest_lon):
    return f"https://www.google.com/maps/dir/{origin_lat},{origin_lon}/{dest_lat},{dest_lon}"

def triage(symptom_text):
    X = vectorizer.transform([symptom_text])
    label = triage_model.predict(X)[0]
    result = {
        "severity": "Moderate",
        "urgency": "Within 2 hours",
        "first_aid": [],
        "dont": [],
        "required_capabilities": {}
    }
    if label == "snakebite":
        result.update({
            "severity": "Critical",
            "urgency": "Immediate (within 30 minutes)",
            "first_aid": ["Keep the patient calm and still.", "Immobilize the bitten limb at heart level.", "Remove tight clothing, rings, or bracelets."],
            "dont": ["Do NOT cut the wound.", "Do NOT suck the venom.", "Do NOT apply a tourniquet.", "Do NOT give alcohol or stimulants."],
            "priority": 1,
            "required_capabilities": {"has_antivenom": 1}
        })
    elif label == "child_not_breathing_well":
        result.update({
            "severity": "Critical",
            "urgency": "Immediate (call ambulance now)",
            "first_aid": ["Check responsiveness and breathing.", "If not breathing, start CPR if trained.", "Place child on a firm surface."],
            "dont": ["Do NOT leave the child alone.", "Do NOT give food or drink."],
            "priority": 2,
            "required_capabilities": {"has_pediatric_icu": 1}
        })
    elif label == "seizure":
        result.update({
            "severity": "High",
            "urgency": "Within 1 hour",
            "first_aid": ["Lay the person on their side.", "Clear the area around them.", "Loosen tight clothing around the neck."],
            "dont": ["Do NOT put anything in their mouth.", "Do NOT try to restrain their movements."],
            "priority": 3,
            "required_capabilities": {"has_oxygen": 1}
        })
    else:
        result.update({
            "first_aid": ["Monitor symptoms.", "If symptoms worsen, seek medical care immediately."],
            "dont": ["Do NOT self-medicate heavily without medical advice."]
        })
    return result

def find_best_hospitals(user_lat, user_lon, triage_result, hospitals_df, top_n=3):
    df = hospitals_df.copy()
    for cap, needed in triage_result["required_capabilities"].items():
        df = df[df[cap] == needed]
    if df.empty:
        df = hospitals_df.copy()
    df["distance_km"] = df.apply(lambda row: haversine(user_lat, user_lon, row["lat"], row["lon"]), axis=1)
    df["status"] = df.apply(simulate_status, axis=1)
    status_order = {"Available": 0, "Busy": 1, "Offline": 2}
    df["status_rank"] = df["status"].map(status_order)
    priority = triage_result.get("priority", 3)
    if priority == 1:
        df = df.sort_values(["status_rank", "distance_km"])
    elif priority == 2:
        df = df.sort_values(["status_rank", "distance_km"])
    else:
        df = df.sort_values(["distance_km", "status_rank"])
    return df.head(top_n)

# PAGE 1: HOME
def page_home():
    st.markdown("<h1>🏥 HCRS</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:white;'>Hospital Capability Routing System</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>⚡ Fast</h3>
            <p>Real-time routing to the right facility</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>📍 Live GPS</h3>
            <p>Auto-detect your location</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>🎯 Smart</h3>
            <p>AI-powered hospital matching</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h3>About HCRS</h3>", unsafe_allow_html=True)
    st.info("""
    HCRS is an emergency response system that helps you find the right hospital based on your symptoms and location.
    - Describe your emergency symptoms
    - Allow GPS or enter location manually
    - Get directed to the nearest facility with the required care level
    """)
    
    st.markdown("<h3>Contact & Support</h3>", unsafe_allow_html=True)
    st.write("📞 Emergency: available on request")
    st.write("📧 Email: jideunochioma@gmail.com")
    st.write("🌐 Website: www.hcrs.com")
    
    st.markdown("---")
    st.markdown("<h3>Let's Get Started</h3>", unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Select Your City")
            st.session_state.city = st.selectbox("City", ["Abuja", "Lagos"], key="city_select")
        
        with col2:
            st.subheader("📍 Your Location")
            location = streamlit_geolocation()
            if isinstance(location, dict):
                lat = location.get('latitude')
                lon = location.get('longitude')
                if lat is not None and lon is not None:
                    st.session_state.lat = lat
                    st.session_state.lon = lon
                    st.success(f"GPS detected: {lat:.4f}, {lon:.4f}")
                else:
                    st.session_state.lat = st.number_input("Latitude", value=st.session_state.lat, key="lat_input")
                    st.session_state.lon = st.number_input("Longitude", value=st.session_state.lon, key="lon_input")
            else:
                st.session_state.lat = st.number_input("Latitude", value=st.session_state.lat, key="lat_input2")
                st.session_state.lon = st.number_input("Longitude", value=st.session_state.lon, key="lon_input2")
        
        st.subheader("📝 Describe Your Emergency")
        st.session_state.symptoms = st.text_area("What is happening?", value=st.session_state.symptoms, height=100, key="symptoms_input")
        
        if st.button("➜ Continue to Results", use_container_width=True):
            if not st.session_state.symptoms.strip():
                st.error("Please describe your emergency before continuing.")
            else:
                st.session_state.page = 'results'
                st.rerun()

# PAGE 2: RESULTS
def page_results():
    st.markdown("<h1>🏥 HCRS - Results</h1>", unsafe_allow_html=True)
    
    # Perform triage and find hospitals
    triage_result = triage(st.session_state.symptoms)
    st.session_state.triage_result = triage_result
    
    filtered = hospitals_df[hospitals_df["city"].str.strip().str.lower() == st.session_state.city.lower()]
    best = find_best_hospitals(user_lat=st.session_state.lat, user_lon=st.session_state.lon, triage_result=triage_result, hospitals_df=filtered)
    st.session_state.best_hospitals = best
    
    # Triage section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Severity</h3>
            <p style='font-size: 1.5em; font-weight: bold;'>{triage_result['severity']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Urgency</h3>
            <p style='font-size: 1.2em;'>{triage_result['urgency']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        if st.session_state.lat and st.session_state.lon:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Location</h3>
                <p>{st.session_state.lat:.4f}, {st.session_state.lon:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # First aid
    st.markdown("<h3>🚑 Immediate First Aid</h3>", unsafe_allow_html=True)
    for step in triage_result["first_aid"]:
        st.success(f"✓ {step}")
    
    st.markdown("<h3>⚠️ Do NOT</h3>", unsafe_allow_html=True)
    for d in triage_result["dont"]:
        st.error(f"✗ {d}")
    
    st.markdown("---")
    
    # Hospital recommendations
    st.markdown(f"<h3>🏨 Recommended Hospitals in {st.session_state.city}</h3>", unsafe_allow_html=True)
    
    if best is not None and isinstance(best, pd.DataFrame) and not best.empty:
        best_hospital = best.iloc[0]
        eta = estimate_ambulance_eta(best_hospital['distance_km'])
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #F8FAFC 0%, #2563EB 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;'>
            <h3 style='color: white;'>🎯 PRIMARY RECOMMENDATION</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<h4>{best_hospital['name']}</h4>", unsafe_allow_html=True)
        with col2:
            st.metric("Distance", f"{best_hospital['distance_km']:.1f} km")
        with col3:
            st.metric("ETA", f"{eta} min")
        with col4:
            st.metric("Status", best_hospital['status'])
        
        maps_link = google_maps_route_url(st.session_state.lat, st.session_state.lon, best_hospital['lat'], best_hospital['lon'])
        st.markdown(f"[🗺️ Open Route in Google Maps]({maps_link})")
        st.markdown("#### Your Route on Google Maps")
        components.iframe(google_maps_embed_url(best_hospital['lat'], best_hospital['lon']), height=300)
        
        if len(best) > 1:
            st.markdown("<h3>🏨 Other Nearby Options</h3>", unsafe_allow_html=True)
            for _, row in best.iloc[1:].iterrows():
                st.markdown(f"""
                <div style='background: #f0f0f0; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <h4>{row['name']}</h4>
                    <p>Distance: <b>{row['distance_km']:.1f} km</b> | Status: <b>{row['status']}</b> | 24/7: {'Yes' if row.get('emergency_247', False) else 'No'}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No facility with the required capability found. Showing nearest hospitals as fallback.")
        fallback = filtered.copy()
        fallback['distance_km'] = fallback.apply(lambda row: haversine(st.session_state.lat, st.session_state.lon, row['lat'], row['lon']), axis=1)
        fallback = fallback.sort_values('distance_km').head(3)
        
        for _, row in fallback.iterrows():
            st.markdown(f"""
            <div style='background: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                <h4>{row['name']}</h4>
                <p>Distance: <b>{row['distance_km']:.1f} km</b> | 24/7: {'Yes' if row.get('emergency_247', False) else 'No'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Start", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
    
    with col2:
        if st.button("✓ Confirm & Thank You", use_container_width=True):
            st.session_state.page = 'thankyou'
            st.rerun()

# PAGE 3: THANK YOU
def page_thankyou():
    st.markdown("<h1 style='text-align: center;'>✓ Thank You</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #F8FAFC 0%, #2563EB 100%); color: white; padding: 40px; border-radius: 15px; text-align: center; margin: 20px 0;'>
        <h2 style='color: white;'>Your Emergency Request is Confirmed</h2>
        <p style='font-size: 1.2em; margin: 20px 0;'>Help is on the way to you!</p>
    </div>
    """, unsafe_allow_html=True)
    
    best_hospital = st.session_state.best_hospitals.iloc[0] if st.session_state.best_hospitals is not None and not st.session_state.best_hospitals.empty else None
    
    if best_hospital is not None:
        try:
            facility_name = str(best_hospital['name'])
            distance = float(best_hospital['distance_km'])
            eta = estimate_ambulance_eta(distance)
            severity = str(st.session_state.triage_result['severity'])
            
            st.markdown(f"""
            <h3>Summary of Your Request</h3>
            <ul>
                <li><b>Facility:</b> {facility_name}</li>
                <li><b>City:</b> {st.session_state.city}</li>
                <li><b>Condition Severity:</b> {severity}</li>
                <li><b>Estimated Ambulance Time:</b> {eta} minutes</li>
                <li><b>Your Location:</b> {st.session_state.lat:.4f}, {st.session_state.lon:.4f}</li>
            </ul>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not display request summary. Error: {str(e)}")
    else:
        st.warning("No hospital request information found. Please start a new request.")
    
    st.info("📞 Keep your phone accessible. The hospital will contact you shortly.")
    st.success("✓ Share your live location with emergency contacts if needed.")
    st.warning("⚠️ If your condition worsens, call emergency services immediately.")
    
    st.markdown("---")
    
    if st.button("🔄 Start New Emergency Request", use_container_width=True):
        st.session_state.page = 'home'
        st.session_state.symptoms = ""
        st.rerun()

# Main app logic
if st.session_state.page == 'home':
    page_home()
elif st.session_state.page == 'results':
    page_results()
elif st.session_state.page == 'thankyou':
    page_thankyou()
