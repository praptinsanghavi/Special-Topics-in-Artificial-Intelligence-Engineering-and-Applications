import streamlit as st
import numpy as np
import torch
import time
import pandas as pd
from src.simulation.kirana_env import KiranaClusterEnv
from src.agents.procurement_agent import ProcurementAgent

# Page Config
st.set_page_config(page_title="RetailUnion ERP", layout="wide")
st.title("RetailUnion | Smart Supply Chain ERP")

# Initialize Session State
if 'env' not in st.session_state:
    st.session_state.env = KiranaClusterEnv()
    st.session_state.obs, _ = st.session_state.env.reset()
    st.session_state.history_cash = []
    st.session_state.history_inv = []
    
    # Force Reset if old session data persists (Migration Fix)
    if isinstance(st.session_state.get('obs'), dict):
        st.cache_resource.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.env = KiranaClusterEnv()
        st.session_state.obs, _ = st.session_state.env.reset()
        st.session_state.history_cash = []
        st.session_state.history_inv = []
    
    st.session_state.ai_brain = ProcurementAgent(3, 3)
    try:
        st.session_state.ai_brain.policy_net.load_state_dict(torch.load("erp_brain.pth"))
        st.session_state.ai_brain.policy_net.eval()
        st.toast("AI Copilot Loaded", icon="🤖")
    except:
        st.warning("Training AI...")

# Sidebar: Controls
with st.sidebar:
    st.header("🛒 Operations Console")
    
    with st.expander("Help: How this works", expanded=True):
        st.write("""
        **The Goal**: Keep inventory between 10-50 units.
        
        **The Challenge**:
        *   **Manufacturer**: Cheap (₹70) but takes **3 Days** to arrive.
        *   **Distributor**: Fast (Instant) but Expensive (₹100).
        
        **Strategy**: Plan ahead! Order from Manufacturer *before* you run out.
        """)
    
    curr_obs = st.session_state.obs
    inv = int(curr_obs[0])
    cash = curr_obs[1]*1000
    pending = int(curr_obs[2])
    
    col1, col2 = st.columns(2)
    col1.metric("Stock On Hand", inv, help="Available for sale today")
    col2.metric("Cash Balance", f"₹{cash:,.0f}")
    
    st.metric("Pipeline Stock", pending, help="Ordered but not yet arrived")
    
    st.divider()
    
    # Recommendation Engine
    ai_action = st.session_state.ai_brain.select_action(curr_obs)
    rec_text = "HOLD"
    if ai_action == 1: rec_text = "ORDER MANUFACTURER (Plan Ahead)"
    elif ai_action == 2: rec_text = "ORDER DISTRIBUTOR (Panic!)"
    
    st.success(f"🤖 AI Recommendation: **{rec_text}**")
    
    st.subheader("Action")
    action_name = st.radio("Execute:", [
        "Hold (Do Nothing)",
        "Order Manufacturer (₹70, +3 Days)",
        "Order Distributor (₹100, Instant)"
    ])
    
    if "Hold" in action_name: action = 0
    elif "Manufacturer" in action_name: action = 1
    elif "Distributor" in action_name: action = 2
    
    step = st.button("End Day ▶️", type="primary")

if step:
    # Step Environment
    next_obs, reward, term, trunc, info = st.session_state.env.step(action)
    st.session_state.obs = next_obs
    
    st.session_state.history_cash.append(info['cash'])
    st.session_state.history_inv.append(st.session_state.obs[0])
    
    # Last Log
    st.session_state.last_info = info

# Main Area
col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("Financial & Stock Performance")
    
    df = pd.DataFrame({
        "Cash (₹)": st.session_state.history_cash,
        "Inventory (Units)": [x*100 for x in st.session_state.history_inv] # Scale for visibility
    })
    st.line_chart(df)
    st.caption("Note: Inventory scaled x100 for comparison.")

with col_b:
    st.subheader("Supply Chain Pipeline")
    
    # Visualize Pending Orders
    pending_orders = st.session_state.env.pending_orders
    # Create valid list even if empty
    # Format: (Days Left, Qty)
    
    if pending_orders:
        for days, qty in pending_orders:
            st.info(f"🚚 **{qty} Units** arriving in **{days} Days**")
    else:
        st.write("Is Pipeline Empty? Order from Manufacturer!")
        
    if 'last_info' in st.session_state:
        li = st.session_state.last_info
        st.divider()
        st.write("Yesterday's Ledger:")
        st.write(f"Sales: {li.get('sold',0)} | Revenue: ₹{li.get('revenue',0)}")
        if li.get('arrived', 0) > 0:
            st.success(f"🎉 Shipment Arrived: {li['arrived']} units!")
