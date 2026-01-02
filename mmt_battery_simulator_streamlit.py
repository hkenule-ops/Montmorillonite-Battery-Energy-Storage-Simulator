import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import streamlit as st
import pandas as pd
from io import BytesIO

# For PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# Constants
F = 96485  # Faraday's constant (C/mol)
R = 8.314  # Gas constant (J/mol·K)

def main():
    st.set_page_config(page_title="Montmorillonite Battery Energy Storage Simulator", layout="wide")
    st.title("Montmorillonite Battery Energy Storage Simulator")

    # Default parameters
    default_params = {
        'diffusionCoeff': 1e-14,
        'exchangeCurrentDensity': 0.01,
        'temperature': 298,
        'activeMass': 0.001,
        'dischargeCurrent': 0.1,
        'dischargeTime': 3600,
        'avgVoltage': 2.5,
        'alphaA': 0.5,
        'alphaC': 0.5,
        'cycles': 50,
        'capacityFade': 0.002
    }

    # Sidebar for parameters
    st.sidebar.header("Parameters")

    st.sidebar.subheader("Material Properties")
    diffusion_coeff = st.sidebar.number_input("Diffusion Coefficient (m²/s)", value=default_params['diffusionCoeff'], format="%.2e")
    exchange_current_density = st.sidebar.number_input("Exchange Current Density (A/m²)", value=default_params['exchangeCurrentDensity'], format="%.2e")
    temperature = st.sidebar.number_input("Temperature (K)", value=default_params['temperature'])
    active_mass = st.sidebar.number_input("Active Mass (kg)", value=default_params['activeMass'])

    st.sidebar.subheader("Electrochemical Parameters")
    discharge_current = st.sidebar.number_input("Discharge Current (A)", value=default_params['dischargeCurrent'])
    discharge_time = st.sidebar.number_input("Discharge Time (s)", value=default_params['dischargeTime'])
    avg_voltage = st.sidebar.number_input("Average Voltage (V)", value=default_params['avgVoltage'])
    cycles = st.sidebar.number_input("Number of Cycles", value=default_params['cycles'], step=1)
    capacity_fade = st.sidebar.number_input("Capacity Fade Rate", value=default_params['capacityFade'])

    params = {
        'diffusionCoeff': diffusion_coeff,
        'exchangeCurrentDensity': exchange_current_density,
        'temperature': temperature,
        'activeMass': active_mass,
        'dischargeCurrent': discharge_current,
        'dischargeTime': discharge_time,
        'avgVoltage': avg_voltage,
        'alphaA': default_params['alphaA'],  # Fixed in original, so keeping default
        'alphaC': default_params['alphaC'],  # Fixed in original
        'cycles': cycles,
        'capacityFade': capacity_fade
    }

    if st.sidebar.button("Run Simulation"):
        results = run_simulation(params)
        display_results(results)

def run_simulation(p):
    specific_capacity = (p['dischargeCurrent'] * p['dischargeTime']) / p['activeMass']
    energy_density = (specific_capacity * p['avgVoltage']) / 1000
    power_density = energy_density / (p['dischargeTime'] / 3600)

    time_steps = 100
    charge_discharge = []
    for i in range(time_steps + 1):
        t_norm = i / time_steps
        voltage = p['avgVoltage'] * (1 - 0.3 * t_norm)
        capacity = specific_capacity * t_norm
        charge_discharge.append({
            'time': (p['dischargeTime'] * t_norm) / 3600,
            'voltage': voltage,
            'capacity': capacity
        })

    cycle_life = []
    for cycle in range(p['cycles'] + 1):
        retention = 100 * np.exp(-p['capacityFade'] * cycle)
        remaining_capacity = specific_capacity * (retention / 100)
        cycle_life.append({'cycle': cycle, 'retention': retention, 'capacity': remaining_capacity})

    kinetics = []
    for eta in np.arange(-0.3, 0.31, 0.01):
        anodic = np.exp((p['alphaA'] * F * eta) / (R * p['temperature']))
        cathodic = np.exp((-p['alphaC'] * F * eta) / (R * p['temperature']))
        current_density = p['exchangeCurrentDensity'] * (anodic - cathodic)
        kinetics.append({'overpotential': eta, 'currentDensity': current_density})

    diffusion = []  # Computed but not used in plots in original
    for x in range(51):
        pos = x / 50
        conc = 1 - np.exp(-p['diffusionCoeff'] * 1e12 * pos)
        diffusion.append({'position': pos, 'concentration': conc})

    comparison = [
        {"metric": "Specific Capacity", "MMT": (specific_capacity / 372) * 100, "LiIon": 100},
        {"metric": "Energy Density", "MMT": (energy_density / 250) * 100, "LiIon": 100},
        {"metric": "Cycle Life", "MMT": 75, "LiIon": 100},
        {"metric": "Sustainability", "MMT": 95, "LiIon": 40},
        {"metric": "Cost Effectiveness", "MMT": 90, "LiIon": 60},
        {"metric": "Power Density", "MMT": (power_density / 500) * 100, "LiIon": 100},
    ]

    return {
        'specificCapacity': round(specific_capacity, 2),
        'energyDensity': round(energy_density, 2),
        'powerDensity': round(power_density, 2),
        'chargeDischargeData': charge_discharge,
        'cycleLifeData': cycle_life,
        'kineticsData': kinetics,
        'diffusionData': diffusion,
        'comparisonData': comparison
    }

def display_results(r):
    st.header("Simulation Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Specific Capacity", f"{r['specificCapacity']} mAh/g")
    with col2:
        st.metric("Energy Density", f"{r['energyDensity']} Wh/kg")
    with col3:
        st.metric("Power Density", f"{r['powerDensity']} W/kg")

    # Export buttons
    pdf_data = export_pdf(r)
    st.download_button("Export as PDF Report", data=pdf_data, file_name=f"MMT_Battery_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")

    excel_data = export_excel(r)
    st.download_button("Export to Excel", data=excel_data, file_name=f"MMT_Battery_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Plots
    st.subheader("Charge/Discharge Profile")
    fig, ax = plt.subplots(figsize=(10, 6))
    times = [d['time'] for d in r['chargeDischargeData']]
    ax.plot(times, [d['voltage'] for d in r['chargeDischargeData']], label='Voltage (V)', color='#8b5cf6', linewidth=3)
    ax2 = ax.twinx()
    ax2.plot(times, [d['capacity'] for d in r['chargeDischargeData']], label='Capacity (mAh/g)', color='#10b981', linewidth=3)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Voltage (V)')
    ax2.set_ylabel('Capacity (mAh/g)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    st.pyplot(fig)

    st.subheader("Cycle Life Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([d['cycle'] for d in r['cycleLifeData']], [d['retention'] for d in r['cycleLifeData']], color='#3b82f6', linewidth=3, marker='o')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Capacity Retention (%)')
    st.pyplot(fig)

    st.subheader("Butler-Volmer Kinetics")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([d['overpotential'] for d in r['kineticsData']], [d['currentDensity'] for d in r['kineticsData']], color='#ef4444', linewidth=3)
    ax.set_xlabel('Overpotential (V)')
    ax.set_ylabel('Current Density (A/m²)')
    st.pyplot(fig)

    st.subheader("Performance Comparison: MMT vs Li-ion")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    metrics = [d['metric'] for d in r['comparisonData']]
    values_mmt = [d['MMT'] for d in r['comparisonData']]
    values_li = [d['LiIon'] for d in r['comparisonData']]
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    values_mmt += values_mmt[:1]
    values_li += values_li[:1]
    angles += angles[:1]
    ax.plot(angles, values_mmt, 'o-', linewidth=3, label='MMT Clay', color='#10b981')
    ax.fill(angles, values_mmt, alpha=0.3, color='#10b981')
    ax.plot(angles, values_li, 'o-', linewidth=3, label='Li-ion', color='#3b82f6')
    ax.fill(angles, values_li, alpha=0.3, color='#3b82f6')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    st.pyplot(fig)

def export_pdf(r):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Montmorillonite Clay Battery Simulation Report", styles['Title']))
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))

    # Metrics Table
    data = [["Metric", "Value", "Unit"],
            ["Specific Capacity", f"{r['specificCapacity']}", "mAh/g"],
            ["Energy Density", f"{r['energyDensity']}", "Wh/kg"],
            ["Power Density", f"{r['powerDensity']}", "W/kg"]]
    table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
    table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),
                               ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                               ('ALIGN',(0,0),(-1,-1),'CENTER'),
                               ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                               ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                               ('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(table)
    story.append(Spacer(1, 0.5*inch))

    # Save plots in memory
    plot_buffers = save_plots_for_pdf(r)

    for title, buf in plot_buffers.items():
        story.append(Paragraph(title, styles['Heading2']))
        img = Image(buf, width=6.5*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 0.4*inch))

    # Comparison Table
    story.append(Paragraph("Performance Comparison with Li-ion", styles['Heading2']))
    comp_data = [["Metric", "MMT (%)", "Li-ion (%)"]]
    for item in r['comparisonData']:
        comp_data.append([item['metric'], f"{item['MMT']:.1f}", "100"])
    comp_table = Table(comp_data)
    comp_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightblue),
                                    ('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(comp_table)

    doc.build(story)
    buffer.seek(0)
    return buffer

def save_plots_for_pdf(r):
    buffers = {}

    # Charge/Discharge
    fig, ax1 = plt.subplots(figsize=(10, 6))
    times = [d['time'] for d in r['chargeDischargeData']]
    ax1.plot(times, [d['voltage'] for d in r['chargeDischargeData']], label='Voltage', color='#8b5cf6', linewidth=3)
    ax2 = ax1.twinx()
    ax2.plot(times, [d['capacity'] for d in r['chargeDischargeData']], label='Capacity', color='#10b981', linewidth=3)
    ax1.set_title('Charge/Discharge Profile', fontsize=14, fontweight='bold')
    buf1 = BytesIO()
    fig.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf1.seek(0)
    buffers["Charge/Discharge Profile"] = buf1

    # Cycle Life
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([d['cycle'] for d in r['cycleLifeData']], [d['retention'] for d in r['cycleLifeData']], color='#3b82f6', linewidth=3)
    ax.set_title('Cycle Life Analysis', fontsize=14, fontweight='bold')
    buf2 = BytesIO()
    fig.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf2.seek(0)
    buffers["Cycle Life"] = buf2

    # Butler-Volmer
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([d['overpotential'] for d in r['kineticsData']], [d['currentDensity'] for d in r['kineticsData']], color='#ef4444', linewidth=3)
    ax.set_title('Butler-Volmer Kinetics', fontsize=14, fontweight='bold')
    buf3 = BytesIO()
    fig.savefig(buf3, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf3.seek(0)
    buffers["Butler-Volmer Kinetics"] = buf3

    # Radar
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    metrics = [d['metric'] for d in r['comparisonData']]
    values_mmt = [d['MMT'] for d in r['comparisonData']] + [d['MMT'] for d in r['comparisonData']][:1]
    values_li = [100] * len(metrics) + [100]
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist() + [0]
    ax.plot(angles, values_mmt, 'o-', linewidth=3, label='MMT', color='#10b981')
    ax.fill(angles, values_mmt, alpha=0.3, color='#10b981')
    ax.plot(angles, values_li, 'o-', linewidth=3, label='Li-ion', color='#3b82f6')
    ax.fill(angles, values_li, alpha=0.3, color='#3b82f6')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    buf4 = BytesIO()
    fig.savefig(buf4, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf4.seek(0)
    buffers["Performance Comparison"] = buf4

    return buffers

def export_excel(r):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        pd.DataFrame([{
            'Parameter': 'Specific Capacity (mAh/g)', 'Value': r['specificCapacity']
        }, {
            'Parameter': 'Energy Density (Wh/kg)', 'Value': r['energyDensity']
        }, {
            'Parameter': 'Power Density (W/kg)', 'Value': r['powerDensity']
        }]).to_excel(writer, sheet_name='Summary', index=False)

        pd.DataFrame(r['chargeDischargeData']).to_excel(writer, sheet_name='Discharge Profile', index=False)
        pd.DataFrame(r['cycleLifeData']).to_excel(writer, sheet_name='Cycle Life', index=False)
        pd.DataFrame(r['comparisonData']).to_excel(writer, sheet_name='Comparison', index=False)

    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    main()