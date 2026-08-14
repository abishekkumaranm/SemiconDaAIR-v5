# SemiconDaAIR Interactive Web Dashboard Application

**Target**: KLA / SEMICON India Hackathon — AI-Based Restoration of Degraded Images for Semiconductor Inspection  
**Location**: [`dashboard/`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/dashboard/)  

---

## 🌟 Overview & Key Features

This standalone interactive web application provides a live demonstration of `SemiconDaAIR-v2` metrology image restoration, real-time noise simulation, degradation router expert probability tracking, line-edge roughness (LER) sub-nanometer profile analysis, and competition benchmarks.

### Key Interactive Features:
1. **Before / After Drag-and-Drop Split Visualizer**: Compare degraded input ($128 \times 128$) vs restored output ($256 \times 256$) side-by-side.
2. **Real-Time Noise Simulator Sliders**: Dynamically adjust:
   - Additive Gaussian Noise ($\sigma$)
   - Multiplicative Speckle Noise ($\sigma_s$)
   - 2x Spatial Downsampling Ratio
3. **Degradation Router Expert Gauges**: Live probability visualization of $E_{\text{Gaussian}}$, $E_{\text{Speckle}}$, and $E_{\text{SR}}$ routing weights.
4. **Line-Edge Roughness (LER) Metrology Profile**: Real-time cross-section intensity line graph demonstrating edge preservation without ringing artifacts.
5. **Hackathon Q&A Accordion**: Built-in answers to the 24 evaluation questions for hackathon judges.

---

## 🚀 How to Launch the Web Application

### Method 1: Open Directly in Any Web Browser
Simply double-click [`dashboard/index.html`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/dashboard/index.html) or open it directly in Google Chrome, Microsoft Edge, or Firefox.

### Method 2: Launch via Local Python HTTP Server
```powershell
python -m http.server 8000 --directory "dashboard"
```
Then navigate to `http://localhost:8000` in your web browser!
