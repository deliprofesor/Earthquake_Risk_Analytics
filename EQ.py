"""
Tam Python analizi: Veri hazırlığı, zaman serisi, Gutenberg-Richter, magnitude-depth, fay segment analizi, moment->Mw hesaplama, çıktıların kaydı ve görseller.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

# ---------- Ayarlar ----------
INPUT_CSV = "marmara_faults_earthquakes_2000_2025.csv"
OUT_DIR = "analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams['figure.dpi'] = 120

# ---------- 0. Paket kontrol notu ----------
# Gerekirse: pip install pandas numpy matplotlib scikit-learn

# ---------- 1) Veri Hazırlığı ----------
df = pd.read_csv(INPUT_CSV)

print("Toplam kayıt sayısı:", len(df))

# Tarih dönüşümü
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
if df['Date'].isnull().any():
    print("UYARI: Date dönüşümünde NaT değerleri var. İlk 5:")
    print(df[df['Date'].isnull()].head())

# Ek sütunlar
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Mag_bin'] = pd.cut(df['Magnitude_ML'], bins=[-1,2,3,4,5,10], labels=['<2','2-3','3-4','4-5','5+'])

# Eksik özet
missing_summary = df.isnull().sum()
print("Eksik değerler (sütun bazlı):\n", missing_summary)

# ---------- 2) Zaman Serisi Analizi ----------
# Yıllık ve aylık özet
annual = df.groupby('Year', dropna=True).agg(
    Count=('Magnitude_ML','size'),
    Mean_Mag=('Magnitude_ML','mean')
).reset_index()

# Monthly: resample kullanabilmek için Date indeksli olmalı; bazı tarihlerde NaT olabilir -> dropna
monthly = df.dropna(subset=['Date']).set_index('Date').resample('M').agg(
    Count=('Magnitude_ML','size'),
    Mean_Mag=('Magnitude_ML','mean')
).reset_index()

# Grafik: yıllık count
try:
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(annual['Year'], annual['Count'], marker='o')
    ax.set_title('Annual earthquake counts (2000-2025)')
    ax.set_xlabel('Year'); ax.set_ylabel('Count'); ax.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "annual_counts.png")); plt.close(fig)
except Exception as e:
    print("annual_counts grafik oluşturulamadı:", e)

# Grafik: yıllık mean magnitude
try:
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(annual['Year'], annual['Mean_Mag'], marker='o')
    ax.set_title('Annual mean magnitude (2000-2025)')
    ax.set_xlabel('Year'); ax.set_ylabel('Mean Magnitude (ML)'); ax.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "annual_mean_magnitude.png")); plt.close(fig)
except Exception as e:
    print("annual_mean_magnitude grafik oluşturulamadı:", e)

# ---------- 3) Gutenberg-Richter (log10 N = a - b M) ----------
# Birikimli N(M>=m) hesapla
mags = np.sort(df['Magnitude_ML'].dropna().values)
if mags.size == 0:
    print("UYARI: Magnitude verisi yok, Gutenberg-Richter hesaplanamaz.")
    cum_df = pd.DataFrame(columns=['M','N'])
    a_val, b_val = np.nan, np.nan
else:
    m_min = np.floor(mags.min()*10)/10
    m_max = np.ceil(mags.max()*10)/10
    m_bins = np.arange(m_min, m_max+0.1, 0.1)

    cum_counts = []
    for m in m_bins:
        N = int((df['Magnitude_ML'] >= m).sum())
        cum_counts.append((m, N))
    cum_df = pd.DataFrame(cum_counts, columns=['M', 'N'])

    # Regresyon için mantıklı bir eşik seç: önce M>=2.0 ile dene; eğer yeterli değilse eşiği düşür
    def compute_gr(reg_threshold=2.0):
        reg_df = cum_df[(cum_df['M'] >= reg_threshold) & (cum_df['N'] > 0)].copy()
        # yeterli nokta var mı?
        if len(reg_df) >= 3:  # en az 3 nokta ile kabaca fit
            X = reg_df[['M']].values
            y = np.log10(reg_df['N'].values).reshape(-1,1)
            lr = LinearRegression().fit(X, y)
            a_val = float(lr.intercept_[0])
            b_val = float(-lr.coef_[0][0])
            return reg_df, a_val, b_val, lr
        else:
            return reg_df, np.nan, np.nan, None

    reg_df, a_val, b_val, lr_model = compute_gr(2.0)
    if reg_df.empty or lr_model is None:
        # eşiği düşürerek tekrar dene (ör. 1.5)
        reg_df, a_val, b_val, lr_model = compute_gr(1.5)
        if reg_df.empty or lr_model is None:
            # yine olmazsa daha düşük eşik
            reg_df, a_val, b_val, lr_model = compute_gr(1.0)

    if lr_model is None:
        print("UYARI: Gutenberg-Richter için yeterli veri bulunamadı (regresyon atlandı).")
    else:
        print(f"Gutenberg-Richter fit bulundu: a = {a_val:.4f}, b = {b_val:.4f}")

    # Grafik (varsa)
    if (reg_df is not None) and (not reg_df.empty) and (lr_model is not None):
        try:
            fig, ax = plt.subplots(figsize=(7,5))
            ax.scatter(reg_df['M'], np.log10(reg_df['N']), s=20, label='data')
            ax.plot(reg_df['M'], lr_model.predict(reg_df[['M']].values).ravel(),
                    color='red', linewidth=2,
                    label=f'fit: log10N={a_val:.2f} - {b_val:.2f}M')
            ax.set_xlabel('Magnitude M'); ax.set_ylabel('log10 N (M>=m)')
            ax.set_title('Gutenberg-Richter (log10 N vs M)')
            ax.legend(); ax.grid(True)
            plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "gutenberg_richter.png")); plt.close(fig)
        except Exception as e:
            print("gutenberg_richter grafik oluşturulamadı:", e)

# ---------- 4) Magnitude vs Depth (ilişki ve grafik) ----------
mag_depth_corr = None
if ('Magnitude_ML' in df.columns) and ('Depth_km' in df.columns):
    try:
        mag_depth_corr = df['Magnitude_ML'].corr(df['Depth_km'])
        print("Magnitude-Depth korelasyonu:", mag_depth_corr)
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(df['Depth_km'], df['Magnitude_ML'], s=8, alpha=0.5)
        ax.set_xlabel('Depth (km)'); ax.set_ylabel('Magnitude (ML)')
        ax.set_title(f'Magnitude vs Depth (corr={mag_depth_corr:.3f})')
        ax.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "magnitude_vs_depth.png")); plt.close(fig)
    except Exception as e:
        print("Magnitude-Depth analizi sırasında hata:", e)
else:
    print("UYARI: Magnitude_ML veya Depth_km sütunu eksik, magnitude-depth analizi yapılamadı.")

# ---------- 5) Fay Segment Analizi ----------
if 'Nearest_Fault' in df.columns:
    try:
        fault_summary = df.groupby('Nearest_Fault').agg(
            Count=('Magnitude_ML','size'),
            Mean_Mag=('Magnitude_ML','mean'),
            Mean_Slip_Rate_mm_per_yr=('Slip_Rate_mm_per_yr','mean'),
            Mean_Slip_Deficit_m=('Slip_Deficit_m','mean'),
            Mean_Elapsed_Time_yr=('Elapsed_Time_yr','mean'),
            Mean_Mw_Potential=('Mw_Potential','mean')
        ).reset_index().sort_values('Count', ascending=False)
        top_faults = fault_summary.head(10)
        print("Top 10 faults by event count:\n", top_faults[['Nearest_Fault','Count']])
    except Exception as e:
        print("Fay segment analizi sırasında hata:", e)
        fault_summary = pd.DataFrame()
else:
    print("UYARI: Nearest_Fault sütunu bulunamadı, fay analizi atlandı.")
    fault_summary = pd.DataFrame()

# ---------- 6) Potansiyel hesaplama: Moment -> Mw dönüşümü ----------
if 'Moment_Potential_Nm' in df.columns:
    df['Moment_Potential_Nm'] = pd.to_numeric(df['Moment_Potential_Nm'], errors='coerce')
    # Mw_from_M0: sadece pozitif M0 değerleri için hesapla
    with np.errstate(invalid='ignore'):
        df['Mw_from_M0'] = np.where(df['Moment_Potential_Nm'] > 0,
                                    (2.0/3.0) * (np.log10(df['Moment_Potential_Nm']) - 9.1),
                                    np.nan)
    mw_compare = df[['Mw_Potential','Mw_from_M0']].describe()
else:
    print("UYARI: Moment_Potential_Nm sütunu yok, Mw_from_M0 hesaplanamadı.")
    df['Mw_from_M0'] = np.nan
    mw_compare = pd.DataFrame()

# ---------- 7) Çıktıların kaydı ----------
try:
    annual.to_csv(os.path.join(OUT_DIR, "annual_summary.csv"), index=False)
    monthly.to_csv(os.path.join(OUT_DIR, "monthly_summary.csv"), index=False)
    cum_df.to_csv(os.path.join(OUT_DIR, "gutenberg_richter.csv"), index=False)
    fault_summary.to_csv(os.path.join(OUT_DIR, "fault_summary.csv"), index=False)
    df.to_csv(os.path.join(OUT_DIR, "dataset_with_mw_from_m0.csv"), index=False)
except Exception as e:
    print("CSV kaydetme sırasında hata:", e)

# summary.txt kaydı
try:
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(f"Run timestamp: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Gutenberg-Richter a: {a_val if 'a_val' in locals() else np.nan}, b: {b_val if 'b_val' in locals() else np.nan}\n")
        f.write(f"Magnitude-Depth corr: {mag_depth_corr}\n\n")
        f.write("Top 10 faults by count:\n")
        try:
            if not fault_summary.empty:
                f.write(fault_summary.head(10).to_string(index=False))
        except:
            f.write("Fay özeti bulunamadı.\n")
        f.write("\n\nMissing summary:\n")
        f.write(missing_summary.to_string())
    print("Analiz tamamlandı. Çıktılar klasörde:", OUT_DIR)
except Exception as e:
    print("summary.txt yazma sırasında hata:", e)

    
