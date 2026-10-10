from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Analisis Attrition Karyawan",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "employee_data_cleaned.csv"
SATISFACTION_LABELS = {
    1: "Rendah",
    2: "Sedang-rendah",
    3: "Sedang-tinggi",
    4: "Tinggi",
}


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["Attrition"] = pd.to_numeric(data["Attrition"], errors="coerce")
    data = data.dropna(subset=["Attrition"]).copy()
    data["Attrition"] = data["Attrition"].astype(int)

    for column in [
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "WorkLifeBalance",
        "JobInvolvement",
        "RelationshipSatisfaction",
    ]:
        data[f"{column}Label"] = data[column].map(SATISFACTION_LABELS)

    for column, label in [
        ("MonthlyIncome", "Pendapatan bulanan"),
        ("TotalWorkingYears", "Total pengalaman kerja"),
        ("YearsAtCompany", "Masa kerja di perusahaan"),
        ("Age", "Kelompok usia"),
    ]:
        data[f"{column}Group"] = pd.qcut(
            data[column], q=4, duplicates="drop"
        ).astype(str)

    return data


try:
    employee_data = load_data()
except (FileNotFoundError, pd.errors.ParserError, KeyError) as error:
    st.error(f"Data tidak dapat dibaca: {error}")
    st.stop()

if employee_data.empty:
    st.error("Tidak ada data dengan label Attrition yang dapat dianalisis.")
    st.stop()

st.title("Analisis Attrition Karyawan")
st.caption(
    "Peta pola attrition untuk membantu HR menentukan hal yang perlu diselidiki. "
    "Temuan ini bersifat deskriptif, bukan bukti sebab-akibat."
)

total_employees = len(employee_data)
attrition_count = int(employee_data["Attrition"].sum())
attrition_rate = employee_data["Attrition"].mean()
retained_count = total_employees - attrition_count

metric_columns = st.columns(4)
metric_columns[0].metric("Karyawan berlabel", f"{total_employees:,}")
metric_columns[1].metric("Attrition", f"{attrition_count:,}")
metric_columns[2].metric("Tingkat attrition", f"{attrition_rate:.1%}")
metric_columns[3].metric("Tidak attrition", f"{retained_count:,}")

st.divider()

feature_options = {
    "Overtime": "OverTime",
    "Perjalanan dinas": "BusinessTravel",
    "Departemen": "Department",
    "Jabatan": "JobRole",
    "Kepuasan kerja": "JobSatisfactionLabel",
    "Kepuasan lingkungan kerja": "EnvironmentSatisfactionLabel",
    "Keseimbangan kehidupan-kerja": "WorkLifeBalanceLabel",
    "Keterlibatan kerja": "JobInvolvementLabel",
    "Kepuasan hubungan kerja": "RelationshipSatisfactionLabel",
    "Status pernikahan": "MaritalStatus",
    "Opsi saham": "StockOptionLevel",
    "Pendapatan bulanan": "MonthlyIncomeGroup",
    "Total pengalaman kerja": "TotalWorkingYearsGroup",
    "Masa kerja di perusahaan": "YearsAtCompanyGroup",
    "Kelompok usia": "AgeGroup",
}

left, right = st.columns([1, 2])
with left:
    minimum_group_size = st.slider(
        "Minimum karyawan per kelompok",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
        help="Kelompok yang lebih kecil disembunyikan agar perbandingan tidak terlalu mudah menyesatkan.",
    )
    selected_label = st.selectbox("Bandingkan faktor", list(feature_options))

with right:
    st.subheader("Tingkat attrition menurut faktor")
    selected_column = feature_options[selected_label]
    grouped = (
        employee_data.groupby(selected_column, dropna=False)["Attrition"]
        .agg(karyawan="size", attrition="sum", tingkat="mean")
        .reset_index()
        .rename(columns={selected_column: "Kelompok"})
    )
    grouped = grouped[grouped["karyawan"] >= minimum_group_size].copy()

    if grouped.empty:
        st.info("Tidak ada kelompok yang memenuhi minimum karyawan tersebut.")
    else:
        grouped["Kelompok"] = grouped["Kelompok"].fillna("Tidak diketahui").astype(str)
        grouped = grouped.sort_values("tingkat", ascending=True)
        chart = px.bar(
            grouped,
            x="tingkat",
            y="Kelompok",
            orientation="h",
            text=grouped["tingkat"].map(lambda value: f"{value:.1%}"),
            hover_data={
                "karyawan": True,
                "attrition": True,
                "tingkat": ":.1%",
            },
            labels={
                "tingkat": "Tingkat attrition",
                "Kelompok": "Kelompok",
                "karyawan": "Jumlah karyawan",
                "attrition": "Jumlah attrition",
            },
            color="tingkat",
            color_continuous_scale="Tealrose",
        )
        chart.update_layout(
            xaxis_tickformat=".0%",
            xaxis_range=[0, min(1, max(grouped["tingkat"].max() * 1.2, 0.1))],
            coloraxis_showscale=False,
            margin=dict(l=10, r=15, t=10, b=10),
            height=max(300, 48 * len(grouped)),
        )
        chart.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(chart, width="stretch")

        display_table = grouped.copy()
        display_table["Tingkat attrition"] = display_table["tingkat"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(
            display_table[["Kelompok", "karyawan", "attrition", "Tingkat attrition"]]
            .rename(
                columns={
                    "karyawan": "Karyawan",
                    "attrition": "Kasus attrition",
                }
            ),
            hide_index=True,
            width="stretch",
        )

st.divider()
st.subheader("Pola yang perlu ditindaklanjuti")

factor_summaries = []
for feature_label, column in feature_options.items():
    summary = (
        employee_data.groupby(column, dropna=False)["Attrition"]
        .agg(karyawan="size", tingkat="mean")
        .reset_index()
    )
    summary = summary[summary["karyawan"] >= minimum_group_size]
    if len(summary) < 2:
        continue
    highest = summary.loc[summary["tingkat"].idxmax()]
    lowest = summary.loc[summary["tingkat"].idxmin()]
    factor_summaries.append(
        {
            "Faktor": feature_label,
            "Kelompok dengan tingkat tertinggi": str(highest[column]),
            "Tingkat tertinggi": highest["tingkat"],
            "Kelompok dengan tingkat terendah": str(lowest[column]),
            "Tingkat terendah": lowest["tingkat"],
            "Selisih": highest["tingkat"] - lowest["tingkat"],
            "Jumlah kelompok": len(summary),
        }
    )

if factor_summaries:
    factor_table = pd.DataFrame(factor_summaries).sort_values(
        "Selisih", ascending=False
    )
    top_factors = factor_table.head(3)
    for _, factor in top_factors.iterrows():
        st.markdown(
            f"- **{factor['Faktor']}**: kelompok `{factor['Kelompok dengan tingkat tertinggi']}` "
            f"memiliki tingkat attrition {factor['Tingkat tertinggi']:.1%}, sedangkan "
            f"`{factor['Kelompok dengan tingkat terendah']}` {factor['Tingkat terendah']:.1%} "
            f"(selisih {factor['Selisih']:.1%})."
        )
    st.caption(
        "Urutan didasarkan pada selisih tingkat tertinggi dan terendah, bukan uji kausal. "
        "Kelompok kecil di bawah batas minimum tidak disertakan."
    )
else:
    st.info("Belum ada faktor dengan setidaknya dua kelompok yang cukup besar.")

with st.expander("Cara membaca dan batasan analisis"):
    st.markdown(
        """
        - Tingkat attrition adalah jumlah karyawan dengan `Attrition = 1` dibagi jumlah karyawan di kelompok tersebut.
        - Kelompok pendapatan, usia, dan masa kerja dibentuk menjadi kuartil dari data yang tersedia.
        - Hubungan yang terlihat tidak membuktikan penyebab. Faktor lain, kondisi organisasi, dan bias pada data dapat memengaruhi hasil.
        - Gunakan temuan untuk menentukan pertanyaan lanjutan, lalu validasi lewat exit interview, survei, dan diskusi dengan karyawan.
        - Jangan gunakan skor atau pola ini sebagai dasar tunggal keputusan terhadap individu.
        """
    )

with st.expander("Ringkasan data agregat"):
    st.write(
        f"Dataset berisi **{total_employees:,}** karyawan dengan label attrition "
        f"yang tersedia. **{attrition_count:,}** memiliki label attrition dan "
        f"**{retained_count:,}** tidak."
    )
    st.write(
        "Dashboard hanya menampilkan ringkasan kelompok; ID dan baris data individu tidak ditampilkan."
    )