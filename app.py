from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


@st.cache_data(show_spinner="Menguji kontribusi faktor dengan validasi silang...")
def calculate_multivariable_importance(data):
    excluded_columns = [
        "Attrition",
        "EmployeeId",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ]
    derived_columns = [
        column
        for column in data.columns
        if column.endswith("Label") or column.endswith("Group")
    ]
    predictors = data.drop(columns=excluded_columns + derived_columns)
    target = data["Attrition"]

    numeric_columns = predictors.select_dtypes(include="number").columns.tolist()
    categorical_columns = predictors.select_dtypes(exclude="number").columns.tolist()
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_importances = []
    fold_auc_scores = []
    for fold_number, (train_indices, test_indices) in enumerate(
        cross_validation.split(predictors, target)
    ):
        train_features = predictors.iloc[train_indices]
        test_features = predictors.iloc[test_indices]
        train_target = target.iloc[train_indices]
        test_target = target.iloc[test_indices]

        model.fit(train_features, train_target)
        fold_auc_scores.append(
            roc_auc_score(test_target, model.predict_proba(test_features)[:, 1])
        )
        permutation = permutation_importance(
            model,
            test_features,
            test_target,
            scoring="roc_auc",
            n_repeats=8,
            random_state=42 + fold_number,
            n_jobs=1,
        )
        fold_importances.append(
            pd.Series(permutation.importances_mean, index=predictors.columns)
        )

    importance_by_fold = pd.DataFrame(fold_importances)
    importance = pd.DataFrame(
        {
            "Faktor": importance_by_fold.columns,
            "Penurunan AUC": importance_by_fold.mean(axis=0).values,
            "Konsistensi positif": (importance_by_fold > 0).mean(axis=0).values,
        }
    ).sort_values("Penurunan AUC", ascending=False)
    mean_auc = sum(fold_auc_scores) / len(fold_auc_scores)
    auc_std = pd.Series(fold_auc_scores).std()
    return importance, mean_auc, auc_std


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
st.subheader("Kontribusi faktor setelah dipertimbangkan bersama")
importance, mean_auc, auc_std = calculate_multivariable_importance(employee_data)
auc_column, method_column = st.columns([1, 3])
auc_column.metric("ROC-AUC validasi silang", f"{mean_auc:.3f} +/- {auc_std:.3f}")
method_column.caption(
    "Model regresi logistik diuji dengan stratified 5-fold cross-validation. "
    "ROC-AUC merangkum kemampuan model membedakan dua kelas pada tiap fold."
)

factor_labels = {
    "Age": "Usia",
    "BusinessTravel": "Perjalanan dinas",
    "DailyRate": "Tarif harian",
    "Department": "Departemen",
    "DistanceFromHome": "Jarak dari rumah",
    "Education": "Pendidikan",
    "EducationField": "Bidang pendidikan",
    "EnvironmentSatisfaction": "Kepuasan lingkungan kerja",
    "Gender": "Gender",
    "HourlyRate": "Tarif per jam",
    "JobInvolvement": "Keterlibatan kerja",
    "JobLevel": "Level jabatan",
    "JobRole": "Peran pekerjaan",
    "JobSatisfaction": "Kepuasan kerja",
    "MaritalStatus": "Status pernikahan",
    "MonthlyIncome": "Pendapatan bulanan",
    "MonthlyRate": "Tarif bulanan",
    "NumCompaniesWorked": "Jumlah perusahaan sebelumnya",
    "OverTime": "Overtime",
    "PercentSalaryHike": "Kenaikan gaji",
    "PerformanceRating": "Penilaian kinerja",
    "RelationshipSatisfaction": "Kepuasan hubungan kerja",
    "StockOptionLevel": "Level opsi saham",
    "TotalWorkingYears": "Total pengalaman kerja",
    "TrainingTimesLastYear": "Frekuensi pelatihan tahun lalu",
    "WorkLifeBalance": "Keseimbangan kehidupan-kerja",
    "YearsAtCompany": "Masa kerja di perusahaan",
    "YearsInCurrentRole": "Masa kerja di peran saat ini",
    "YearsSinceLastPromotion": "Tahun sejak promosi terakhir",
    "YearsWithCurrManager": "Masa kerja dengan manajer saat ini",
}
top_importance = importance.head(10).copy()
top_importance["Faktor"] = top_importance["Faktor"].map(factor_labels).fillna(
    top_importance["Faktor"]
)
top_importance["Konsistensi positif"] = (
    top_importance["Konsistensi positif"] * 100
).map(lambda value: f"{value:.0f}%")
top_importance["Penurunan AUC"] = top_importance["Penurunan AUC"].map(
    lambda value: f"{value:.4f}"
)

importance_chart = px.bar(
    top_importance.sort_values("Penurunan AUC"),
    x="Penurunan AUC",
    y="Faktor",
    orientation="h",
    color="Konsistensi positif",
    labels={
        "Penurunan AUC": "Penurunan rata-rata ROC-AUC setelah faktor diacak",
        "Konsistensi positif": "Fold dengan kontribusi positif",
    },
    color_continuous_scale="Tealrose",
)
importance_chart.update_layout(
    margin=dict(l=10, r=15, t=10, b=10),
    height=400,
    coloraxis_colorbar_title="Fold (%)",
)
st.plotly_chart(importance_chart, width="stretch")
st.dataframe(
    top_importance[["Faktor", "Penurunan AUC", "Konsistensi positif"]],
    hide_index=True,
    width="stretch",
)
st.caption(
    "Semakin besar penurunan AUC setelah satu faktor diacak, semakin besar kontribusi "
    "prediktif relatif faktor itu pada model ini. Ini bukan arah pengaruh atau bukti sebab-akibat; "
    "faktor yang saling berkaitan dapat berbagi kontribusi."
)

st.divider()
st.subheader("Pola per faktor secara terpisah")

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