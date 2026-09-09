# """
# Application Streamlit — Prédiction de l'état d'un portable
# Conversion directe de l'application Gradio d'origine.

# Lancement en local :  streamlit run app.py
# """

# import numpy as np
# import pandas as pd
# import joblib as jb
# import streamlit as st


# # Configuration de la page
# st.set_page_config(
#     page_title="Prédiction de l'état d'un portable",
#     page_icon="📱",
#     layout="centered",
# )

# DESCRIPTION = (
#     "Ce modèle de machine learning permet de prédire l'état d'un portable en partant "
#     "du prix, de l'adresse, de la marque, de la dimension de l'écran, du nombre de RAM "
#     "et du stockage."
# )


# # Chargement des artefacts (mis en cache : chargés une seule fois)

# @st.cache_resource
# def load_artifacts():
#     encoders = jb.load("encoders.joblib")   # encodeurs (adresse, marque)
#     uniques = jb.load("uniques.joblib")     # valeurs uniques
#     scaler = jb.load("scaler.joblib")       # normaliseur
#     xgb = jb.load("xgb_model.joblib")       # modèle
#     return encoders, uniques, scaler, xgb


# encoders, uniques, scaler, xgb = load_artifacts()
# clasnames = uniques[2]  # noms des classes



# # Fonction de prédiction simple

# def Pred_func(prix, adresse, marque, dim_ecr, ram, stockage):
#     # Encoder l'adresse et la marque
#     adresse = encoders[0].transform([adresse])[0]
#     marque = encoders[1].transform([marque])[0]
#     # Vecteur des valeurs numériques
#     x_new = np.array([prix, adresse, marque, dim_ecr, ram, stockage])
#     x_new = x_new.reshape(1, -1)  # conversion en un tableau 2D
#     # Normaliser les données
#     x_new = scaler.transform(x_new)
#     # Prédire
#     y_pred = xgb.predict(x_new)
#     return clasnames[y_pred[0]]



# # Fonction de prédiction multiple

# def Pred_func_csv(file):
#     # Lire le fichier csv
#     df = pd.read_csv(file)
#     predictions = []
#     # Boucle sur les lignes du dataframe
#     for row in df.iloc[:, :].values:
#         # prédiction simple
#         y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4], row[5])
#         predictions.append(y_pred)
#     df["etat"] = predictions
#     return df



# # Interface

# st.title("📱 Prédiction de l'état d'un portable")

# onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# # ----------------------------- Onglet 1 -------------------------------
# with onglet1:
#     st.subheader("Prédire l'état d'un portable avec une entrée")
#     st.write(DESCRIPTION)

#     with st.form("formulaire_simple"):
#         col1, col2 = st.columns(2)
#         with col1:
#             prix = st.number_input("Prix", value=0.0, step=1000.0, format="%.2f")
#             adresse = st.selectbox("Adresse", options=list(uniques[0]))
#             marque = st.selectbox("Marque", options=list(uniques[1]))
#         with col2:
#             dim_ecr = st.number_input("Dimension écran", value=0.0, step=0.1, format="%.2f")
#             ram = st.number_input("Nombre de RAM", value=0.0, step=1.0, format="%.2f")
#             stockage = st.number_input("Stockage", value=0.0, step=1.0, format="%.2f")

#         soumettre = st.form_submit_button("Prédire", type="primary")

#     if soumettre:
#         try:
#             resultat = Pred_func(prix, adresse, marque, dim_ecr, ram, stockage)
#             st.success(f"**État du portable :** {resultat}")
#         except Exception as e:
#             st.error(f"Erreur lors de la prédiction : {e}")

# # ----------------------------- Onglet 2 -------------------------------
# with onglet2:
#     st.subheader("Prédire l'état d'un portable avec plusieurs entrées")
#     st.write(DESCRIPTION)
#     st.caption(
#         "Le fichier CSV doit contenir, dans cet ordre, les colonnes : "
#         "prix, adresse, marque, dimension écran, RAM, stockage."
#     )

#     fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

#     if fichier is not None:
#         try:
#             with st.spinner("Prédictions en cours…"):
#                 df_resultat = Pred_func_csv(fichier)

#             st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
#             st.dataframe(df_resultat, use_container_width=True)

#             st.download_button(
#                 label="⬇️ Télécharger le fichier CSV",
#                 data=df_resultat.to_csv(index=False).encode("utf-8"),
#                 file_name="predictions.csv",
#                 mime="text/csv",
#                 type="primary",
#             )
#         except Exception as e:
#             st.error(f"Erreur lors du traitement du fichier : {e}")
"""
Application Streamlit — Prédiction de l'état d'un portable
Version améliorée : design responsive (rouge foncé / chocolat),
indicateurs, graphique camembert et tableau des scores.

Lancement en local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ------------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Prédiction de l'état d'un portable",
    page_icon="📱",
    layout="wide",
)

DESCRIPTION = (
    "Ce modèle de machine learning permet de prédire l'état d'un portable en partant "
    "du prix, de l'adresse, de la marque, de la dimension de l'écran, du nombre de RAM "
    "et du stockage."
)

# Palette de couleurs : rouge foncé + chocolat
COLOR_DARK_RED = "#7A0C0C"
COLOR_DARK_RED_2 = "#5C0909"
COLOR_CHOCOLATE = "#7B3F00"
COLOR_CHOCOLATE_LIGHT = "#A0522D"
COLOR_CREAM = "#F5E9DC"
COLOR_TEXT = "#2E1503"

PALETTE_CAMEMBERT = [
    COLOR_DARK_RED, COLOR_CHOCOLATE, COLOR_CHOCOLATE_LIGHT,
    "#B85C38", "#C1440E", "#4A0404", "#8B5A2B",
]


# ------------------------------------------------------------------
# CSS personnalisé — responsive, rouge foncé / chocolat
# ------------------------------------------------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
        /* ---------- Fond général ---------- */
        .stApp {{
            background: linear-gradient(160deg, {COLOR_CREAM} 0%, #EDD9C0 100%);
        }}

        /* ---------- Titre principal ---------- */
        .app-header {{
            background: linear-gradient(120deg, {COLOR_DARK_RED} 0%, {COLOR_CHOCOLATE} 100%);
            padding: 28px 24px;
            border-radius: 18px;
            color: white;
            text-align: center;
            box-shadow: 0 6px 18px rgba(90, 30, 10, 0.35);
            margin-bottom: 24px;
        }}
        .app-header h1 {{
            margin: 0;
            font-size: clamp(1.4rem, 3vw, 2.2rem);
            letter-spacing: 0.5px;
        }}
        .app-header p {{
            margin: 8px 0 0 0;
            font-size: clamp(0.85rem, 1.4vw, 1rem);
            opacity: 0.92;
        }}

        /* ---------- Onglets ---------- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {COLOR_CREAM};
            border-radius: 10px 10px 0 0;
            padding: 10px 18px;
            color: {COLOR_CHOCOLATE};
            font-weight: 600;
            border: 1px solid {COLOR_CHOCOLATE_LIGHT};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {COLOR_DARK_RED} !important;
            color: white !important;
        }}

        /* ---------- Cartes indicateurs (KPI) ---------- */
        .kpi-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin: 18px 0;
        }}
        .kpi-card {{
            flex: 1 1 200px;
            background: white;
            border-left: 6px solid {COLOR_DARK_RED};
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 3px 10px rgba(90, 30, 10, 0.12);
            transition: transform 0.15s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-3px);
        }}
        .kpi-card .kpi-label {{
            font-size: 0.85rem;
            color: {COLOR_CHOCOLATE};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}
        .kpi-card .kpi-value {{
            font-size: 1.7rem;
            font-weight: 800;
            color: {COLOR_DARK_RED_2};
            margin-top: 4px;
        }}

        /* ---------- Boutons ---------- */
        div.stButton > button, button[kind="primary"] {{
            background: linear-gradient(120deg, {COLOR_DARK_RED} 0%, {COLOR_CHOCOLATE} 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            padding: 10px 20px !important;
            box-shadow: 0 4px 10px rgba(90, 30, 10, 0.3);
        }}
        div.stButton > button:hover, button[kind="primary"]:hover {{
            filter: brightness(1.08);
        }}

        /* ---------- Résultat prédiction ---------- */
        .result-box {{
            background: linear-gradient(120deg, {COLOR_CHOCOLATE} 0%, {COLOR_DARK_RED} 100%);
            color: white;
            padding: 18px 22px;
            border-radius: 14px;
            font-size: 1.2rem;
            font-weight: 700;
            text-align: center;
            box-shadow: 0 5px 14px rgba(90, 30, 10, 0.3);
            margin: 14px 0;
        }}

        /* ---------- Tableau des scores ---------- */
        .score-table thead tr th {{
            background-color: {COLOR_DARK_RED} !important;
            color: white !important;
        }}
        .score-table tbody tr:nth-child(even) {{
            background-color: {COLOR_CREAM} !important;
        }}

        /* ---------- Responsive : petits écrans ---------- */
        @media (max-width: 640px) {{
            .kpi-card {{
                flex: 1 1 100%;
            }}
            .app-header {{
                padding: 18px 14px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card_html(label, value):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """


# ------------------------------------------------------------------
# Chargement des artefacts (mis en cache : chargés une seule fois)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")   # encodeurs (adresse, marque)
    uniques = jb.load("uniques.joblib")     # valeurs uniques
    scaler = jb.load("scaler.joblib")       # normaliseur
    xgb = jb.load("xgb_model.joblib")       # modèle
    return encoders, uniques, scaler, xgb


encoders, uniques, scaler, xgb = load_artifacts()
clasnames = uniques[2]  # noms des classes


# ------------------------------------------------------------------
# Fonction de prédiction simple + scores (probabilités)
# ------------------------------------------------------------------
def prepare_input(prix, adresse, marque, dim_ecr, ram, stockage):
    adresse_enc = encoders[0].transform([adresse])[0]
    marque_enc = encoders[1].transform([marque])[0]
    x_new = np.array([prix, adresse_enc, marque_enc, dim_ecr, ram, stockage])
    x_new = x_new.reshape(1, -1)
    x_new = scaler.transform(x_new)
    return x_new


def Pred_func(prix, adresse, marque, dim_ecr, ram, stockage):
    """Prédiction de la classe (compatibilité avec la version d'origine)."""
    x_new = prepare_input(prix, adresse, marque, dim_ecr, ram, stockage)
    y_pred = xgb.predict(x_new)
    return clasnames[y_pred[0]]


def Pred_func_with_scores(prix, adresse, marque, dim_ecr, ram, stockage):
    """Prédiction de la classe + scores (probabilités) par classe."""
    x_new = prepare_input(prix, adresse, marque, dim_ecr, ram, stockage)
    y_pred = xgb.predict(x_new)
    classe = clasnames[y_pred[0]]

    if hasattr(xgb, "predict_proba"):
        proba = xgb.predict_proba(x_new)[0]
    else:
        # Repli si le modèle ne fournit pas de probabilités
        proba = np.zeros(len(clasnames))
        proba[y_pred[0]] = 1.0

    scores = {str(clasnames[i]): float(proba[i]) for i in range(len(clasnames))}
    return classe, scores


# ------------------------------------------------------------------
# Fonction de prédiction multiple (CSV) + scores
# ------------------------------------------------------------------
def Pred_func_csv(file):
    df = pd.read_csv(file)
    predictions = []
    scores_rows = []

    for row in df.iloc[:, :].values:
        classe, scores = Pred_func_with_scores(
            row[0], row[1], row[2], row[3], row[4], row[5]
        )
        predictions.append(classe)
        scores_rows.append(scores)

    df["etat"] = predictions
    scores_df = pd.DataFrame(scores_rows)
    scores_df.columns = [f"score_{c}" for c in scores_df.columns]
    df_final = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
    return df_final


# ------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------
inject_css()

st.markdown(
    f"""
    <div class="app-header">
        <h1>📱 Prédiction de l'état d'un portable</h1>
        <p>{DESCRIPTION}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

onglet1, onglet2 = st.tabs(["🔎 Prédiction simple", "📂 Prédiction multiple"])

# =====================================================================
# Onglet 1 — Prédiction simple
# =====================================================================
with onglet1:
    st.subheader("Prédire l'état d'un portable avec une entrée")

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)
        with col1:
            prix = st.number_input("Prix", value=0.0, step=1000.0, format="%.2f")
            adresse = st.selectbox("Adresse", options=list(uniques[0]))
            marque = st.selectbox("Marque", options=list(uniques[1]))
        with col2:
            dim_ecr = st.number_input("Dimension écran", value=0.0, step=0.1, format="%.2f")
            ram = st.number_input("Nombre de RAM", value=0.0, step=1.0, format="%.2f")
            stockage = st.number_input("Stockage", value=0.0, step=1.0, format="%.2f")

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            classe, scores = Pred_func_with_scores(
                prix, adresse, marque, dim_ecr, ram, stockage
            )

            # ---- Résultat principal ----
            st.markdown(
                f'<div class="result-box">État prédit : {classe}</div>',
                unsafe_allow_html=True,
            )

            # ---- Indicateurs (KPI) ----
            classe_score = scores.get(classe, 0.0)
            deuxieme = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            deuxieme_label, deuxieme_val = (deuxieme[1] if len(deuxieme) > 1 else ("—", 0.0))

            st.markdown(
                f"""
                <div class="kpi-container">
                    {kpi_card_html("Classe prédite", classe)}
                    {kpi_card_html("Confiance", f"{classe_score*100:.1f}%")}
                    {kpi_card_html("2ᵉ classe possible", deuxieme_label)}
                    {kpi_card_html("Nb. classes", len(scores))}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_graph, col_table = st.columns([1, 1])

            # ---- Graphique camembert ----
            with col_graph:
                fig = px.pie(
                    names=list(scores.keys()),
                    values=list(scores.values()),
                    color_discrete_sequence=PALETTE_CAMEMBERT,
                    hole=0.35,
                    title="Répartition des scores par classe",
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(
                    font_color=COLOR_TEXT,
                    title_font_color=COLOR_DARK_RED_2,
                    legend_title_text="Classes",
                    margin=dict(t=60, b=10, l=10, r=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ---- Tableau des scores ----
            with col_table:
                st.markdown("**Tableau des scores par classe**")
                scores_df = (
                    pd.DataFrame(
                        {"Classe": list(scores.keys()), "Score": list(scores.values())}
                    )
                    .sort_values("Score", ascending=False)
                    .reset_index(drop=True)
                )
                scores_df["Score (%)"] = (scores_df["Score"] * 100).round(2)

                st.dataframe(
                    scores_df[["Classe", "Score (%)"]],
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# =====================================================================
# Onglet 2 — Prédiction multiple
# =====================================================================
with onglet2:
    st.subheader("Prédire l'état d'un portable avec plusieurs entrées")
    st.caption(
        "Le fichier CSV doit contenir, dans cet ordre, les colonnes : "
        "prix, adresse, marque, dimension écran, RAM, stockage."
    )

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = Pred_func_csv(fichier)

            nb_lignes = len(df_resultat)
            classe_majoritaire = df_resultat["etat"].mode().iloc[0]
            nb_classes_distinctes = df_resultat["etat"].nunique()
            score_cols = [c for c in df_resultat.columns if c.startswith("score_")]
            confiance_moy = (
                df_resultat.apply(
                    lambda r: r[f"score_{r['etat']}"] if f"score_{r['etat']}" in df_resultat.columns else np.nan,
                    axis=1,
                ).mean()
                * 100
            )

            # ---- Indicateurs (KPI) ----
            st.markdown(
                f"""
                <div class="kpi-container">
                    {kpi_card_html("Prédictions réalisées", nb_lignes)}
                    {kpi_card_html("État majoritaire", classe_majoritaire)}
                    {kpi_card_html("États distincts", nb_classes_distinctes)}
                    {kpi_card_html("Confiance moyenne", f"{confiance_moy:.1f}%")}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_graph, col_table = st.columns([1, 1])

            # ---- Graphique camembert : répartition des états ----
            with col_graph:
                repartition = df_resultat["etat"].value_counts().reset_index()
                repartition.columns = ["etat", "count"]
                fig2 = px.pie(
                    repartition,
                    names="etat",
                    values="count",
                    color_discrete_sequence=PALETTE_CAMEMBERT,
                    hole=0.35,
                    title="Répartition des états prédits",
                )
                fig2.update_traces(textposition="inside", textinfo="percent+label")
                fig2.update_layout(
                    font_color=COLOR_TEXT,
                    title_font_color=COLOR_DARK_RED_2,
                    legend_title_text="État",
                    margin=dict(t=60, b=10, l=10, r=10),
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ---- Tableau récapitulatif des scores moyens par classe ----
            with col_table:
                st.markdown("**Score moyen par classe (toutes lignes)**")
                if score_cols:
                    moyennes = df_resultat[score_cols].mean().sort_values(ascending=False)
                    moyennes_df = pd.DataFrame(
                        {
                            "Classe": [c.replace("score_", "") for c in moyennes.index],
                            "Score moyen (%)": (moyennes.values * 100).round(2),
                        }
                    )
                    st.dataframe(
                        moyennes_df, use_container_width=True, hide_index=True
                    )

            st.markdown("---")
            st.markdown("**Détail ligne par ligne (données + état + scores)**")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")

