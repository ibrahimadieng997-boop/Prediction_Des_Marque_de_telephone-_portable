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
import gradio as gr
import pandas as pd
import numpy as np
import joblib as jb
import plotly.express as px

# --- CHARGEMENT DES OBJETS ENTRAÎNÉS ---
# Ces 4 fichiers doivent être dans le même dossier que ce script.
encoders = jb.load('encoders.joblib')   # encoders[0] = adresse, encoders[1] = marque
uniques = jb.load('uniques.joblib')     # uniques[0] = adresses, uniques[1] = marques, uniques[2] = classnames
scaler = jb.load('scaler.joblib')
rf = jb.load('rf_model.joblib')

clasnames = uniques[2]

FEATURE_NAMES = ['Prix', 'Adresse', 'Marque', 'Écran', 'RAM', 'Stockage']


# --- FONCTION DE PRÉDICTION UNIQUE ---
def predict_one(prix, adresse, marque, ecran, ram, stockage):
    """Retourne (etat_predit, confiance_%) pour une observation."""
    adresse_enc = encoders[0].transform([adresse])[0]
    marque_enc = encoders[1].transform([marque])[0]

    x_new = np.array([prix, adresse_enc, marque_enc, ecran, ram, stockage], dtype=float)
    x_new = x_new.reshape(1, -1)
    x_new = scaler.transform(x_new)

    y_pred = rf.predict(x_new)[0]
    proba = rf.predict_proba(x_new)[0]
    confiance = round(float(np.max(proba)) * 100, 1)
    etat_predit = clasnames[y_pred]

    return etat_predit, confiance


# --- CALLBACK GRADIO : ONGLET PRÉDICTION UNIQUE ---
def Pred_func(prix, adresse, marque, ecran, ram, stockage):
    etat_predit, confiance = predict_one(prix, adresse, marque, ecran, ram, stockage)

    # Score du modèle en général (accuracy sur le train/test), à adapter si tu l'as stocké quelque part.
    # Ici on affiche la confiance du modèle sur cette prédiction précise à la place.
    kpi_html = f"""
    <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px;">
        <div style="flex: 1; min-width: 180px; background: #3B0A11; color: #FFFFFF; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); text-align: center; border-bottom: 4px solid #D4AF37;">
            <div style="font-size: 24px; margin-bottom: 5px;">📱</div>
            <div style="font-size: 11px; text-transform: uppercase; color: #FFFFFF; font-weight: bold; letter-spacing: 0.5px;">État Prédit</div>
            <div style="font-size: 20px; font-weight: bold; color: #FFFFFF; margin-top: 5px;">{etat_predit}</div>
        </div>
        <div style="flex: 1; min-width: 180px; background: #5C131D; color: #FFFFFF; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); text-align: center; border-bottom: 4px solid #C0392B;">
            <div style="font-size: 24px; margin-bottom: 5px;">⚡</div>
            <div style="font-size: 11px; text-transform: uppercase; color: #FFFFFF; font-weight: bold; letter-spacing: 0.5px;">Confiance</div>
            <div style="font-size: 20px; font-weight: bold; color: #FFFFFF; margin-top: 5px;">{confiance}%</div>
        </div>
    </div>
    """

    importance_data = {
        'Variable': FEATURE_NAMES,
        'Poids (%)': (rf.feature_importances_ * 100).round(1)
    }

    chocolat_rouge_palette = ['#2D080D', '#3B0A11', '#4A0E17', '#5C131D', '#7A1C28', '#9E2A2B']

    fig_pie = px.pie(
        importance_data,
        values='Poids (%)',
        names='Variable',
        title="Importance relative des variables dans la prédiction",
        color_discrete_sequence=chocolat_rouge_palette
    )
    fig_pie.update_layout(
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(color="#3B0A11", size=14)
    )

    df_recap = pd.DataFrame({
        "Caractéristique": ["Prix", "Adresse", "Marque", "Dimension Écran", "RAM", "Stockage", "Prédiction", "Score Confiance"],
        "Valeur": [f"{prix} CFA", str(adresse), str(marque), f"{ecran} pouces", f"{ram} Go", f"{stockage} Go", etat_predit, f"{confiance}%"]
    })

    return kpi_html, fig_pie, df_recap


# --- CALLBACK GRADIO : ONGLET PRÉDICTION PAR LOT (CSV) ---
def Pred_func_csv(file):
    # Avec type="filepath", `file` est directement un chemin (str)
    df = pd.read_csv(file)

    etats = []
    confiances = []
    for row in df.iloc[:, :6].values:
        etat, conf = predict_one(row[0], row[1], row[2], row[3], row[4], row[5])
        etats.append(etat)
        confiances.append(f"{conf}%")

    df['Etat_Predit'] = etats
    df['Score_Confiance'] = confiances

    output_path = "predictions_resultats.csv"
    df.to_csv(output_path, index=False)

    kpi_batch = f"""
    <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px;">
        <div style="flex: 1; min-width: 180px; background: #3B0A11; color: #FFFFFF; padding: 15px; border-radius: 10px; text-align: center; border-bottom: 4px solid #D4AF37;">
            <div style="font-size: 24px;">📁</div>
            <div style="font-size: 11px; text-transform: uppercase; color: #FFFFFF; font-weight: bold;">Lignes traitées</div>
            <div style="font-size: 20px; font-weight: bold; color: #FFFFFF; margin-top: 5px;">{len(df)}</div>
        </div>
        <div style="flex: 1; min-width: 180px; background: #4A0E17; color: #FFFFFF; padding: 15px; border-radius: 10px; text-align: center; border-bottom: 4px solid #27AE60;">
            <div style="font-size: 24px;">✅</div>
            <div style="font-size: 11px; text-transform: uppercase; color: #FFFFFF; font-weight: bold;">Statut</div>
            <div style="font-size: 20px; font-weight: bold; color: #FFFFFF; margin-top: 5px;">Succès</div>
        </div>
    </div>
    """

    return output_path, df.head(5), kpi_batch


# --- STYLES CSS SUR MESURE ---
custom_css = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #FAFAFA;
}

.header-box {
    text-align: center;
    padding: 24px;
    background: linear-gradient(135deg, #1A0306 0%, #3B0A11 50%, #5C131D 100%);
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(45, 8, 13, 0.4);
}

.header-box h1 {
    color: #FFFFFF !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}
.header-box p {
    color: #FFFFFF !important;
    font-size: 14px !important;
}

.gradio-container h3 {
    color: #3B0A11 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #5C131D;
    padding-bottom: 6px;
    margin-bottom: 12px;
}

button.primary-btn, button.primary {
    background: linear-gradient(135deg, #3B0A11 0%, #5C131D 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

button.primary-btn:hover, button.primary:hover {
    background: linear-gradient(135deg, #4A0E17 0%, #7A1C28 100%) !important;
    box-shadow: 0 4px 10px rgba(59, 10, 17, 0.5) !important;
}
"""

# uniques[0] = liste des adresses possibles, uniques[1] = liste des marques possibles
adresses_choix = list(uniques[0])
marques_choix = list(uniques[1])

# --- STRUCTURE GRADIO BLOCKS ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="red"), css=custom_css) as demo:

    gr.HTML("""
        <div class="header-box">
            <h1>📱 Tableau de bord d'Évaluation & Prédiction d'État de Portables</h1>
            <p>Estimez l'état d'un téléphone portable à l'aide de notre modèle d'apprentissage automatique.</p>
        </div>
    """)

    with gr.Tabs():
        # --- ONGLET 1: PREDICTION INDIVIDUELLE ---
        with gr.TabItem("📊 Prédiction Unique"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Caractéristiques du Portable")
                    prix = gr.Number(label="Prix en CFA", value=350)
                    adresse = gr.Dropdown(choices=adresses_choix, label="Adresse", value=adresses_choix[0])
                    marque = gr.Dropdown(choices=marques_choix, label="Marque", value=marques_choix[0])
                    ecran = gr.Number(label="Dimension écran (pouces)", value=6.1)
                    ram = gr.Number(label="RAM (Go)", value=8)
                    stockage = gr.Number(label="Stockage (Go)", value=128)
                    btn_predict = gr.Button("🚀 Lancer la prédiction", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### 📈 Résultat de l'Analyse")
                    kpi_output = gr.HTML("<p style='text-align:center; color:#5C131D; font-weight:600;'>Remplissez le formulaire et cliquez sur 'Lancer la prédiction'.</p>")
                    with gr.Row():
                        plot_output = gr.Plot(label="Répartition & Importance")
                        table_output = gr.Dataframe(label="Récapitulatif des données & score", interactive=False)

            btn_predict.click(
                fn=Pred_func,
                inputs=[prix, adresse, marque, ecran, ram, stockage],
                outputs=[kpi_output, plot_output, table_output]
            )

        # --- ONGLET 2: PREDICTION PAR LOT (CSV) ---
        with gr.TabItem("📁 Prédiction Multiple (Batch CSV)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Importer vos données")
                    gr.Markdown("Colonnes attendues, dans l'ordre : `prix, adresse, marque, ecran, ram, stockage`")
                    file_input = gr.File(label="Fichier CSV d'entrée", file_types=[".csv"], type="filepath")
                    btn_batch = gr.Button("⚙️ Traiter le fichier", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### 📥 Résultats du traitement")
                    batch_kpi = gr.HTML()
                    file_output = gr.File(label="Télécharger le fichier complété")
                    table_batch = gr.Dataframe(label="Aperçu des 5 premières lignes traitées", interactive=False)

            btn_batch.click(
                fn=Pred_func_csv,
                inputs=[file_input],
                outputs=[file_output, table_batch, batch_kpi]
            )

# Lancement de l'application
if __name__ == "__main__":
    demo.launch(share=True)
