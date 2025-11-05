import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

# ==============================
# 1️⃣ Exemple de données IA
# ==============================
# Tu peux remplacer ça par tes vrais résultats
data = {
    "epoch": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "accuracy": [0.45, 0.52, 0.61, 0.68, 0.72, 0.75, 0.79, 0.81, 0.83, 0.85],
    "loss": [1.2, 1.0, 0.85, 0.7, 0.6, 0.5, 0.42, 0.36, 0.33, 0.3]
}
df = pd.DataFrame(data)

# Résumés rapides
current_accuracy = df["accuracy"].iloc[-1]
best_accuracy = df["accuracy"].max()
final_loss = df["loss"].iloc[-1]

# ==============================
# 2️⃣ Création du site Dash (avec Bootstrap CDN pour un style responsive)
# ==============================
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Résultats IA - MarioKartIA"

# ==============================
# 3️⃣ Création des graphiques (styles améliorés)
# ==============================
palette = ["#1f77b4", "#ff7f0e"]  # couleurs cohérentes

fig_accuracy = px.line(
    df,
    x="epoch",
    y="accuracy",
    title="📈 Accuracy au fil des epochs",
    markers=True,
    color_discrete_sequence=[palette[0]]
)
fig_accuracy.update_traces(mode="lines+markers", hovertemplate="Epoch %{x}<br>Accuracy: %{y:.2f}")
fig_accuracy.update_layout(
    template="plotly_white",
    margin=dict(l=20, r=20, t=50, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Arial", size=12)
)

fig_loss = px.line(
    df,
    x="epoch",
    y="loss",
    title="📉 Loss au fil des epochs",
    markers=True,
    color_discrete_sequence=[palette[1]]
)
fig_loss.update_traces(mode="lines+markers", hovertemplate="Epoch %{x}<br>Loss: %{y:.3f}")
fig_loss.update_layout(
    template="plotly_white",
    margin=dict(l=20, r=20, t=50, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

# ==============================
# 4️⃣ Mise en page du site (Bootstrap + cartes métriques)
# ==============================
header_style = {
    "background": "linear-gradient(90deg,#0d6efd,#6610f2)",
    "color": "white",
    "padding": "18px 24px",
    "borderRadius": "8px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.15)"
}

card_style = {
    "borderRadius": "8px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.06)",
    "padding": "12px",
    "background": "white"
}

app.layout = html.Div([
    html.Div([
        html.H1("Résultats de l'IA — MarioKartIA", style={"margin": 0}),
        html.P("Visualisation des performances du modèle", style={"margin": 0, "opacity": "0.9"})
    ], style=header_style, className="mb-4"),

    html.Div(className="container-fluid", children=[
        # Row métriques
        html.Div(className="row mb-4", children=[
            html.Div(className="col-md-4 mb-2", children=[
                html.Div(style=card_style, children=[
                    html.H6("Accuracy actuelle", style={"marginBottom": "6px", "color": "#333"}),
                    html.H2(f"{current_accuracy:.2f}", style={"margin": 0, "color": palette[0]})
                ])
            ]),
            html.Div(className="col-md-4 mb-2", children=[
                html.Div(style=card_style, children=[
                    html.H6("Meilleure accuracy", style={"marginBottom": "6px", "color": "#333"}),
                    html.H2(f"{best_accuracy:.2f}", style={"margin": 0, "color": "#20c997"})
                ])
            ]),
            html.Div(className="col-md-4 mb-2", children=[
                html.Div(style=card_style, children=[
                    html.H6("Loss finale", style={"marginBottom": "6px", "color": "#333"}),
                    html.H2(f"{final_loss:.3f}", style={"margin": 0, "color": palette[1]})
                ])
            ])
        ]),

        # Row graphiques responsive
        html.Div(className="row", children=[
            html.Div(className="col-lg-6 col-md-12 mb-4", children=[
                html.Div(style={**card_style, "height": "100%"}, children=[
                    dcc.Graph(figure=fig_accuracy, config={"displayModeBar": False})
                ])
            ]),
            html.Div(className="col-lg-6 col-md-12 mb-4", children=[
                html.Div(style={**card_style, "height": "100%"}, children=[
                    dcc.Graph(figure=fig_loss, config={"displayModeBar": False})
                ])
            ])
        ]),

        # Footer
        html.Div(className="row", children=[
            html.Div(className="col-12 text-muted", style={"fontSize": "12px", "paddingTop": "8px"}, children=[
                "Données simulées — remplacez par vos vrais logs de training."
            ])
        ])
    ])
], style={"fontFamily": "Inter, Arial, sans-serif", "padding": "18px", "background": "#f8f9fa"})

# ==============================
# 5️⃣ Lancement du serveur
# ==============================
if __name__ == '__main__':
    app.run(debug=True)
