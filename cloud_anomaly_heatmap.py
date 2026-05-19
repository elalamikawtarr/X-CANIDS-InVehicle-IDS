import plotly.express as px

# 1. Préparation des données pour le graphique
feature_names = df_brut.select_dtypes(include=[np.number]).columns
df_plot = pd.DataFrame({
    'Signal': feature_names,
    'MSE_Error': errors
}).sort_values(by='MSE_Error', ascending=False)

# 2. Création du graphique interactif
fig = px.bar(df_plot.head(15), 
             x='MSE_Error', 
             y='Signal', 
             orientation='h',
             title="<b>X-CANIDS : Diagnostic d'Anomalie par Signal</b>",
             labels={'MSE_Error': 'Score d\'Erreur (MSE)'},
             color='MSE_Error',
             color_continuous_scale='OrRd')

fig.update_layout(
    template='plotly_dark',
    yaxis={'categoryorder':'total ascending'},
    title_x=0.5
)

fig.show()
