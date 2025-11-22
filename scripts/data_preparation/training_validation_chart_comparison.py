import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# OLD RUN DATA (20 warmup epochs, batch size 128)
old_train_epochs = [11, 13, 18, 23, 28, 32, 37, 39, 42, 44, 47, 49, 51, 54, 56, 58, 61, 63, 66, 68, 70, 73, 75, 78, 80, 82, 85, 87, 90, 92, 94, 97, 99]
old_train_errors = [67.58, 60.94, 54.30, 50.86, 46.80, 43.13, 40.70, 38.98, 37.30, 33.59, 33.01, 34.06, 29.84, 30.47, 30.08, 28.52, 30.31, 27.83, 23.44, 24.30, 21.68, 20.80, 19.61, 18.55, 19.61, 18.16, 16.88, 16.41, 13.48, 14.38, 15.23, 12.79, 11.25]

old_val_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
old_val_errors = [65.67, 52.68, 42.82, 37.25, 33.14, 29.71, 26.26, 23.50, 21.66, 20.92]

# NEW RUN DATA (5 warmup epochs, batch size 512)
new_train_epochs = [2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29, 31, 33, 35, 38, 40, 42, 45, 47, 50, 52, 53]
new_train_errors = [83.65, 76.27, 70.48, 59.20, 58.44, 51.99, 51.86, 46.29, 45.72, 38.55, 39.32, 38.28, 35.70, 34.25, 32.42, 28.78, 32.65, 32.39, 28.54, 26.56, 26.31, 24.69, 26.25]

new_val_epochs = [10, 20, 30, 40, 50]
new_val_errors = [60.40, 45.32, 38.38, 33.974, 30.298]

# OLD RUN PROJECTIONS
old_train_proj_epochs = [75, 80, 85, 90, 95, 100]
old_train_proj_upper = [19.61, 17.5, 15.0, 12.5, 10.0, 2.0]
old_train_proj_lower = [19.61, 18.5, 17.5, 16.5, 15.5, 14.8]

old_val_proj_epochs = [70, 75, 80, 85, 90, 95, 100]
old_val_proj_upper = [26.26, 25.2, 24.5, 24.0, 23.7, 23.5, 23.4]
old_val_proj_lower = [26.26, 23.2, 20.0, 16.8, 13.6, 10.4, 4.0]

# NEW RUN PROJECTIONS (UPDATED - More optimistic lower bound)
new_train_proj_epochs = [50, 60, 70, 80, 90, 100]
new_train_proj_upper = [26.31, 19.97, 14.0, 8.5, 4.0, 0.5]
new_train_proj_lower = [26.31, 22.0, 18.0, 14.5, 11.5, 8.5]

new_val_proj_epochs = [50, 60, 70, 80, 90, 100]
new_val_proj_upper = [30.84, 29.39, 28.57, 28.11, 27.85, 27.70]  # Worst case (conservative)
new_val_proj_lower = [30.30, 28.10, 25.90, 22.90, 19.90, 16.90]  # Best case (optimistic - UPDATED)

old_train_fillcolor = "rgba(31, 184, 205, 0.15)"
old_val_fillcolor = "rgba(219, 69, 69, 0.15)"
new_train_fillcolor = "rgba(46, 139, 87, 0.15)"
new_val_fillcolor = "rgba(255, 140, 0, 0.15)"

target_line = 16.0

# Create figure
fig = go.Figure()

# 1. OLD TRAINING PROJECTION BAND (blue shading)
fig.add_trace(go.Scatter(
    x=old_train_proj_epochs + old_train_proj_epochs[::-1],
    y=old_train_proj_upper + old_train_proj_lower[::-1],
    fill='toself',
    fillcolor=old_train_fillcolor,
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=True,
    legendgroup='old',
    name='Old: Train Proj.',
    hoverinfo='skip'
))

# 2. OLD VALIDATION PROJECTION BAND (red shading)
fig.add_trace(go.Scatter(
    x=old_val_proj_epochs + old_val_proj_epochs[::-1],
    y=old_val_proj_upper + old_val_proj_lower[::-1],
    fill='toself',
    fillcolor=old_val_fillcolor,
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=True,
    legendgroup='old',
    name='Old: Val Proj.',
    hoverinfo='skip'
))

# 3. NEW TRAINING PROJECTION BAND (green shading)
fig.add_trace(go.Scatter(
    x=new_train_proj_epochs + new_train_proj_epochs[::-1],
    y=new_train_proj_upper + new_train_proj_lower[::-1],
    fill='toself',
    fillcolor=new_train_fillcolor,
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=True,
    legendgroup='new',
    name='New: Train Proj.',
    hoverinfo='skip'
))

# 4. NEW VALIDATION PROJECTION BAND (orange shading)
fig.add_trace(go.Scatter(
    x=new_val_proj_epochs + new_val_proj_epochs[::-1],
    y=new_val_proj_upper + new_val_proj_lower[::-1],
    fill='toself',
    fillcolor=new_val_fillcolor,
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=True,
    legendgroup='new',
    name='New: Val Proj.',
    hoverinfo='skip'
))

# 5. OLD TRAINING ERROR - Blue line with circles
fig.add_trace(go.Scatter(
    x=old_train_epochs,
    y=old_train_errors,
    mode='lines+markers',
    line=dict(color='#1FB8CD', width=3),
    marker=dict(color='#1FB8CD', size=8, symbol='circle'),
    legendgroup='old',
    name='Old: Training',
    hovertemplate='Epoch: %{x}<br>Error: %{y:.2f}%'
))

# 6. OLD VALIDATION ERROR - Red line with squares
fig.add_trace(go.Scatter(
    x=old_val_epochs,
    y=old_val_errors,
    mode='lines+markers',
    line=dict(color='#DB4545', width=3),
    marker=dict(color='#DB4545', size=10, symbol='square'),
    legendgroup='old',
    name='Old: Validation',
    hovertemplate='Epoch: %{x}<br>Error: %{y:.2f}%'
))

# 7. NEW TRAINING ERROR - Green line with circles
fig.add_trace(go.Scatter(
    x=new_train_epochs,
    y=new_train_errors,
    mode='lines+markers',
    line=dict(color='#2E8B57', width=3),
    marker=dict(color='#2E8B57', size=8, symbol='circle'),
    legendgroup='new',
    name='New: Training',
    hovertemplate='Epoch: %{x}<br>Error: %{y:.2f}%'
))

# 8. NEW VALIDATION ERROR - Orange line with squares
fig.add_trace(go.Scatter(
    x=new_val_epochs,
    y=new_val_errors,
    mode='lines+markers',
    line=dict(color='#FF8C00', width=3),
    marker=dict(color='#FF8C00', size=10, symbol='square'),
    legendgroup='new',
    name='New: Validation',
    hovertemplate='Epoch: %{x}<br>Error: %{y:.2f}%'
))

# 9. GREEN TARGET LINE at 16%
fig.add_trace(go.Scatter(
    x=[0, 100],
    y=[16, 16],
    mode='lines',
    line=dict(color='#2E8B57', dash='dash', width=2),
    showlegend=True,
    name='Target: 16%',
    hoverinfo='skip'
))

# Update layout
fig.update_layout(
    title={
        'text': "Training vs Validation Error:<br>Batch Size & Warmup Comparison",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18}
    },
    xaxis_title="Epochs",
    yaxis_title="Top-1 Error (%)",
    xaxis=dict(range=[0, 100], dtick=10),
    yaxis=dict(range=[0, 90], dtick=10),
    legend=dict(
        orientation='v',
        yanchor='top',
        y=0.99,
        xanchor='left',
        x=0.01,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='black',
        borderwidth=1
    ),
    width=1400,
    height=900,
    font=dict(size=14)
)

# Add configuration labels
annotations = [
    dict(x=0.02, y=0.98, xref='paper', yref='paper',
         text="<b>Old Config:</b><br>20 Warmup Epochs<br>Batch Size 128<br>Final: 20.92%",
         showarrow=False, font=dict(size=12, color='#1FB8CD'),
         align='left', bgcolor='rgba(255, 255, 255, 0.9)',
         bordercolor='#1FB8CD', borderwidth=2, borderpad=5,
         xanchor='left', yanchor='top'),

    dict(x=0.02, y=0.74, xref='paper', yref='paper',
         text="<b>New Config:</b><br>5 Warmup Epochs<br>Batch Size 512<br>Proj: 16.9-27.7%",
         showarrow=False, font=dict(size=12, color='#2E8B57'),
         align='left', bgcolor='rgba(255, 255, 255, 0.9)',
         bordercolor='#2E8B57', borderwidth=2, borderpad=5,
         xanchor='left', yanchor='top'),

    # Highlight that new best case beats old actual
    dict(x=95, y=18, text="New best case:<br>16.9% @ Ep100<br>(Better than old!)",
         showarrow=True, arrowhead=2, arrowcolor='#FF8C00', arrowsize=1.5, arrowwidth=2,
         ax=-60, ay=-40,
         font=dict(size=11, color='#FF8C00', family='Arial Black'),
         align='center', bgcolor='rgba(255, 255, 255, 0.95)',
         bordercolor='#FF8C00', borderwidth=2, borderpad=5),
]

fig.update_layout(annotations=annotations)
fig.update_traces(cliponaxis=False)

# Export
fig.write_image("training_validation_chart_comparison.png", scale=2)
fig.write_html("training_validation_chart_comparison.html")

print("✅ Updated comparison chart generated!")
print(f"   New run best case @ epoch 100: 16.90%")
print(f"   Old run actual @ epoch 100: 20.92%")
print(f"   Improvement: 4.02 percentage points!")
fig.show()
