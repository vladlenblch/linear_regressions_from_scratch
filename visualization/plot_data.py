import plotly.graph_objects as go

def plot_data_points(x_raw, y, title="Синтезированные данные"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_raw,
        y=y,
        mode='markers',
        name='Data points',
        marker=dict(
            size=6,
            color='#64B5F6'
        )
    ))
    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='y',
        width=700,
        height=500,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',
            zeroline=False
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    return fig
