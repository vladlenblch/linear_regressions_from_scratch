import streamlit as st
import numpy as np
from core.data import generate_linear_data
from visualization.plot_data import plot_data_points


def main():
    st.title("Линейная регрессия")
    st.markdown("""
    Реализованы разные способы обучения регрессии с нуля:
    1) MSE minimization (через аналитическое решение)
    2) MAE minimization (через subgradient descent)
    3) Gradient descent
    4) Stochastic gradient descent
    """)

    st.sidebar.header("Синтезированные данные")
    n_samples = st.sidebar.slider("Количество значений", min_value=10, max_value=500, value=100, step=10)
    noise_level = st.sidebar.slider("Уровень шума", min_value=0.0, max_value=5.0, value=1.0, step=0.05)
    true_intercept = st.sidebar.slider("Свободный коэффициент", min_value=-5.0, max_value=5.0, value=0.0, step=0.05)
    true_slope = st.sidebar.slider("Коэффициент наклона", min_value=-5.0, max_value=5.0, value=1.0, step=0.05)

    X, y, x_raw = generate_linear_data(
        n_samples=n_samples,
        noise_level=noise_level,
        true_intercept=true_intercept,
        true_slope=true_slope
    )

    st.subheader("Информация о датасете")
    st.write(f"Сгенерировано {n_samples} точек с уровнем шума {noise_level}")
    st.write(f"Исходная прямая: y = {true_intercept:.2f} + {true_slope:.2f} * x")

    fig = plot_data_points(x_raw, y)
    st.plotly_chart(fig)

    st.session_state['X'] = X
    st.session_state['y'] = y
    st.session_state['x_raw'] = x_raw
    st.session_state['true_params'] = (true_intercept, true_slope)


if __name__ == "__main__":
    main()
