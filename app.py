import streamlit as st
import numpy as np
from core.data import generate_linear_data
from visualization.plot_data import plot_data_points
from visualization.plot_regression import plot_regression
# from core.ols import OLSRegression
from core.ols_sklearn import SklearnOLSRegression


def main():
    st.title("Линейная регрессия")
    st.markdown("""
    Реализованы разные способы обучения регрессии с нуля:
    1) OLS - MSE minimization (через аналитическое решение)
    2) MAE minimization (через subgradient descent)
    3) Gradient descent
    4) Stochastic gradient descent
    """)

    st.sidebar.header("Синтезированные данные")
    n_samples = st.sidebar.slider("Количество значений", min_value=10, max_value=500, value=100, step=5)
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

    # my_ols = OLSRegression().fit(X, y)
    sklearn_ols = SklearnOLSRegression().fit(X, y)

    st.header("1) OLS - MSE minimization (аналитическое решение)")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Моя реализация")
        st.write("позже")
        # fig_my = plot_regression(x_raw, y, my_ols)
        # st.plotly_chart(fig_my)

    with col2:
        st.subheader("Sklearn")
        fig_sk = plot_regression(x_raw, y, sklearn_ols)
        st.plotly_chart(fig_sk)

    st.header("2) MAE minimization (subgradient descent)")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Моя реализация")
        st.write("позже")

    with col2:
        st.subheader("Sklearn")
        st.write("позже")

    st.header("3) Gradient descent")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Моя реализация")
        st.write("позже")

    with col2:
        st.subheader("Sklearn")
        st.write("позже")

    st.header("4) Stochastic gradient descent")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Моя реализация")
        st.write("позже")

    with col2:
        st.subheader("Sklearn")
        st.write("позже")

    st.session_state['X'] = X
    st.session_state['y'] = y
    st.session_state['x_raw'] = x_raw
    st.session_state['true_params'] = (true_intercept, true_slope)


if __name__ == "__main__":
    main()
