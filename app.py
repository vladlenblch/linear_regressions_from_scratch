import streamlit as st

from visualization.plot_data import plot_data_points
from visualization.plot_regression import plot_regression

from utils.metrics import mse
from utils.data import generate_linear_data

from my_models.ols import MyOLSRegression
from my_models.sgd import MySGDRegressor

from sklearn_models.ols_sklearn import SklearnOLSRegression
from sklearn_models.sgd_sklearn import SklearnSGDRegressor


def main():
    st.set_page_config(layout="wide")
    st.title("Линейная регрессия")
    st.markdown("""
    Реализованы разные способы обучения регрессии с нуля:
    1) OLS - MSE minimization (через аналитическое решение)
    2) Stochastic gradient descent
    3) MAE minimization (через subgradient descent)
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

    st.subheader("Информация о данных")
    st.write(f"Сгенерировано {n_samples} точек с уровнем шума {noise_level}")
    st.write(f"Исходная прямая: y = {true_intercept:.2f} + {true_slope:.2f}x")

    fig = plot_data_points(x_raw, y)
    st.plotly_chart(fig)

    st.header("1) OLS - MSE minimization")
    st.subheader("Аналитическое решение")
    col1, col2 = st.columns(2)

    my_ols = MyOLSRegression().fit(X, y)
    sklearn_ols = SklearnOLSRegression().fit(X, y)

    y_pred_ols_my = my_ols.predict(X)
    y_pred_ols_sklearn = sklearn_ols.predict(X).tolist()

    mse_ols_my = mse(y, y_pred_ols_my)
    mse_ols_sklearn = mse(y, y_pred_ols_sklearn)

    with col1:
        st.subheader("Моя реализация")
        fig_ols_my = plot_regression(x_raw, y, my_ols)
        st.plotly_chart(fig_ols_my)
        w0_ols_my, w1_ols_my = my_ols.weights
        st.write(f"Уравнение: y = {w0_ols_my:.3f} + {w1_ols_my:.3f}x")
        st.write(f"MSE: {mse_ols_my:.5f}")

    with col2:
        st.subheader("Sklearn")
        fig_ols_sklearn = plot_regression(x_raw, y, sklearn_ols)
        st.plotly_chart(fig_ols_sklearn)
        w0_ols_sklearn, w1_ols_sklearn = sklearn_ols.weights
        st.write(f"Уравнение: y = {w0_ols_sklearn:.3f} + {w1_ols_sklearn:.3f}x")
        st.write(f"MSE: {mse_ols_sklearn:.5f}")

    st.header("2) Stochastic gradient descent")
    col1, col2 = st.columns(2)

    my_sgd = MySGDRegressor().fit(X, y)
    sklearn_sgd = SklearnSGDRegressor().fit(X, y)

    y_pred_sgd_my = my_sgd.predict(X)
    y_pred_sgd_sklearn = sklearn_sgd.predict(X).tolist()

    mse_sgd_my = mse(y, y_pred_sgd_my)
    mse_sgd_sklearn = mse(y, y_pred_sgd_sklearn)

    with col1:
        st.subheader("Моя реализация")
        fig_sgd_my = plot_regression(x_raw, y, my_sgd)
        st.plotly_chart(fig_sgd_my)
        w0_sgd_my, w1_sgd_my = my_sgd.weights
        st.write(f"Уравнение: y = {w0_sgd_my:.3f} + {w1_sgd_my:.3f}x")
        st.write(f"MSE: {mse_sgd_my:.5f}")

    with col2:
        st.subheader("Sklearn")
        fig_sgd_sklearn = plot_regression(x_raw, y, sklearn_sgd)
        st.plotly_chart(fig_sgd_sklearn)
        w0_sgd_sklearn, w1_sgd_sklearn = sklearn_sgd.weights
        st.write(f"Уравнение: y = {w0_sgd_sklearn:.3f} + {w1_sgd_sklearn:.3f}x")
        st.write(f"MSE: {mse_sgd_sklearn:.5f}")

    st.header("3) MAE minimization)")
    st.subheader("Subgradient descent")
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
