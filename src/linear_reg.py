
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from hashlib import sha256
import os
import numpy as np

# _SCENARIO = [[["hcan","lidar_mean_all"],["hveg", "lidar_mean_VG_all"]],
#             [["hcan","lidar_mean"],["hveg", "lidar_mean_VG"]],
#             [["hcan_sd","lidar_std_all"],["hcan_sd", "lidar_std_VG_all"]]
#         ]

# _AX_TITLES  = ["Végétation et sol", "Végétation seule"]
# _SUP_TITLES = ["Stratégie fenêtre 10x10", "Stratégie diagonale 2x10", "Stratégie fenêtre 10x10", "Stratégie fenêtre 10x10", "Stratégie diagonale 2x10"]
# _AX_LABELS  = [["Relevé hauteur moyenne (cm)", "Moyenne lidar (cm)"],
#                 ["Relevé hauteur moyenne (cm)", "Moyenne lidar (cm)"],
#                 ["Relevé écart-type hauteur (cm)", "Ecart-type lidar (cm)"],
#                 ]

_SCENARIO = [[["Hmean_canopy","lidar_mean_all"],["Hmean_vegetation", "lidar_mean_VG_all"]],
            [["Hmean_canopy","lidar_mean"],["Hmean_vegetation", "lidar_mean_VG"]],
            [["std_canopy","lidar_std_all"],["std_vegetation", "lidar_std_VG_all"]],
            [["Hmedian_canopy","lidar_median_all"],["Hmedian_vegetation", "lidar_median_VG_all"]],
            [["Hmedian_canopy","lidar_median"],["Hmedian_vegetation", "lidar_median_VG"]]]

_AX_TITLES  = ["Végétation et sol", "Végétation seule"]
_SUP_TITLES = ["Stratégie fenêtre 10x10", "Stratégie diagonale 2x10", "Stratégie fenêtre 10x10", "Stratégie fenêtre 10x10", "Stratégie diagonale 2x10"]
_AX_LABELS  = [["Relevé hauteur moyenne (cm)", "Moyenne lidar (cm)"],
                ["Relevé hauteur moyenne (cm)", "Moyenne lidar (cm)"],
                ["Relevé écart-type hauteur (cm)", "Ecart-type lidar (cm)"],
                ["Relevé hauteur médiane (cm)", "Médiane lidar (cm)"],
                ["Relevé hauteur médiane (cm)", "Médiane lidar (cm)"]]



def LinearRegressionOrthogonal(x,y):

    # Centrer les données
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_cent, y_cent = x - x_mean, y - y_mean

    # Construire la matrice A
    A = np.vstack((x_cent, y_cent)).T

    # Appliquer la SVD
    U, S, Vt = np.linalg.svd(A)

    # Le vecteur propre associé à la plus petite valeur singulière
    a_ortho, b_ortho = Vt[1]  # Deuxième vecteur propre donne la direction

    # Calcul de la pente
    slope = -a_ortho / b_ortho
    intercept = y_mean - slope * x_mean
    return slope, intercept

def vertical_distance(x, y, a, b):
    """Distance verticale (OLS)"""
    return np.abs(y - (a * x + b))

def orthogonal_distance(x, y, a, b):
    """Distance perpendiculaire (ODR)"""
    return np.abs(a * x - y + b) / np.sqrt(a**2 + 1)


def plot_scenario(pdf, scenario, name_fig, ax_labels,suptitle, save=None):
    f, ax = plt.subplots(1, 2, figsize=(10, 5.75), sharey=True)
    for i, sc in enumerate(scenario):
        subdf = pdf[(pdf.isForest == 0)].dropna(subset=[sc[0], sc[1]])
        X = subdf[sc[1]].values.reshape(-1, 1)
        Y = subdf[sc[0]].values
        slope, intercept = LinearRegressionOrthogonal(X.ravel(), Y)
        lr = LinearRegression().fit(X, Y)

        # Scatter plots
        ax[i].grid(True, which='both', linewidth=0.5, alpha=0.5)
        ax[i].scatter(subdf.loc[subdf.isGrassland == 0, sc[1]], subdf.loc[subdf.isGrassland == 0, sc[0]], 
                      marker='o', color='tab:blue', label='Non-grassland', s=40, edgecolors='k', linewidth=0.75, alpha=0.7)
        ax[i].scatter(subdf.loc[subdf.isGrassland == 1, sc[1]], subdf.loc[subdf.isGrassland == 1, sc[0]], 
                      marker='d', color='tab:orange', label='Grassland', s=40, edgecolors='k', linewidth=0.75, alpha=0.7)

        # Regression lines
        x_range = np.linspace(0, 0.8, 100)
        ax[i].plot(x_range, lr.coef_[0] * x_range + lr.intercept_, label='Linear regressor', alpha=0.8, color='tab:red')
        ax[i].plot(x_range, slope * x_range + intercept, label='Orthogonal regressor', alpha=0.8, color='black')
        ax[i].plot(np.linspace(0, 0.7, 100), np.linspace(0, 0.7, 100), 'k--', label='y=x', alpha=0.5)

        # Calcul des métriques
        r2_lin = r2_score(Y, lr.predict(X))
        mse_lin = mean_squared_error(Y, lr.predict(X))
        r2_ortho = r2_score(Y, slope * X.ravel() + intercept)
        mse_ortho = mean_squared_error(Y, slope * X.ravel() + intercept)

        # Texte formaté avec bbox pour meilleure lisibilité
        # textstr = (
        #     f"Linear: y = {lr.coef_[0]:.2f}x + {lr.intercept_:.2f}\n"
        #     f"Ortho: y = {slope:.2f}x + {intercept:.2f}\n"
        #     f"Linear R²: {r2_lin:.3f}, MSE: {mse_lin:.3f}\n"
        #     f"Ortho R²: {r2_ortho:.3f}, MSE: {mse_ortho:.3f}"
        # )
        textstr = (
            f"Linear: y = {lr.coef_[0]:.2f}x + {lr.intercept_:.2f}\n"
            f"Ortho: y = {slope:.2f}x + {intercept:.2f}\n"
            f"Linear MSE: {vertical_distance(X.ravel(), Y, lr.coef_[0], lr.intercept_).mean():.3f}, STD: {vertical_distance(X.ravel(), Y, lr.coef_[0], lr.intercept_).std():.3f}\n"
            f"Ortho MSE: {orthogonal_distance(X.ravel(), Y, slope, intercept).mean():.3f}, STD: {orthogonal_distance(X.ravel(), Y, slope, intercept).std():.3f}"
        )
        ax[i].text(0.35, 0.975, textstr, fontsize=9, verticalalignment='top', horizontalalignment='left',transform=ax[i].transAxes, 
                   bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray', boxstyle='round,pad=0.5'))
 
        # Labels et titres
        if i == 0:
            ax[i].set_ylabel(ax_labels[0])
        ax[i].set_xlabel(ax_labels[1])
        ax[i].set_ylim(0, 0.7)
        ax[i].set_xlim(0, 0.7)
        ax[i].set_title(name_fig[i])


    # Légende commune
    handles, labels = ax[0].get_legend_handles_labels()

    f.legend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.suptitle(suptitle)
    if save != None:
        os.makedirs(os.path.join(save), exist_ok=True)
        strintoenc = f"{suptitle}{name_fig}{ax_labels[0]}{ax_labels[1]}"
        name_fig = sha256(strintoenc.encode()).hexdigest()[:20]
        plt.savefig(os.path.join(save, f'{name_fig}.png'),
                     dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

