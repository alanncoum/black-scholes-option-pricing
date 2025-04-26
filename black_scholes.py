import numpy as np
from scipy.stats import norm

# Fonction de calcul d'un call européen
def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Calcule le prix d'un call européen via le modèle de Black-Scholes ainsi que ses Greeks.

    Paramètres :
    - S : Prix actuel du sous-jacent
    - K : Prix d'exercice (strike)
    - T : Temps jusqu'à maturité (en années)
    - r : Taux sans risque (en décimal)
    - sigma : Volatilité du sous-jacent (en décimal)

    Retour :
    Dictionnaire contenant :
    - Prix du Call (€)
    - Delta, Gamma, Vega, Theta, Rho
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    prix_call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    delta_call = norm.cdf(d1)
    gamma_call = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega_call = S * norm.pdf(d1) * np.sqrt(T)
    theta_call = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2)

    return {
        'Prix du Call (€)': prix_call,
        'Delta': delta_call,
        'Gamma': gamma_call,
        'Vega': vega_call,
        'Theta': theta_call,
        'Rho': rho_call
    }

# Fonction de calcul d'un put européen
def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Calcule le prix d'un put européen via le modèle de Black-Scholes ainsi que ses Greeks.

    Paramètres :
    - S : Prix actuel du sous-jacent
    - K : Prix d'exercice (strike)
    - T : Temps jusqu'à maturité (en années)
    - r : Taux sans risque (en décimal)
    - sigma : Volatilité du sous-jacent (en décimal)

    Retour :
    Dictionnaire contenant :
    - Prix du Put (€)
    - Delta, Gamma, Vega, Theta, Rho
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    prix_put = -S * norm.cdf(-d1) + K * np.exp(-r * T) * norm.cdf(-d2)

    delta_put = norm.cdf(d1) - 1
    gamma_put = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega_put = S * norm.pdf(d1) * np.sqrt(T)
    theta_put = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return {
        'Prix du Put (€)': prix_put,
        'Delta': delta_put,
        'Gamma': gamma_put,
        'Vega': vega_put,
        'Theta': theta_put,
        'Rho': rho_put
    }

# Exemple d'utilisation
if __name__ == "__main__":
    S = 100  # Prix du sous-jacent
    K = 110  # Prix d'exercice
    T = 1    # Temps jusqu'à maturité (1 an)
    r = 0.05 # Taux sans risque (5%)
    sigma = 0.2  # Volatilité (20%)

    # Calcul pour Call
    resultat_call = black_scholes_call(S, K, T, r, sigma)
    print("Résultats du modèle Black-Scholes (Call Européen) :")
    print("-" * 50)
    for k, v in resultat_call.items():
        if "€" in k:
            print(f"{k} : {v:.2f}")
        else:
            print(f"{k} : {v:.4f}")
    print("\n")

    # Calcul pour Put
    resultat_put = black_scholes_put(S, K, T, r, sigma)
    print("Résultats du modèle Black-Scholes (Put Européen) :")
    print("-" * 50)
    for k, v in resultat_put.items():
        if "€" in k:
            print(f"{k} : {v:.2f}")
        else:
            print(f"{k} : {v:.4f}")
