import numpy as np

def bissec(a: float, b: float, tolerancia: float = 1e-12) -> float:
    """
    Método da Bissecção para encontrar uma raiz de f(x) no intervalo [a, b].
    Utilizando o teorema de Bozano, onde, uma função 𝑓(𝑥), contínua em [𝑎, 𝑏] 
    tal que 𝒇 (𝒂) . 𝒇 (𝒃) < 𝟎.
    Então, 𝑓(𝑥) possui pelo menos uma raiz no intervalo [𝑎, 𝑏].
    Note que o teorema não trata da unicidade de raízes, mas sim da
    existência de raízes.

    
    Parâmetros:
    a (float): Limite inferior do intervalo.
    b (float): Limite superior do intervalo.
    tolerancia (float, opcional): Critério de parada. O padrão é 1e-12.

    Retorna:
    float: Aproximação da raiz da função f(x).

    Levanta:
    ValueError: Se f(a) * f(b) > 0 (não há garantia de raiz no intervalo).
    ValueError: Se a tolerância for menor ou igual a zero.
    """
    # Verificando se as entradas são escalares
    if not np.isscalar(a) or not np.isscalar(b) or not np.isscalar(tolerancia):
        raise ValueError("Os valores de a, b e tolerância devem ser escalares.")
    
    # Garantindo que a tolerância seja positiva
    if tolerancia <= 0:
        raise ValueError("A tolerância deve ser maior que zero.")
    
    # Definição da função f(x) a ser analisada
    def f(x):
        return x**2 - 7
    
    # Verificando a condição do Teorema de Bolzano
    if f(a) * f(b) > 0:
        raise ValueError("Não há garantia de raiz nesse intervalo.")
    
    # Cálculo do número de iterações necessárias
    n = np.ceil((np.log(abs(b - a)) - np.log(tolerancia)) / np.log(2))
    n = int(n)
    
    for _ in range(n):
        m = (a + b) / 2  # Ponto médio

        if abs(f(m)) < tolerancia:
            return m
        
        if f(a) * f(m) < 0:
            b = m
        else:
            a = m
    
    return m
