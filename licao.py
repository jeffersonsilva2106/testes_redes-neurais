import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier

st.set_page_config(
    page_title="Hub de Inteligência Artificial", page_icon="🤖", layout="wide"
)
st.title("🤖 Hub de Modelos de Inteligência Artificial")
st.markdown(
    "Navegue pelo menu lateral para testar cada um dos 10 modelos preditivos e classificatórios."
)

st.sidebar.title("Menu de Modelos")
opcao = st.sidebar.radio(
    "Escolha o Desafio:",
    [
        "1. IA das Notas Escolares",
        "2. Detector de Sono Gamer",
        "3. IA do Sorvete",
        "4. Detector de Aprovação Ninja",
        "5. IA do Pet Feliz",
        "6. Detector de Filme Bom",
        "7. IA da Pizza",
        "8. Detector de Música Viral",
        "9. IA da Energia do Café",
        "10. Rede Neural dos Super-Heróis",
    ],
)

st.divider()

# -------------------------------------------------------------
# 1. IA DAS NOTAS ESCOLARES
# -------------------------------------------------------------
if opcao == "1. IA das Notas Escolares":
    st.header("📝 1. IA das Notas Escolares")
    st.subheader("Objetivo: Prever nota baseada nas horas de estudo.")

    estudos = pd.DataFrame(
        {"notas": [1, 2, 4, 6, 8, 10], "horas": [2, 4, 5, 7, 9, 10]}
    )

    X = estudos[["horas"]].values
    y = estudos["notas"].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    st.write("### Dados de Treinamento", estudos)

    horas_input = st.number_input(
        "Digite a quantidade de horas estudadas:",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
    )
    if st.button("🚀 Prever Nota", key="btn_1"):
        predicao = modelo.predict([[horas_input]])[0]
        # Limita a nota máxima teórica a 10 e mínima a 0 para consistência do negócio
        nota_final = max(0.0, min(10.0, predicao))

        if nota_final >= 7.0:
            st.success(f"Nota Estimada: **{nota_final:.1f}** (Serviço Excelente)")
        elif nota_final >= 5.0:
            st.info(f"Nota Estimada: **{nota_final:.1f}** (Serviço Bom)")
        elif nota_final >= 3.0:
            st.warning(f"Nota Estimada: **{nota_final:.1f}** (Serviço Ruim)")
        else:
            st.error(
                f"Nota Estimada: **{nota_final:.1f}** (Serviço Muito Ruim)"
            )

# -------------------------------------------------------------
# 2. DETECTOR DE SONO GAMER
# -------------------------------------------------------------
elif opcao == "2. Detector de Sono Gamer":
    st.header("🎮 2. Detector de Sono Gamer")
    st.subheader(
        "Objetivo: Prever nível de cansaço baseado em horas jogando."
    )

    gamer = pd.DataFrame(
        {"horas_jogo": [1, 2, 4, 6, 8, 10], "cansaco": [1, 2, 3, 5, 8, 10]}
    )

    X = gamer[["horas_jogo"]].values
    y = gamer["cansaco"].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    st.write("### Dados de Treinamento", gamer)

    horas_jogo = st.number_input(
        "Horas jogando contínuas:", min_value=0.0, max_value=24.0, value=5.0
    )
    if st.button("🚀 Analisar Cansaço", key="btn_2"):
        predicao = modelo.predict([[horas_jogo]])[0]
        nivel = max(0.0, min(10.0, predicao))

        if nivel >= 8.0:
            st.error(
                f"Nível de Cansaço: **{nivel:.1f}/10** -> Estado: Serviço Muito Ruim (Exausto)"
            )
        elif nivel >= 5.0:
            st.warning(
                f"Nível de Cansaço: **{nivel:.1f}/10** -> Estado: Serviço Ruim (Muito Cansado)"
            )
        elif nivel >= 2.5:
            st.info(
                f"Nível de Cansaço: **{nivel:.1f}/10** -> Estado: Serviço Bom (Alerta)"
            )
        else:
            st.success(
                f"Nível de Cansaço: **{nivel:.1f}/10** -> Estado: Serviço Excelente (Pronto para outra)"
            )

# -------------------------------------------------------------
# 3. IA DO SORVETE
# -------------------------------------------------------------
elif opcao == "3. IA do Sorvete":
    st.header("🍦 3. IA do Sorvete")
    st.subheader(
        "Objetivo: Prever quantidade de sorvetes vendidos pela temperatura."
    )

    sorvete = pd.DataFrame(
        {
            "temperatura": [18, 20, 24, 27, 30, 35],
            "vendas": [20, 25, 40, 55, 70, 100],
        }
    )

    X = sorvete[["temperatura"]].values
    y = sorvete["vendas"].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    st.write("### Dados de Treinamento", sorvete)

    temp = st.number_input(
        "Temperatura esperada (°C):", min_value=-10.0, max_value=50.0, value=25.0
    )
    if st.button("🚀 Prever Vendas", key="btn_3"):
        vendas = max(0, int(round(modelo.predict([[temp]])[0])))

        if vendas < 25:
            st.error(
                f"Vendas Estimadas: **{vendas} unidades** (Serviço Muito Ruim)"
            )
        elif vendas < 50:
            st.warning(
                f"Vendas Estimadas: **{vendas} unidades** (Serviço Ruim)"
            )
        elif vendas < 75:
            st.info(f"Vendas Estimadas: **{vendas} unidades** (Serviço Bom)")
        else:
            st.success(
                f"Vendas Estimadas: **{vendas} unidades** (Serviço Excelente)"
            )

# -------------------------------------------------------------
# 4. DETECTOR DE APROVAÇÃO NINJA
# -------------------------------------------------------------
elif opcao == "4. Detector de Aprovação Ninja":
    st.header("🥷 4. Detector de Aprovação Ninja")
    st.subheader("Objetivo: Classificar o resultado do aluno pelas faltas.")

    alunos = pd.DataFrame(
        {"faltas": [0, 1, 2, 5, 7, 10], "resultado": [1, 1, 1, 0, 0, 0]}
    )

    X = alunos[["faltas"]].values
    y = alunos["resultado"].values

    modelo = LogisticRegression()
    modelo.fit(X, y)

    st.write("### Dados de Treinamento (1=Aprovado, 0=Reprovado)", alunos)

    faltas = st.number_input(
        "Quantidade de faltas do aluno:", min_value=0, max_value=50, value=3
    )
    if st.button("🚀 Verificar Situação", key="btn_4"):
        predicao = modelo.predict([[faltas]])[0]

        if predicao == 1:
            st.success("Resultado da Classificação: **APROVADO** (Serviço Excelente)")
        else:
            st.error(
                "Resultado da Classificação: **REPROVADO** (Serviço Muito Ruim)"
            )

# -------------------------------------------------------------
# 5. IA DO PET FELIZ
# -------------------------------------------------------------
elif opcao == "5. IA do Pet Feliz":
    st.header("🐶 5. IA do Pet Feliz")
    st.subheader("Objetivo: Prever felicidade do cachorro.")

    pets = pd.DataFrame(
        {"passeios": [1, 2, 3, 4, 5], "felicidade": [2, 4, 5, 8, 10]}
    )

    X = pets[["passeios"]].values
    y = pets["felicidade"].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    st.write("### Dados de Treinamento", pets)

    passeios = st.number_input(
        "Número de passeios realizados na semana:",
        min_value=0,
        max_value=30,
        value=3,
    )
    if st.button("🚀 Medir Felicidade", key="btn_5"):
        felicidade = max(0.0, min(10.0, modelo.predict([[passeios]])[0]))

        if felicidade >= 8.5:
            st.success(
                f"Índice de Felicidade: **{felicidade:.1f}/10** (Serviço Excelente)"
            )
        elif felicidade >= 5.5:
            st.info(
                f"Índice de Felicidade: **{felicidade:.1f}/10** (Serviço Bom)"
            )
        elif felicidade >= 3.0:
            st.warning(
                f"Índice de Felicidade: **{felicidade:.1f}/10** (Serviço Ruim)"
            )
        else:
            st.error(
                f"Índice de Felicidade: **{felicidade:.1f}/10** (Serviço Muito Ruim)"
            )

# -------------------------------------------------------------
# 6. DETECTOR DE FILME BOM
# -------------------------------------------------------------
elif opcao == "6. Detector de Filme Bom":
    st.header("🎬 6. Detector de Filme Bom")
    st.subheader("Objetivo: Prever nota do filme usando duração.")

    filmes = pd.DataFrame(
        {
            "duracao": [80, 90, 100, 110, 120],
            "nota": [4, 5, 7, 8, 9],
        }
    )

    X = filmes[["duracao"]].values
    y = filmes["nota"].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    st.write("### Dados de Treinamento", filmes)

    duracao = st.number_input(
        "Tempo de duração do filme (minutos):",
        min_value=1,
        max_value=300,
        value=105,
    )
    if st.button("🚀 Calcular Nota Projetada", key="btn_6"):
        nota = max(0.0, min(10.0, modelo.predict([[duracao]])[0]))

        if nota >= 8.0:
            st.success(f"Nota Prevista: **{nota:.1f}** (Serviço Excelente)")
        elif nota >= 6.5:
            st.info(f"Nota Prevista: **{nota:.1f}** (Serviço Bom)")
        elif nota >= 4.5:
            st.warning(f"Nota Prevista: **{nota:.1f}** (Serviço Ruim)")
        else:
            st.error(f"Nota Prevista: **{nota:.1f}** (Serviço Muito Ruim)")

# -------------------------------------------------------------
# 7. IA DA PIZZA
# -------------------------------------------------------------
elif opcao == "7. IA da Pizza":
    st.header("🍕 7. IA da Pizza")
    st.subheader("Objetivo: Prever preço da pizza pelo tamanho.")

    pizza = pd.DataFrame(
        {
            "tamanho": [20, 25, 30, 35, 40],
            "preco": [20, 30, 40, 50, 60],
        }
    )

    X = pizza[["tamanho"]].values
    y = pizza["preco"].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    st.write("### Dados de Treinamento", pizza)

    tamanho = st.number_input(
        "Diâmetro da Pizza (cm):", min_value=10, max_value=80, value=35
    )
    if st.button("🚀 Calcular Preço Justo", key="btn_7"):
        preco = max(0.0, modelo.predict([[tamanho]])[0])

        if preco < 25:
            st.error(f"Preço Estimado: **R$ {preco:.2f}** (Serviço Muito Ruim)")
        elif preco < 40:
            st.warning(f"Preço Estimado: **R$ {preco:.2f}** (Serviço Ruim)")
        elif preco < 55:
            st.info(f"Preço Estimado: **R$ {preco:.2f}** (Serviço Bom)")
        else:
            st.success(
                f"Preço Estimado: **R$ {preco:.2f}** (Serviço Excelente)"
            )

# -------------------------------------------------------------
# 8. DETECTOR DE MÚSICA VIRAL
# -------------------------------------------------------------
elif opcao == "8. Detector de Música Viral":
    st.header("🎵 8. Detector de Música Viral")
    st.subheader("Objetivo: Prever chance (indicador) da música viralizar.")

    musica = pd.DataFrame(
        {"bpm": [80, 90, 100, 120, 140], "viral": [1, 2, 4, 7, 10]}
    )

    X = musica[["bpm"]].values
    y = musica["viral"].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    st.write("### Dados de Treinamento", musica)

    bpm = st.number_input(
        "Batidas Por Minuto (BPM):", min_value=40, max_value=250, value=115
    )
    if st.button("🚀 Medir Força Viral", key="btn_8"):
        viralidade = max(0.0, min(10.0, modelo.predict([[bpm]])[0]))

        if viralidade >= 7.5:
            st.success(
                f"Pontuação de Viralização: **{viralidade:.1f}/10** (Serviço Excelente)"
            )
        elif viralidade >= 5.0:
            st.info(
                f"Pontuação de Viralização: **{viralidade:.1f}/10** (Serviço Bom)"
            )
        elif viralidade >= 2.5:
            st.warning(
                f"Pontuação de Viralização: **{viralidade:.1f}/10** (Serviço Ruim)"
            )
        else:
            st.error(
                f"Pontuação de Viralização: **{viralidade:.1f}/10** (Serviço Muito Ruim)"
            )

# -------------------------------------------------------------
# 9. IA DA ENERGIA DO CAFÉ
# -------------------------------------------------------------
elif opcao == "9. IA da Energia do Café":
    st.header("☕ 9. IA da Energia do Café")
    st.subheader("Objetivo: Prever energia baseada em cafés tomados.")

    cafe = pd.DataFrame(
        {"xicaras": [1, 2, 3, 4, 5], "energia": [2, 4, 6, 8, 10]}
    )

    X = cafe[["xicaras"]].values
    y = cafe["energia"].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    st.write("### Dados de Treinamento", cafe)

    xicaras = st.number_input(
        "Número de xícaras de café tomadas:", min_value=0, max_value=15, value=2
    )
    if st.button("🚀 Checar Nível de Energia", key="btn_9"):
        energia = max(0.0, min(10.0, modelo.predict([[xicaras]])[0]))

        if energia >= 8.0:
            st.success(
                f"Nível de Disposição: **{energia:.1f}/10** (Serviço Excelente)"
            )
        elif energia >= 5.0:
            st.info(
                f"Nível de Disposição: **{energia:.1f}/10** (Serviço Bom)"
            )
        elif energia >= 3.0:
            st.warning(
                f"Nível de Disposição: **{energia:.1f}/10** (Serviço Ruim)"
            )
        else:
            st.error(
                f"Nível de Disposição: **{energia:.1f}/10** (Serviço Muito Ruim)"
            )

# -------------------------------------------------------------
# 10. REDE NEURAL DOS SUPER-HERÓIS
# -------------------------------------------------------------
elif opcao == "10. Rede Neural dos Super-Heróis":
    st.header("🦸‍♂️ 10. Rede Neural dos Super-Heróis")
    st.subheader("Objetivo: Classificar se o herói é forte ou fraco.")

    herois = pd.DataFrame(
        {"forca": [1, 2, 3, 7, 8, 10], "heroi": [0, 0, 0, 1, 1, 1]}
    )

    X = herois[["forca"]].values
    y = herois["heroi"].values

    # Utilização do algoritmo de Redes Neurais MLPClassifier para a classificação
    modelo = MLPClassifier(
        hidden_layer_sizes=(5, 5), max_iter=2000, random_state=42
    )
    modelo.fit(X, y)

    st.write("### Dados de Treinamento (1=Forte, 0=Fraco)", herois)

    forca_input = st.number_input(
        "Insira o nível de força do herói (1 a 10):",
        min_value=1,
        max_value=10,
        value=5,
    )
    if st.button("🚀 Classificar Herói", key="btn_10"):
        classe_predita = modelo.predict([[forca_input]])[0]

        if classe_predita == 1:
            st.success(
                f"Classe Classificada pela Rede Neural: **FORTE** (Serviço Excelente)"
            )
        else:
            st.error(
                f"Classe Classificada pela Rede Neural: **FRACO** (Serviço Muito Ruim)"
            )
