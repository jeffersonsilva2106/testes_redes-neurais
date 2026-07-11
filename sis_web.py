import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Diagnóstico de Serviços - IA", page_icon="📊", layout="wide"
)


# Função para treinar a inteligência base da IA (Rede Neural)
@st.cache_resource
def treinar_rede_neural():
    np.random.seed(42)
    X_treino, y_treino = [], []

    for _ in range(300):
        reclamacoes = np.random.randint(0, 50)
        tempo = np.random.uniform(0.5, 5.0)
        satisfacao = np.random.uniform(1.0, 5.0)

        # Regras de negócio memorizadas pela rede
        if reclamacoes > 35 or satisfacao < 2.0:
            nivel = 0  # Muito Ruim
        elif reclamacoes > 20 or satisfacao < 3.5:
            nivel = 1  # Ruim
        elif reclamacoes > 7 or satisfacao < 4.4:
            nivel = 2  # Bom
        else:
            nivel = 3  # Excelente

        X_treino.append([reclamacoes, tempo, satisfacao])
        y_treino.append(nivel)

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)

    mlp = MLPClassifier(
        hidden_layer_sizes=(12, 12),
        max_iter=1200,
        random_state=42,
        learning_rate_init=0.01,
    )
    mlp.fit(X_treino_scaled, y_treino)

    return mlp, scaler


# Inicialização e Treinamento Automático em Background
mlp_model, model_scaler = treinar_rede_neural()

# --- INTERFACE GRÁFICA (STREAMLIT) ---

st.title("🧠 Painel Inteligente de Diagnóstico de Serviços")
st.markdown(
    "Plataforma preditiva utilizando **Redes Neurais Artificiais (MLPClassifier)** "
    "para identificação de gargalos operacionais e direcionamento de treinamentos."
)

# st.hr()

# Seção 1: Upload da Base de Dados
st.subheader("1. Base de Dados da Empresa")
arquivo_carregado = st.file_uploader(
    " Faça o upload do seu arquivo de serviços (.csv)", type=["csv"]
)

# Seção 2: Processamento e Exibição de Resultados
if arquivo_carregado is not None:
    try:
        # Lendo os dados do arquivo carregado pelo usuário
        dados_reais = pd.read_csv(arquivo_carregado)

        # Validação obrigatória do layout das colunas
        colunas_obrigatorias = {
            "servico",
            "reclamacoes",
            "tempo_resolucao_horas",
            "satisfacao_cliente",
        }
        if not colunas_obrigatorias.issubset(dados_reais.columns):
            st.error(
                "❌ Erro na estrutura do arquivo! Certifique-se de que o CSV contém as colunas: "
                "`servico`, `reclamacoes`, `tempo_resolucao_horas`, `satisfacao_cliente`"
            )
        else:
            st.success(" Base de dados importada com sucesso!")

            # Botão para Executar a Análise (como solicitado no Estilo)
            if st.button(
                "📊 Iniciar Análise Preditiva", type="primary", use_container_width=True
            ):

                with st.spinner("A Rede Neural está processando os dados..."):
                    # Preparando dados para predição
                    X_reais = dados_reais[
                        [
                            "reclamacoes",
                            "tempo_resolucao_horas",
                            "satisfacao_cliente",
                        ]
                    ].values
                    X_reais_scaled = model_scaler.transform(X_reais)

                    # Executando a predição inteligente
                    predicoes = mlp_model.predict(X_reais_scaled)

                    # Mapeamentos de Saída Gerencial
                    mapeamento_status = {
                        0: "Serviço Muito Ruim",
                        1: "Serviço Ruim",
                        2: "Serviço Bom",
                        3: "Serviço Excelente",
                    }

                    mapeamento_acao = {
                        0: "⚠️ CRÍTICO: Treinamento imediato e auditoria de processos.",
                        1: "🛑 ATENÇÃO: Reciclagem e monitoramento da equipe.",
                        2: "✅ SOB CONTROLE: Manter o acompanhamento preventivo.",
                        3: "⭐ DESTAQUE: Padrão de excelência. Multiplicar boas práticas.",
                    }

                    # Aplicando os resultados de volta ao DataFrame
                    dados_reais["Nível de Qualidade"] = [
                        mapeamento_status[p] for p in predicoes
                    ]
                    dados_reais["Direcionamento Estratégico"] = [
                        mapeamento_acao[p] for p in predicoes
                    ]

                    # Traduzindo colunas originais para visualização gerencial limpa
                    dados_visualizacao = dados_reais.rename(
                        columns={
                            "servico": "Serviço Analisado",
                            "reclamacoes": "Reclamações",
                            "tempo_resolucao_horas": "Tempo de Resolução (h)",
                            "satisfacao_cliente": "Satisfação do Cliente",
                        }
                    )

                # st.hr()
                st.subheader("2. Classificação de Desempenho Operacional")

                # Exibindo os resultados em uma tabela interativa moderna do Streamlit
                st.dataframe(
                    dados_visualizacao,
                    use_container_width=True,
                    hide_index=True,
                )

                # Resumo Analítico para o Empreendedor
                st.markdown("### 📈 Resumo de Necessidade de Treinamento")
                total_criticos = int(np.sum((predicoes == 0) | (predicoes == 1)))

                if total_criticos > 0:
                    st.warning(
                        f"A IA identificou **{total_criticos} setores** operando abaixo da média recomendada. "
                        "Recomenda-se priorizar o plano de treinamento nestas frentes."
                    )
                else:
                    st.info(
                        "Excelente! Nenhum serviço foi classificado como crítico ou necessitando de treinamento imediato."
                    )

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {str(e)}")
else:
    st.info("Aguardando o upload do arquivo `.csv` para iniciar o diagnóstico.")