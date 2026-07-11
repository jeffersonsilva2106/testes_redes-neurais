import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Configuração da página do Streamlit
st.set_page_config(
    page_title="IA - Análise de Qualidade de Serviços",
    page_icon="📊",
    layout="wide",
)


# Classe de Inteligência Artificial (Rede Neural)
class MotorInteligencia:

    def __init__(self):
        self.scaler = StandardScaler()
        # Rede Neural configurada para avaliar o padrão tridimensional do seu CSV
        self.modelo = MLPClassifier(
            hidden_layer_sizes=(15, 10),
            activation="relu",
            solver="adam",
            max_iter=1500,
            random_state=42,
        )

    def simular_dados_treino(self):
        """Treina a inteligência usando padrões baseados nas métricas reais do seu negócio."""
        np.random.seed(42)
        X_treino = []
        y_treino = []

        for _ in range(800):
            reclamacoes = np.random.randint(1, 50)
            tempo_res = np.random.uniform(0.3, 6.0)
            satisfacao = np.random.uniform(1.0, 5.0)

            X_treino.append([reclamacoes, tempo_res, satisfacao])

            # Lógica multidimensional: prioriza satisfação baixa e alto tempo de resolução
            if satisfacao < 2.0 or (reclamacoes > 30 and tempo_res > 4.0):
                y_treino.append("Serviço Muito Ruim")
            elif satisfacao < 3.2 or tempo_res > 3.0:
                y_treino.append("Serviço Ruim")
            elif satisfacao < 4.5:
                y_treino.append("Serviço Bom")
            else:
                y_treino.append("Serviço Excelente")

        return np.array(X_treino), np.array(y_treino)

    def treinar(self):
        X, y = self.simular_dados_treino()
        X_scaled = self.scaler.fit_transform(X)
        self.modelo.fit(X_scaled, y)

    def predizer(self, dados_novos):
        dados_scaled = self.scaler.transform(dados_novos)
        return self.modelo.predict(dados_scaled)


# --- INTERFACE GRÁFICA (STREAMLIT) ---

st.title("📊 Análise de Serviços com Redes Neurais")
st.markdown("""
Aplica inteligência artificial para diagnosticar a qualidade operacional com base em reclamações, tempo de resposta e satisfação dos clientes.
""")

st.divider()

st.subheader("1. Entrada de Dados")
arquivo_carregado = st.file_uploader(
    "Carregue o arquivo CSV de reclamações da sua empresa", type=["csv"]
)

st.subheader("2. Executar Diagnóstico")
col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    botao_analisar = st.button(
        "🚀 Ativar Função de Análise",
        type="primary",
        use_container_width=True,
        disabled=arquivo_carregado is None,
    )

if arquivo_carregado is None:
    st.info("Aguardando upload do arquivo CSV para habilitar a análise.")

if botao_analisar and arquivo_carregado is not None:
    try:
        with st.spinner("A Rede Neural está processando os dados..."):
            # Lê o arquivo ignorando conflitos de separador (vírgula ou ponto e vírgula)
            df = pd.read_csv(arquivo_carregado, sep=None, engine="python")

            # Normalização de nomes de colunas
            df.columns = df.columns.str.strip().str.lower()

            colunas_obrigatorias = [
                "servico",
                "reclamacoes",
                "tempo_resolucao_horas",
                "satisfacao_cliente",
            ]

            if not all(c in df.columns for c in colunas_obrigatorias):
                st.error(
                    f"❌ Erro de formatação! O arquivo precisa ter exatamente as colunas: {colunas_obrigatorias}"
                )
            else:
                # Converte e limpa os dados numéricos
                df["reclamacoes"] = pd.to_numeric(df["reclamacoes"])
                df["tempo_resolucao_horas"] = pd.to_numeric(
                    df["tempo_resolucao_horas"]
                )
                df["satisfacao_cliente"] = pd.to_numeric(
                    df["satisfacao_cliente"]
                )

                # Agrupa os dados por serviço para consolidar as repetições do seu arquivo
                df_consolidado = (
                    df.groupby("servico")
                    .agg({
                        "reclamacoes": "sum",
                        "tempo_resolucao_horas": "mean",
                        "satisfacao_cliente": "mean",
                    })
                    .reset_index()
                )

                # Treina o cérebro do algoritmo
                ia = MotorInteligencia()
                ia.treinar()

                # Extrai as características e realiza a predição via IA
                caracteristicas = df_consolidado[[
                    "reclamacoes",
                    "tempo_resolucao_horas",
                    "satisfacao_cliente",
                ]].values
                df_consolidado["Classificação (Status)"] = ia.predizer(
                    caracteristicas
                )

                # Identifica o pior serviço usando um índice combinado de risco 
                # (Mais reclamações, maior tempo e menor satisfação)
                df_consolidado["indice_alerta"] = (
                    df_consolidado["reclamacoes"]
                    * df_consolidado["tempo_resolucao_horas"]
                ) / df_consolidado["satisfacao_cliente"]
                pior_registro = df_consolidado.loc[
                    df_consolidado["indice_alerta"].idxmax()
                ]

                st.success("Análise de Inteligência Artificial Concluída!")

                # Alerta Gerencial Dinâmico
                st.subheader("🚨 Diagnóstico de Treinamento Urgente")
                st.error(f"""
                **Atenção Líder:** O gargalo crítico identificado na empresa é o **{pior_registro['servico']}**.
                Este setor acumula um total de **{pior_registro['reclamacoes']} reclamações**, com tempo médio de resolução de **{pior_registro['tempo_resolucao_horas']:.1f} horas** e satisfação de **{pior_registro['satisfacao_cliente']:.1f}/5.0**.
                **Ação recomendada:** Intervenção imediata com foco em reciclagem de processos e treinamento operacional.
                """)

                # Renomeia para exibição limpa no Dashboard do empreendedor
                df_visual = df_consolidado.rename(
                    columns={
                        "servico": "Serviço",
                        "reclamacoes": "Total de Reclamações",
                        "tempo_resolucao_horas": "Tempo Médio Resolução (Hrs)",
                        "satisfacao_cliente": "Média de Satisfação",
                    }
                )

                st.subheader("📋 Relatório Classificatório Geral (Consolidado)")
                st.dataframe(
                    df_visual[[
                        "Serviço",
                        "Total de Reclamações",
                        "Tempo Médio Resolução (Hrs)",
                        "Média de Satisfação",
                        "Classificação (Status)",
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )

    except Exception as e:
        st.error(
            f"Ocorreu um erro inesperado ao processar o seu arquivo: {str(e)}"
        )