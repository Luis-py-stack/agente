import streamlit as st
import os
import io
from groq import Groq
from audio_recorder_streamlit import audio_recorder
from typing import List, Literal
from typing_extensions import Annotated, TypedDict
import operator

# Imports para LangGraph y LangChain
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Set page config
st.set_page_config(
    page_title="Asistente Virtual de Ingeniería Civil",
    page_icon="🏗️",
    layout="wide",
)

# Sidebar for API Keys
with st.sidebar:
    st.header("Configuración de API Keys")
    groq_api_key = st.text_input("GROQ API Key", type="password", help="Ingresa tu clave de Groq.")
    tavily_api_key = st.text_input("Tavily API Key", type="password", help="Ingresa tu clave de Tavily.")

    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key

# Initialize session state
if "transcription" not in st.session_state:
    st.session_state["transcription"] = None
if "final_questions" not in st.session_state:
    st.session_state["final_questions"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

# Header
st.title("🏗️ Asistente Virtual de Ingeniería Civil")
st.markdown("---")
st.write("Sube o graba tu solicitud de proyecto y nuestro equipo de agentes de IA generará un cuestionario técnico exhaustivo.")

def transcribir_con_groq(audio_bytes: bytes) -> str:
    """Envía los bytes de audio directamente a Groq para su transcripción."""
    try:
        client = Groq()
        # Enviar el audio como un archivo tuple (nombre, contenido, content_type)
        # Usamos wav como extension generica, Groq lo manejará.
        file_tuple = ("audio.wav", audio_bytes)
        transcription = client.audio.transcriptions.create(
            file=file_tuple,
            model="whisper-large-v3",
            response_format="json",
            language="es",
            temperature=0.0
        )
        return transcription.text
    except Exception as e:
        return f"Error en transcripción: {str(e)}"

# Seccion de Audio
st.subheader("1. Captura de Audio 🎙️")
col1, col2 = st.columns(2)

audio_bytes = None

with col1:
    st.write("Graba tu audio en vivo:")
    # Aumentamos pause_threshold para evitar que la grabación se corte al hacer una pausa natural al hablar
    recorded_audio = audio_recorder(
        text="Haz clic para grabar",
        recording_color="#e83e8c",
        neutral_color="#6c757d",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=60.0
    )
    if recorded_audio:
        st.success("¡Audio grabado con éxito!")
        audio_bytes = recorded_audio
        st.audio(recorded_audio, format="audio/wav")

with col2:
    st.write("O sube un archivo de audio (.wav, .mp3, .m4a):")
    uploaded_audio = st.file_uploader("Subir archivo de audio", type=["wav", "mp3", "m4a"], label_visibility="collapsed")
    if uploaded_audio is not None:
        st.success("¡Archivo subido con éxito!")
        audio_bytes = uploaded_audio.read()
        st.audio(audio_bytes, format=uploaded_audio.type)

# Procesamiento inicial
if audio_bytes and not st.session_state["is_processing"] and st.session_state["transcription"] is None:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Por favor, configura tu GROQ API Key en la barra lateral antes de continuar.")
    else:
        st.session_state["is_processing"] = True
        with st.status("🎙️ Transcribiendo audio...", expanded=True) as status:
            transcription = transcribir_con_groq(audio_bytes)
            if "Error en transcripción:" in transcription:
                status.update(label="Error en la transcripción", state="error")
                st.error(transcription)
                st.session_state["is_processing"] = False
            else:
                status.update(label="Transcripción completada", state="complete")
                st.session_state["transcription"] = transcription
                st.session_state["is_processing"] = False
                st.rerun()

# --- LANGGRAPH LOGIC ---
MAX_ITERATIONS = 2

class GraphState(TypedDict):
    user_request: str
    draft_questions: str
    critique: str
    search_queries: List[str]
    research_context: Annotated[str, operator.add]
    iteration: int
    is_approved: bool

class EvaluatorOutput(BaseModel):
    is_approved: bool = Field(description="True si las preguntas cubren todos los aspectos técnicos necesarios. False en caso contrario.")
    critique: str = Field(description="Crítica detallada de lo que falta o está mal en las preguntas actuales.")
    search_queries: List[str] = Field(description="Lista de búsquedas (máx 2) para Google si se requiere investigar normativas o estándares de construcción. Vacío si no es necesario.")

def get_graph():
    # Only initialize if keys are present (to avoid crashing on boot)
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        return None

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    tavily_tool = TavilySearchResults(max_results=2)

    def interviewer_node(state: GraphState) -> dict:
        iteracion_actual = state.get('iteration', 0) + 1
        st.toast(f"Agente Entrevistador: Generando Borrador (Iteración {iteracion_actual})", icon="🧠")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un Ingeniero Civil Arquitecto experto. Tu objetivo es generar una lista exhaustiva de preguntas para un cliente que solicita una cotización.

            Contexto de investigación disponible: {research_context}

            Crítica del revisor (si existe): {critique}

            Usa la crítica y la investigación para refinar tus preguntas. Entrega SOLO la lista de preguntas, organizada por categorías (ej. Topografía, Materiales, Legal)."""),
            ("human", "Solicitud original del cliente: {user_request}\n\nPreguntas actuales: {draft_questions}")
        ])

        chain = prompt | llm

        response = chain.invoke({
            "user_request": state["user_request"],
            "draft_questions": state.get("draft_questions", "Ninguna todavía."),
            "critique": state.get("critique", "Ninguna todavía."),
            "research_context": state.get("research_context", "Sin investigación previa.")
        })

        return {
            "draft_questions": response.content,
            "iteration": iteracion_actual
        }

    def evaluator_node(state: GraphState) -> dict:
        st.toast("Agente Evaluador: Reflexionando sobre las preguntas", icon="🔍")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un Revisor Senior de Proyectos de Construcción. Analiza la lista de preguntas generada.
            Asegúrate de que no falte preguntar sobre: normativas locales de uso de suelo, tipo de terreno, acceso de maquinaria pesada, servicios básicos existentes, etc.
            Sé muy crítico. Si las preguntas son muy superficiales, recházalas (is_approved=False).
            Si necesitas saber sobre una norma actual para pedirle el dato exacto al cliente, genera un 'search_queries'."""),
            ("human", "Solicitud del cliente: {user_request}\n\nPreguntas generadas: {draft_questions}")
        ])

        evaluator_llm = llm.with_structured_output(EvaluatorOutput)
        chain = prompt | evaluator_llm

        result: EvaluatorOutput = chain.invoke({
            "user_request": state["user_request"],
            "draft_questions": state["draft_questions"]
        })

        st.toast(f"Evaluador -> Aprobado: {result.is_approved}", icon="✅" if result.is_approved else "❌")

        return {
            "is_approved": result.is_approved,
            "critique": result.critique,
            "search_queries": result.search_queries
        }

    def research_node(state: GraphState) -> dict:
        st.toast("Herramienta Web: Buscando Normativas/Contexto", icon="🌐")
        queries = state.get("search_queries", [])
        new_context = ""

        for q in queries:
            st.toast(f"Buscando: {q}", icon="🔎")
            docs = tavily_tool.invoke({"query": q})
            for doc in docs:
                new_context += f"- Fuente: {doc.get('url', 'N/A')}\nContenido: {doc.get('content', '')}\n\n"

        return {"research_context": new_context}

    def route_after_evaluation(state: GraphState) -> Literal["research_node", "interviewer_node", "END"]:
        if state["is_approved"] or state["iteration"] >= MAX_ITERATIONS:
            return "END"
        elif len(state.get("search_queries", [])) > 0:
            return "research_node"
        else:
            return "interviewer_node"

    builder = StateGraph(GraphState)

    builder.add_node("interviewer_node", interviewer_node)
    builder.add_node("evaluator_node", evaluator_node)
    builder.add_node("research_node", research_node)

    builder.add_edge(START, "interviewer_node")
    builder.add_edge("interviewer_node", "evaluator_node")

    builder.add_conditional_edges(
        "evaluator_node",
        route_after_evaluation,
        {
            "research_node": "research_node",
            "interviewer_node": "interviewer_node",
            "END": END
        }
    )

    builder.add_edge("research_node", "interviewer_node")
    return builder.compile()


if st.session_state["transcription"]:
    st.subheader("2. Transcripción 📝")
    with st.expander("Ver transcripción original", expanded=True):
        st.write(st.session_state["transcription"])

    # Check if we need to run LangGraph
    if st.session_state["final_questions"] is None:
        if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
             st.warning("Configura tus API Keys (GROQ y TAVILY) en la barra lateral para continuar con el análisis.")
        else:
             st.subheader("3. Análisis Técnico (LangGraph) 🤖")
             with st.status("Iniciando equipo de Agentes de Ingeniería Civil...", expanded=True) as status:
                 try:
                     graph = get_graph()
                     initial_state = {
                         "user_request": st.session_state["transcription"],
                         "iteration": 0,
                         "research_context": ""
                     }
                     # Execute graph and get final state
                     final_state = graph.invoke(initial_state, config={"recursion_limit": 15})
                     st.session_state["final_questions"] = final_state.get("draft_questions")
                     status.update(label="Análisis completado", state="complete", expanded=False)
                 except Exception as e:
                     status.update(label="Error en el análisis", state="error", expanded=True)
                     st.error(f"Se produjo un error al procesar con LangGraph: {e}")

    # Display final questions if available
    if st.session_state["final_questions"]:
        st.subheader("4. Cuestionario Técnico Final 📋")
        st.markdown(st.session_state["final_questions"])

        # Reset button
        if st.button("Grabar o subir un nuevo audio", type="primary"):
            st.session_state["transcription"] = None
            st.session_state["final_questions"] = None
            st.session_state["is_processing"] = False
            # Clear file uploader by triggering a rerun (Streamlit component states reset naturally or if keys change)
            st.rerun()
