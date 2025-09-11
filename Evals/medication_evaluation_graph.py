"""
LangGraph implementation for medication evaluation with multi-agent architecture.

This graph implements a two-layer evaluation system:
1. First layer: Three specialized agents analyze text independently
2. Second layer: Consensus agent combines decisions and provides final classification

Updated for LangChain 0.3.27, LangGraph 0.6.7 and related dependencies
"""

from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# State definition
class EvaluationState(TypedDict):
    """State for the medication evaluation graph"""
    original_text: str
    transcribed_text: str
    medication_classification: Literal["NINGUNA", "LEVE", "GRAVE", None]
    dosage_classification: Literal["NINGUNA", "LEVE", "GRAVE", None]
    consistency_classification: Literal["NINGUNA", "LEVE", "GRAVE", None]
    final_classification: Literal["NINGUNA", "LEVE", "GRAVE", None]
    explanations: List[str]
    consensus_explanation: str


# Initialize LLM (lazy initialization to avoid import-time API key requirement)
def get_llm():
    """Get the LLM instance with API key from environment"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
        api_key=api_key
    )


def medication_agent(state: EvaluationState) -> EvaluationState:
    """MedicationAgent: Evaluates medication name fidelity"""

    llm = get_llm()

    prompt = """Eres un experto en medicina clínica y en terminología farmacológica.
Tu tarea es comparar el texto original con la transcripción y evaluar la fidelidad de los nombres de medicamentos.

Instrucciones:
• Considera que conoces los nombres comerciales y genéricos de los fármacos.
• Marca error solo si el medicamento cambia de identidad (por ejemplo, un fármaco distinto o de otra clase terapéutica).
• No marques como error diferencias de formato, abreviaturas o variantes de escritura si el significado clínico es el mismo.

Clasifica el resultado en una única categoría:
• NINGUNA → el medicamento es el mismo.
• LEVE → variación poco clara de escritura, pero se reconoce como el mismo medicamento.
• GRAVE → el medicamento transcrito corresponde a otro diferente.

TEXTO ORIGINAL:
{original_text}

TEXTO TRANSCRITO:
{transcribed_text}

Responde SOLO con la categoría (NINGUNA, LEVE o GRAVE) sin explicación adicional."""

    response = llm.invoke([
        HumanMessage(content=prompt.format(
            original_text=state["original_text"],
            transcribed_text=state["transcribed_text"]
        ))
    ])

    classification = response.content.strip().upper()

    # Ensure valid classification
    if classification not in ["NINGUNA", "LEVE", "GRAVE"]:
        classification = "NINGUNA"  # Default fallback

    # Only update the medication classification field
    new_state = state.copy()
    new_state["medication_classification"] = classification
    return new_state


def dosage_agent(state: EvaluationState) -> EvaluationState:
    """DosageAgent: Evaluates dosage accuracy"""

    llm = get_llm()

    prompt = """Eres un experto en farmacología clínica y en posología.
Tu tarea es comparar el texto original con la transcripción y comprobar si la dosis está bien transcrita.

Instrucciones:
• Marca error solo si cambia la cantidad, la unidad o la frecuencia de la dosis.
• No marques como error diferencias de estilo o de formato si el significado es el mismo (ejemplo: "200 mg/día" y "200 miligramos al día").

Clasifica el resultado en una única categoría:
• NINGUNA → la dosis tiene el mismo significado.
• LEVE → hay una diferencia menor que puede generar ligera confusión, pero no cambia la dosis.
• GRAVE → la dosis, la unidad o la frecuencia han cambiado de forma significativa.

TEXTO ORIGINAL:
{original_text}

TEXTO TRANSCRITO:
{transcribed_text}

Responde SOLO con la categoría (NINGUNA, LEVE o GRAVE) sin explicación adicional."""

    response = llm.invoke([
        HumanMessage(content=prompt.format(
            original_text=state["original_text"],
            transcribed_text=state["transcribed_text"]
        ))
    ])

    classification = response.content.strip().upper()

    # Ensure valid classification
    if classification not in ["NINGUNA", "LEVE", "GRAVE"]:
        classification = "NINGUNA"  # Default fallback

    # Only update the dosage classification field
    new_state = state.copy()
    new_state["dosage_classification"] = classification
    return new_state


def consistency_agent(state: EvaluationState) -> EvaluationState:
    """ConsistencyAgent: Evaluates overall coherence"""

    llm = get_llm()

    prompt = """Eres un experto en redacción médica y en coherencia clínica.
Tu tarea es comparar el texto original con la transcripción y verificar si se mantiene la coherencia de la información (síntomas, diagnósticos, alergias, instrucciones).

Instrucciones:
• Marca error solo si cambia el sentido clínico.
• Ignora diferencias de estilo, pequeñas omisiones o reformulaciones que no alteran el significado.

Clasifica el resultado en una única categoría:
• NINGUNA → no hay cambios de significado clínico.
• LEVE → se omite o cambia un detalle secundario, sin afectar al sentido clínico principal.
• GRAVE → cambia el significado de forma importante (ejemplo: de "no tiene alergias" a "tiene alergias").

TEXTO ORIGINAL:
{original_text}

TEXTO TRANSCRITO:
{transcribed_text}

Responde SOLO con la categoría (NINGUNA, LEVE o GRAVE) sin explicación adicional."""

    response = llm.invoke([
        HumanMessage(content=prompt.format(
            original_text=state["original_text"],
            transcribed_text=state["transcribed_text"]
        ))
    ])

    classification = response.content.strip().upper()

    # Ensure valid classification
    if classification not in ["NINGUNA", "LEVE", "GRAVE"]:
        classification = "NINGUNA"  # Default fallback

    # Only update the consistency classification field
    new_state = state.copy()
    new_state["consistency_classification"] = classification
    return new_state


def consensus_agent(state: EvaluationState) -> EvaluationState:
    """ConsensusAgent: Combines classifications with algorithmic decision making"""

    # Extract classifications
    med_class = state["medication_classification"]
    dosage_class = state["dosage_classification"]
    consistency_class = state["consistency_classification"]

    classifications = [med_class, dosage_class, consistency_class]

    # Apply consensus rules
    if "GRAVE" in classifications:
        final_classification = "GRAVE"
    elif classifications.count("LEVE") >= 2:
        final_classification = "LEVE"
    elif classifications.count("NINGUNA") >= 2:
        final_classification = "NINGUNA"
    else:
        # Default to most severe non-GRAVE classification if tie
        if "LEVE" in classifications:
            final_classification = "LEVE"
        else:
            final_classification = "NINGUNA"

    # Create explanation
    explanation = f"""Clasificación final: {final_classification}

Análisis de agentes:
• Medicamentos: {med_class}
• Dosis: {dosage_class}
• Coherencia: {consistency_class}

Reglas aplicadas:
• Si cualquiera es GRAVE → final = GRAVE
• Si la mayoría es LEVE → final = LEVE
• Si la mayoría son NINGUNA → final = NINGUNA"""

    return {
        **state,
        "final_classification": final_classification,
        "consensus_explanation": explanation
    }


# Build the graph
def create_medication_evaluation_graph():
    """Create the medication evaluation LangGraph"""

    # Initialize StateGraph
    workflow = StateGraph(EvaluationState)

    # Add nodes (agents)
    workflow.add_node("medication_agent", medication_agent)
    workflow.add_node("dosage_agent", dosage_agent)
    workflow.add_node("consistency_agent", consistency_agent)
    workflow.add_node("consensus_agent", consensus_agent)

    # Define workflow edges
    # Sequential execution to avoid concurrent updates
    workflow.add_edge(START, "medication_agent")
    workflow.add_edge("medication_agent", "dosage_agent")
    workflow.add_edge("dosage_agent", "consistency_agent")
    workflow.add_edge("consistency_agent", "consensus_agent")

    # End the workflow
    workflow.add_edge("consensus_agent", END)

    # Compile the graph
    graph = workflow.compile()

    return graph


# Create the graph instance
medication_evaluation_graph = create_medication_evaluation_graph()


if __name__ == "__main__":
    # Example usage
    test_state = {
        "original_text": "El paciente toma 200 mg de ibuprofeno cada 8 horas para el dolor articular. No tiene alergias conocidas.",
        "transcribed_text": "El paciente toma 200 miligramos de iboprofen cada 8 horas para el dolor articular. Tiene alergias conocidas.",
        "medication_classification": None,
        "dosage_classification": None,
        "consistency_classification": None,
        "final_classification": None,
        "explanations": [],
        "consensus_explanation": ""
    }

    try:
        result = medication_evaluation_graph.invoke(test_state)

        print("=== RESULTADO DE EVALUACIÓN ===")
        print(f"Texto original: {result['original_text']}")
        print(f"Texto transcrito: {result['transcribed_text']}")
        print(f"\nClasificaciones:")
        print(f"• Medicamentos: {result['medication_classification']}")
        print(f"• Dosis: {result['dosage_classification']}")
        print(f"• Coherencia: {result['consistency_classification']}")
        print(f"• Final: {result['final_classification']}")
        print(f"\nExplicación del consenso:\n{result['consensus_explanation']}")

    except ValueError as e:
        print(f"❌ Error: {e}")
        print("💡 Necesitas configurar OPENAI_API_KEY para ejecutar las pruebas")
        print("   Ejemplo: export OPENAI_API_KEY='tu_clave_aqui'")
