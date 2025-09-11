"""
Ejemplo simple de uso del sistema de evaluación de transcripciones médicas
"""

from medication_evaluation_graph import medication_evaluation_graph


def main():
    """Ejemplo de uso básico"""

    # Caso de prueba: Medicamento con error grave
    print("🔍 Evaluando transcripción médica...")

    initial_state = {
        "original_text": "El paciente toma Celebrex 200 mg una vez al día para el dolor articular. No presenta alergias conocidas.",
        "transcribed_text": "El paciente toma Cerebyx 200 mg una vez al día para el dolor articular. Presenta alergias conocidas.",
        "medication_classification": None,
        "dosage_classification": None,
        "consistency_classification": None,
        "final_classification": None,
        "explanations": [],
        "consensus_explanation": ""
    }

    # Ejecutar evaluación
    result = medication_evaluation_graph.invoke(initial_state)

    # Mostrar resultados
    print("\n" + "="*60)
    print("📋 RESULTADO DE EVALUACIÓN")
    print("="*60)

    print(f"\n📝 TEXTO ORIGINAL:")
    print(f"   {result['original_text']}")

    print(f"\n🎙️ TEXTO TRANSCRITO:")
    print(f"   {result['transcribed_text']}")

    print(f"\n🔍 CLASIFICACIONES INDIVIDUALES:")
    print(f"   • Medicamentos: {result['medication_classification']}")
    print(f"   • Dosis:        {result['dosage_classification']}")
    print(f"   • Coherencia:   {result['consistency_classification']}")

    print(f"\n🏆 CLASIFICACIÓN FINAL:")
    print(f"   {result['final_classification']}")

    print(f"\n📊 ANÁLISIS DE CONSENSO:")
    print(f"{result['consensus_explanation']}")

    # Interpretación del resultado
    print(f"\n💡 INTERPRETACIÓN:")
    if result['final_classification'] == "GRAVE":
        print("   ⚠️  ERROR CRÍTICO: Requiere revisión inmediata por profesional médico")
    elif result['final_classification'] == "LEVE":
        print("   ⚡ ERROR MENOR: Revisar pero no crítico para la atención del paciente")
    else:
        print("   ✅ SIN ERRORES: Transcripción fiel al contenido original")


if __name__ == "__main__":
    main()
