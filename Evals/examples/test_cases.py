"""
Ejemplos de casos de prueba para el sistema de evaluación de transcripciones médicas
"""

from medication_evaluation_graph import medication_evaluation_graph


def test_medication_errors():
    """Casos de prueba para errores en medicamentos"""

    test_cases = [
        {
            "name": "Medicamento correcto",
            "original": "El paciente toma ibuprofeno 600 mg cada 8 horas",
            "transcribed": "El paciente toma ibuprofeno 600 mg cada 8 horas",
            "expected_medication": "NINGUNA",
            "expected_final": "NINGUNA"
        },
        {
            "name": "Abreviatura aceptable",
            "original": "Administrar amoxicilina 500 mg",
            "transcribed": "Administrar amox 500 mg",
            "expected_medication": "NINGUNA",
            "expected_final": "NINGUNA"
        },
        {
            "name": "Variación ortográfica leve",
            "original": "Tratamiento con ibuprofeno",
            "transcribed": "Tratamiento con iboprofeno",
            "expected_medication": "LEVE",
            "expected_final": "LEVE"
        },
        {
            "name": "Medicamento diferente (GRAVE)",
            "original": "Recetado Celebrex para dolor",
            "transcribed": "Recetado Cerebyx para dolor",
            "expected_medication": "GRAVE",
            "expected_final": "GRAVE"
        }
    ]

    return test_cases


def test_dosage_errors():
    """Casos de prueba para errores en dosis"""

    test_cases = [
        {
            "name": "Dosis correcta",
            "original": "Tomar 200 mg cada 12 horas",
            "transcribed": "Tomar 200 miligramos cada 12 horas",
            "expected_dosage": "NINGUNA",
            "expected_final": "NINGUNA"
        },
        {
            "name": "Formato diferente pero mismo significado",
            "original": "200 mg/día",
            "transcribed": "200 miligramos al día",
            "expected_dosage": "NINGUNA",
            "expected_final": "NINGUNA"
        },
        {
            "name": "Cambio en frecuencia (GRAVE)",
            "original": "Una vez al día",
            "transcribed": "Tres veces al día",
            "expected_dosage": "GRAVE",
            "expected_final": "GRAVE"
        },
        {
            "name": "Cambio en cantidad (GRAVE)",
            "original": "20 mg de warfarina",
            "transcribed": "200 mg de warfarina",
            "expected_dosage": "GRAVE",
            "expected_final": "GRAVE"
        }
    ]

    return test_cases


def test_consistency_errors():
    """Casos de prueba para errores de coherencia"""

    test_cases = [
        {
            "name": "Información coherente",
            "original": "Paciente estable, sin complicaciones",
            "transcribed": "Paciente estable, sin complicaciones",
            "expected_consistency": "NINGUNA",
            "expected_final": "NINGUNA"
        },
        {
            "name": "Cambio en alergias (GRAVE)",
            "original": "No tiene alergias conocidas",
            "transcribed": "Tiene alergias conocidas",
            "expected_consistency": "GRAVE",
            "expected_final": "GRAVE"
        },
        {
            "name": "Omisión de detalle secundario",
            "original": "Paciente con hipertensión y diabetes",
            "transcribed": "Paciente con hipertensión",
            "expected_consistency": "LEVE",
            "expected_final": "LEVE"
        }
    ]

    return test_cases


def test_consensus_scenarios():
    """Casos de prueba para reglas de consenso"""

    test_cases = [
        {
            "name": "Un GRAVE determina resultado GRAVE",
            "original": "Compleja evaluación médica",
            "transcribed": "Compleja evaluación médica",
            "expected_final": "GRAVE"
        },
        {
            "name": "Mayoría LEVE determina LEVE",
            "original": "Evaluación médica estándar",
            "transcribed": "Evaluación médica estándar",
            "expected_final": "LEVE"
        },
        {
            "name": "Mayoría NINGUNA determina NINGUNA",
            "original": "Evaluación médica perfecta",
            "transcribed": "Evaluación médica perfecta",
            "expected_final": "NINGUNA"
        }
    ]

    return test_cases


def run_test_case(test_case):
    """Ejecuta un caso de prueba individual"""

    initial_state = {
        "original_text": test_case["original"],
        "transcribed_text": test_case["transcribed"],
        "medication_classification": None,
        "dosage_classification": None,
        "consistency_classification": None,
        "final_classification": None,
        "explanations": [],
        "consensus_explanation": ""
    }

    result = medication_evaluation_graph.invoke(initial_state)

    print(f"\n=== {test_case['name']} ===")
    print(f"Original: {test_case['original']}")
    print(f"Transcrito: {test_case['transcribed']}")
    print(f"Resultado esperado: {test_case.get('expected_final', 'N/A')}")
    print(f"Resultado obtenido: {result['final_classification']}")
    print("Clasificaciones individuales:")
    print(f"  • Medicamentos: {result['medication_classification']}")
    print(f"  • Dosis: {result['dosage_classification']}")
    print(f"  • Coherencia: {result['consistency_classification']}")

    return result


def run_all_tests():
    """Ejecuta todos los casos de prueba"""

    print("🚀 Iniciando pruebas del sistema de evaluación médica")

    # Ejecutar pruebas de medicamentos
    print("\n📊 PRUEBAS DE MEDICAMENTOS")
    for test_case in test_medication_errors():
        run_test_case(test_case)

    # Ejecutar pruebas de dosis
    print("\n💊 PRUEBAS DE DOSIS")
    for test_case in test_dosage_errors():
        run_test_case(test_case)

    # Ejecutar pruebas de coherencia
    print("\n📝 PRUEBAS DE COHERENCIA")
    for test_case in test_consistency_errors():
        run_test_case(test_case)

    print("\n✅ Pruebas completadas")


if __name__ == "__main__":
    run_all_tests()
