#!/usr/bin/env python3
"""
Evaluador de Transcripciones Médicas - CLI Tool

Herramienta de línea de comandos para evaluar la calidad de transcripciones médicas
comparando texto original con texto transcrito.

Uso:
    python med_eval.py <archivo_original.json> <archivo_transcrito.json> [opciones]

Ejemplos:
    python med_eval.py original.json transcribed.json
    python med_eval.py data/original.json data/transcribed.json --output results.json
    python med_eval.py file1.json file2.json --quiet
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import os

# Añadir el directorio actual al path para importar módulos locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medication_evaluation_graph import medication_evaluation_graph


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Carga un archivo JSON y retorna su contenido."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Archivo no encontrado: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Archivo JSON inválido {file_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al leer {file_path}: {e}")
        sys.exit(1)


def extract_text_from_json(data: Dict[str, Any], file_path: str) -> str:
    """Extrae el texto del archivo JSON. Busca en campos comunes."""
    # Campos comunes donde puede estar el texto
    text_fields = ['text', 'content', 'transcript', 'transcription', 'message']

    # Buscar en campos de primer nivel
    for field in text_fields:
        if field in data and isinstance(data[field], str):
            return data[field]

    # Buscar en campos anidados comunes
    if 'data' in data and isinstance(data['data'], dict):
        for field in text_fields:
            if field in data['data'] and isinstance(data['data'][field], str):
                return data['data'][field]

    # Si no encuentra campos conocidos, intentar extraer todo el texto
    text_content = ""
    for key, value in data.items():
        if isinstance(value, str) and len(value) > 10:  # Texto significativo
            text_content += value + " "

    if text_content.strip():
        return text_content.strip()

    print(f"⚠️  Advertencia: No se pudo extraer texto claro de {file_path}")
    print(f"   Campos encontrados: {list(data.keys())}")
    return str(data)  # Fallback: convertir todo a string


def save_results_json(results: Dict[str, Any], output_path: str) -> None:
    """Guarda los resultados en un archivo JSON."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Resultados guardados en: {output_path}")
    except Exception as e:
        print(f"❌ Error al guardar resultados: {e}")


def format_terminal_output(result: Dict[str, Any], quiet: bool = False) -> None:
    """Formatea y muestra los resultados en terminal."""
    if not quiet:
        print("\n" + "="*70)
        print("🏥 EVALUACIÓN DE TRANSCRIPCIÓN MÉDICA")
        print("="*70)

        print(f"\n📝 TEXTO ORIGINAL:")
        print(f"   {result['original_text'][:100]}{'...' if len(result['original_text']) > 100 else ''}")

        print(f"\n🎙️ TEXTO TRANSCRITO:")
        print(f"   {result['transcribed_text'][:100]}{'...' if len(result['transcribed_text']) > 100 else ''}")

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

        print("\n" + "="*70)
    else:
        # Modo silencioso: solo mostrar clasificación final
        print(f"{result['final_classification']}")


def create_evaluation_state(original_text: str, transcribed_text: str) -> Dict[str, Any]:
    """Crea el estado inicial para la evaluación."""
    return {
        "original_text": original_text,
        "transcribed_text": transcribed_text,
        "medication_classification": None,
        "dosage_classification": None,
        "consistency_classification": None,
        "final_classification": None,
        "explanations": [],
        "consensus_explanation": ""
    }


def main():
    """Función principal del CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluador de Transcripciones Médicas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python med_eval.py original.json transcribed.json
  python med_eval.py data/original.json data/transcribed.json --output results.json
  python med_eval.py file1.json file2.json --quiet
        """
    )

    parser.add_argument(
        'original_file',
        help='Archivo JSON con el texto original'
    )

    parser.add_argument(
        'transcribed_file',
        help='Archivo JSON con el texto transcrito'
    )

    parser.add_argument(
        '--output', '-o',
        help='Nombre del archivo de salida (por defecto: results.json en el directorio del archivo transcrito)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Modo silencioso: solo mostrar la clasificación final'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Modo detallado: mostrar información adicional del proceso'
    )

    args = parser.parse_args()

    # Validar que los archivos existen
    original_path = Path(args.original_file)
    transcribed_path = Path(args.transcribed_file)

    if not original_path.exists():
        print(f"❌ Error: Archivo original no encontrado: {args.original_file}")
        sys.exit(1)

    if not transcribed_path.exists():
        print(f"❌ Error: Archivo transcrito no encontrado: {args.transcribed_file}")
        sys.exit(1)

    if args.verbose:
        print("🔍 Iniciando evaluación de transcripción médica...")
        print(f"   Archivo original: {args.original_file}")
        print(f"   Archivo transcrito: {args.transcribed_file}")

    try:
        # Cargar archivos JSON
        if args.verbose:
            print("📂 Cargando archivos...")
        original_data = load_json_file(args.original_file)
        transcribed_data = load_json_file(args.transcribed_file)

        # Extraer texto de los archivos
        if args.verbose:
            print("📝 Extrayendo texto...")
        original_text = extract_text_from_json(original_data, args.original_file)
        transcribed_text = extract_text_from_json(transcribed_data, args.transcribed_file)

        if args.verbose:
            print(f"   Texto original: {len(original_text)} caracteres")
            print(f"   Texto transcrito: {len(transcribed_text)} caracteres")

        # Crear estado de evaluación
        evaluation_state = create_evaluation_state(original_text, transcribed_text)

        # Ejecutar evaluación
        if args.verbose:
            print("🔬 Ejecutando evaluación con IA...")
        result = medication_evaluation_graph.invoke(evaluation_state)

        # Determinar ruta de salida
        if args.output:
            output_path = args.output
        else:
            # Crear results.json en el mismo directorio que el archivo transcrito
            output_path = transcribed_path.parent / "results.json"

        # Guardar resultados en JSON
        if args.verbose:
            print(f"💾 Guardando resultados en: {output_path}")
        save_results_json(result, str(output_path))

        # Mostrar resultados en terminal
        format_terminal_output(result, args.quiet)

        # Resumen final
        if not args.quiet:
            print("\n✅ Evaluación completada exitosamente!")
            print(f"📄 Resultados guardados en: {output_path}")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
