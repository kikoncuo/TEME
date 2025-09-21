# STT Dataset Generator

Una herramienta para generar datasets de evaluación de sistemas de reconocimiento de voz (STT) usando OpenAI para generar transcripciones estructuradas y múltiples proveedores TTS (ElevenLabs y Google Gemini) para crear conversaciones de audio.

## Características

- 🤖 **Generación automática de conversaciones** usando OpenAI 
- 🎙️ **Síntesis de voz realista** usando ElevenLabs TTS o Google Gemini TTS
- 📊 **Salida estructurada** con modelos Pydantic
- ⚡ **Procesamiento asíncrono** para generación eficiente en lotes
- 🎛️ **Interfaz de línea de comandos** fácil de usar
- 📁 **Organización automática** de archivos de dataset
- 🔄 **Múltiples proveedores TTS** - ElevenLabs (premium) y Gemini (gratuito)

## Instalación

1. Clona el repositorio y navega al directorio:
```bash
cd Dataset_creation
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

2.b Requisito del sistema: FFmpeg (obligatorio)

FFmpeg es necesario para la manipulación/convertido de audio (usado por pydub y para convertir WAV↔MP3).

- macOS (Homebrew):
```bash
brew install ffmpeg
```

- Linux (Debian/Ubuntu):
```bash
sudo apt update && sudo apt install -y ffmpeg
```

- Linux (Fedora/RHEL):
```bash
sudo dnf install -y ffmpeg
```

- Windows (winget o Chocolatey):
```powershell
# Con Chocolatey
choco install ffmpeg
```

Verificación rápida:
```bash
ffmpeg -version
```

Recuerda refrescar tu app o terminal si ffmpeg -version is not found before running the commands

3. Configura las claves API:
```bash
# Crea un archivo .env con tus claves API
echo "OPENAI_API_KEY=tu_clave_openai_aquí" > .env

# Para ElevenLabs (opcional - premium)
echo "ELEVEN_API_KEY=tu_clave_elevenlabs_aquí" >> .env

# Para Google Gemini TTS (opcional - gratuito)
echo "GOOGLE_API_KEY=tu_clave_google_aquí" >> .env
```

## Uso Rápido

### 1. Generar una conversación rápida

#### Con ElevenLabs (premium)
```bash
python cli.py quick-generate \
  --title "Consulta Médica" \
  --description "Un doctor consulta con un paciente sobre síntomas" \
  --context "Consultorio médico" \
  --participants "Doctor,Paciente" \
  --duration 90 \
  --difficulty medium \
  --domain medical \
  --language es \
  --tts-provider elevenlabs
```

#### Con Google Gemini TTS (gratuito)
```bash
python cli.py quick-generate \
  --title "Consulta Médica" \
  --description "Un doctor consulta con un paciente sobre síntomas" \
  --context "Consultorio médico" \
  --participants "Doctor,Paciente" \
  --duration 90 \
  --difficulty medium \
  --domain medical \
  --language es \
  --tts-provider gemini
```

### 2. Crear configuración de muestra
```bash
python cli.py create-sample-config --output scenarios.json
```

### 3. Generar dataset desde configuración

#### Con ElevenLabs
```bash
# Para un único escenario (prueba)
python cli.py generate --scenarios scenarios.json --single --tts-provider elevenlabs

# Para todo el lote
python cli.py generate --scenarios scenarios.json --max-concurrent 3 --tts-provider elevenlabs
```

#### Con Google Gemini TTS
```bash
# Para un único escenario (prueba)
python cli.py generate --scenarios scenarios.json --single --tts-provider gemini

# Para todo el lote
python cli.py generate --scenarios scenarios.json --max-concurrent 3 --tts-provider gemini
```

### 4. Generar audio desde un transcript existente

Cuando ya tienes un archivo JSON de transcripción (estructura `GeneratedConversation`) y solo quieres sintetizar el audio:

#### ElevenLabs (premium)
```bash
python cli.py synthesize-from-transcript \
  --transcript generated_datasets/gemini_test_medication/cardiology_consultation_01_b6e406b1_transcript.json \
  --language es \
  --tts-provider elevenlabs
```

#### Google Gemini TTS (gratuito)
```bash
python cli.py synthesize-from-transcript \
  --transcript generated_datasets/gemini_test_medication/cardiology_consultation_01_b6e406b1_transcript.json \
  --language es \
  --tts-provider gemini
```

Opcional: usar tus propios mapeos de voces
```bash
python cli.py synthesize-from-transcript \
  -t path/to/transcript.json \
  -l en \
  --tts-provider gemini \
  --voice-mappings voice_mappings_gemini_en.json
```

Notas:
- Si no pasas `--output`, el archivo se creará junto al transcript como `<base>_conversation.mp3` (ElevenLabs) o `<base>_conversation.wav` (Gemini).
- Si no pasas `--voice-mappings`, se cargarán automáticamente los mapeos por idioma y proveedor.

## Guía Paso a Paso Completa

Esta guía te lleva desde la configuración inicial hasta la generación de datasets listos para evaluación STT.

### Paso 1: Configuración Inicial

#### 1.1 Instalar Dependencias
```bash
# Instalar todas las dependencias requeridas
pip install -r requirements.txt
```

#### 1.2 Configurar Variables de Entorno
Crea un archivo `.env` en el directorio raíz:
```bash
# API Keys requeridas
OPENAI_API_KEY=your_openai_api_key_here

# TTS Provider Keys (al menos una requerida)
ELEVEN_API_KEY=your_elevenlabs_api_key_here      # Para ElevenLabs (premium)
GOOGLE_API_KEY=your_google_api_key_here          # Para Google Gemini TTS (gratuito)
```

**Nota**: Asegúrate de que las claves API tengan permisos adecuados:
- **OpenAI**: Para usar GPT-4 en generación de conversaciones
- **ElevenLabs**: Para síntesis de voz premium (requiere suscripción)
- **Google**: Para Gemini TTS (gratuito con cuotas)

### Paso 2: Preparar Archivos de Configuración

#### 2.1 Crear Archivo de Escenarios
Los escenarios definen las conversaciones que se generarán:

```bash
# Crear archivo de escenarios básico con ejemplos médicos
python cli.py create-sample-config
```

Esto crea `scenarios.json` con ejemplos de conversaciones médicas especializadas.

#### 2.2 Editar Escenarios Personalizados
Edita `scenarios.json` para definir tus propios escenarios:

```json
{
  "scenario_id": "consulta_cardiologia_01",
  "title": "Consulta Cardiología - Arritmias",
  "description": "Cardiólogo explica arritmias complejas al paciente",
  "context": "Consulta especializada donde el cardiólogo explica arritmias ventriculares y discute anticoagulantes",
  "participants": ["Cardiólogo", "Paciente"],
  "target_duration": 180,
  "difficulty_level": "hard",
  "language": "es",
  "domain": "cardiology"
}
```

**Consejos para editar escenarios:**
- **scenario_id**: Identificador único, usa formato `tipo_numero`
- **participants**: Roles que participarán (deben existir en `voice_mappings_*.json`)
- **difficulty_level**: "easy", "medium", "hard" (afecta complejidad técnica)
- **language**: "es" para español, "en" para inglés
- **target_duration**: Duración objetivo en segundos (180-300 recomendado)

#### 2.3 Configurar Mapeos de Voces
Los archivos `voice_mappings_es.json` y `voice_mappings_en.json` mapean roles a voces de ElevenLabs:

```json
{
  "speaker_name": "Cardiólogo",
  "voice_id": "ID_DE_VOICE_ELEVENLABS",
  "voice_name": "Nombre de la Voz",
  "voice_description": "Voz masculina profesional para cardiólogos"
}
```

**Para obtener voice_id de ElevenLabs:**
Visitar
https://elevenlabs.io/app/voice-library
Hacer click en los 3 puntos de la derecha de la voz deseada
Click en "Copy Voice ID"

O desde la terminal:
```bash
# Listar voces disponibles (requiere API key configurada)
curl -X GET "https://api.elevenlabs.io/v1/voices" \
  -H "xi-api-key: $ELEVEN_API_KEY"
```

### Paso 3: Generar Dataset Básico

#### 3.1 Generación Rápida (Prueba)
```bash
# Generar una conversación simple para probar el sistema
python cli.py quick-generate \
  --context "Consulta médica básica sobre síntomas de gripe" \
  --participants "Doctor,Paciente" \
  --language es
```

**Resultado:** Un archivo MP3 y JSON en `generated_datasets/quick_*`
- `conversation.mp3`: Audio de la conversación generada
- `conversation.json`: Transcripción estructurada con metadatos

#### 3.2 Generación por Lotes
```bash
# Generar múltiples conversaciones desde archivo de escenarios
python cli.py generate \
  --scenarios scenarios.json \
  --max-concurrent 2
```

**Opciones importantes:**
- `--max-concurrent`: Número de conversaciones a generar en paralelo (2-5 recomendado)
- `--output-dir`: Directorio personalizado para salida
- `--language`: Idioma específico ("es", "en", "auto")

**Resultado esperado:**
```
generated_datasets/
└── batch_20240101_120000/
    ├── conversation_01.json          # Transcripción estructurada
    ├── conversation_01.mp3           # Audio generado
    ├── conversation_02.json
    ├── conversation_02.mp3
    └── metadata.json                 # Información del lote
```

### Paso 4: Post-Procesamiento de Audio (Opcional pero Recomendado)

#### 4.1 Añadir Variabilidad Realista
```bash
# Procesar un archivo individual con efectos aleatorios
python cli.py process-audio \
  -i generated_datasets/batch_xxx/conversation_01.mp3 \
  --noise-level 0.05 \
  --speed-variation 0.1 \
  --volume-variation 0.2 \
  --seed 42
```

#### 4.2 Procesar Directorio Completo
```bash
# Procesar todo un lote con efectos aleatorios
python cli.py process-audio \
  -i generated_datasets/batch_xxx \
  --noise-level 0.03 \
  --speed-variation 0.15 \
  --volume-variation 0.3
```

**Efectos aplicados aleatoriamente:**
- **Ruido de fondo**: Simula ambientes ruidosos (oficinas, hospitales)
- **Variación de velocidad**: Diferentes ritmos de habla (±10-15%) (WIP)
- **Variación de volumen**: Diferentes distancias de grabación (±0.2-0.3dB) (WIP)

**Resultado:**
```
conversation_01.mp3
├── conversation_01_processed_a1b2c3.mp3    # Con ruido de fondo
├── conversation_01_processed_d4e5f6.mp3    # Velocidad modificada
└── conversation_01_processed_g7h8i9.mp3    # Volumen ajustado
```

### Paso 5: Validación y Verificación

#### 5.1 Validar Dataset
```bash
# Verificar integridad del dataset generado
python cli.py validate \
  --directory generated_datasets/batch_xxx
```

**Verificaciones realizadas:**
- ✅ Archivos de audio existen y son válidos
- ✅ Transcripciones JSON tienen estructura correcta
- ✅ Metadatos incluyen información requerida
- ✅ Duraciones de audio coinciden con expectativas

#### 5.2 Obtener Información del Dataset
```bash
# Mostrar estadísticas del lote generado
python cli.py info \
  --directory generated_datasets/batch_xxx
```

### Paso 6: Flujo de Trabajo Completo

#### Ejemplo: Dataset Médico en Español Completo
```bash
# 1. Configurar API keys
echo "OPENAI_API_KEY=sk-..." > .env
echo "ELEVEN_API_KEY=..." >> .env
echo "GOOGLE_API_KEY=..." >> .env

# 2. Crear escenarios médicos de ejemplo
python cli.py create-sample-config

# 3. Editar escenarios para especialidades específicas
# (editar scenarios.json manualmente)

# 4. Generar dataset básico con ElevenLabs (premium)
python cli.py generate \
  --scenarios scenarios.json \
  --max-concurrent 3 \
  --language es \
  --tts-provider elevenlabs

# O con Google Gemini TTS (gratuito)
python cli.py generate \
  --scenarios scenarios.json \
  --max-concurrent 3 \
  --language es \
  --tts-provider gemini

# 5. Añadir variabilidad para evaluación robusta
python cli.py process-audio \
  -i generated_datasets/batch_latest \
  --noise-level 0.05 \
  --speed-variation 0.1

# 6. Validar resultado final
python cli.py validate \
  --directory generated_datasets/batch_latest
```

### Paso 7: Estructura Final del Dataset

Después de completar todos los pasos, tendrás:

```
generated_datasets/
└── batch_20240101_120000/
    ├── metadata.json                    # Información del lote
    ├── conversation_01.json            # Transcripción original
    ├── conversation_01.mp3             # Audio original
    ├── conversation_01_processed_abc123.mp3  # Audio procesado 1
    ├── conversation_01_processed_def456.mp3  # Audio procesado 2
    ├── conversation_02.json
    ├── conversation_02.mp3
    └── batch_cardiology_consultation_01_processed/
        └── [archivos procesados adicionales]
```

### Consejos y Mejores Prácticas

#### Configuración de Escenarios
- **Difícil**: Incluye terminología médica específica, nombres de drogas, síntomas complejos
- **Participantes**: Usa roles específicos (Cardiólogo, Enfermera, Familiar del Paciente)
- **Duración**: 180-300 segundos para conversaciones realistas
- **Dominio**: Especifica el campo médico para terminología apropiada

#### Optimización de Audio
- **Post-procesamiento**: Mejora drásticamente la evaluación STT en condiciones reales
- **Seeds**: Usa `--seed` para resultados reproducibles en tests
- **Variabilidad**: Combina diferentes niveles de ruido y velocidad

#### Gestión de Recursos
- **Concurrente**: Ajusta `--max-concurrent` según límites de tu plan de API
- **Lotes**: Divide escenarios grandes en archivos separados
- **Monitoreo**: Revisa logs para identificar problemas de API temprano

### 4. Procesar audio (opcional)
```bash
# Procesar archivo individual con efectos aleatorios
python cli.py process-audio -i archivo.mp3 --seed 42

# Procesar directorio completo
python cli.py process-audio -i ./generated_datasets/batch_xxx

# Procesar con parámetros personalizados
python cli.py process-audio \
  -i ./generated_datasets/batch_xxx \
  --noise-level 0.08 \
  --speed-variation 0.15 \
  --volume-variation 0.3 \
  --noise-types "white" \
  --seed 123
```

### 5. Validar dataset generado
```bash
python cli.py validate --directory ./generated_datasets/batch_20231201_143022
```

## Estructura de Archivos de Salida

```
generated_datasets/
└── batch_20231201_143022/
    ├── batch_20231201_143022_metadata.json    # Metadatos del lote
    ├── medical_consultation_01_a1b2c3d4_conversation.mp3      # Audio
    ├── medical_consultation_01_a1b2c3d4_transcript.json       # Transcripción
    ├── medical_consultation_01_a1b2c3d4_metadata.json         # Metadatos entrada
    └── ... (más entradas)
```

## Configuración de Escenarios

Los escenarios se definen en formato JSON. Ejemplo:

```json
[
  {
    "scenario_id": "medical_consultation_01",
    "title": "Consulta Médica",
    "description": "Un doctor consulta con un paciente sobre síntomas",
    "context": "Consultorio médico donde un doctor examina a un paciente con síntomas de gripe",
    "participants": ["Doctor", "Paciente"],
    "target_duration": 90,
    "difficulty_level": "medium",
    "language": "en",
    "domain": "medical"
  }
]
```

### Campos de Escenario

- **scenario_id**: Identificador único
- **title**: Título de la conversación
- **description**: Descripción breve
- **context**: Contexto o setting para la conversación
- **participants**: Lista de nombres/roles de participantes
- **target_duration**: Duración objetivo en segundos
- **difficulty_level**: "easy", "medium", "hard"
- **language**: Código de idioma (ej: "en", "es")
- **domain**: Dominio específico (medical, business, casual, etc.)

### Niveles de Dificultad

- **Easy**: Frases claras y simples, superposiciones mínimas, habla formal
- **Medium**: Habla natural con algunos elementos informales, superposiciones ocasionales  
- **Hard**: Frases complejas, habla informal, interrupciones, referencias a ruido de fondo

## Configuración de Voces

El sistema carga automáticamente los mapeos de voz desde archivos JSON según el idioma especificado. Puedes personalizar las voces editando los archivos `voice_mappings_en.json` (inglés) y `voice_mappings_es.json` (español)

### Mapeos por Idioma
- **Español**: `voice_mappings_es.json` (25 voces optimizadas para español)
- **Inglés**: `voice_mappings_en.json` (25 voces optimizadas para inglés)
- **Selección automática**: Basada en el campo `language` del escenario

### Voces Recomendadas por Especialidad Médica
**Profesionales Médicos:**
- **Cardiólogo**: Rachel (voz profesional y precisa)
- **Oncólogo/Radio-oncólogo**: Sarah (voz autoritaria y empática)
- **Neurocirujano/Neurólogo**: Adam (voz calmada y técnica)
- **Psiquiatra**: Sarah (voz empática y profesional)
- **Cirujano/Anestesiólogo**: Rachel (voz autoritaria y calmada)

**Pacientes y Familiares:**
- **Paciente**: Domi (voz conversacional y natural)
- **Padre**: Domi (voz preocupada masculina)
- **Madre**: Rachel (voz angustiada femenina)
- **Familiar**: Domi (voz familiar preocupada)

**Personal Médico Auxiliar:**
- **Enfermera**: Sarah (voz clara y profesional)
- **Residente**: Josh (voz joven y técnica)
- **Perfusionista**: Josh (voz técnica especializada)
- **Endoscopista**: Adam (voz técnica especializada)

## Configuración de Audio

La herramienta utiliza **ElevenLabs v3** con configuración optimizada para STT:

### Configuración Predeterminada
- **Modelo**: `eleven_v3` (modelo más avanzado con audio tags)
- **Formato**: `mp3_44100_128` (alta calidad, tamaño optimizado)
- **Estabilidad**: `0.5` (equilibrio entre expresividad y consistencia)
  - `0.3`: Creativa (más emocional, mejor para tags)
  - `0.5`: Natural (equilibrio recomendado)
  - `0.8`: Robusta (muy estable, menos expresiva)
- **Speaker Boost**: `True` (mejora similitud con voz original)
- **Text Normalization**: `auto` (manejo automático de números y símbolos)
- **Audio Tags**: `True` (habilitado para expresividad emocional)

### Audio Tags Automáticos (v3)
La herramienta aplica automáticamente audio tags basados en características de voz generadas por OpenAI:

**Mapeo Automático de Características a Tags:**
- `"anxious"` → `[nervous]`
- `"whispering"` → `[whispers]`
- `"excited"` → `[excited]`
- `"questioning"` → `[questioning]`
- `"professional"` → `[professional]`
- `"warm"` → `[warm]`
- `"curious"` → `[curious]`
- `"frustrated"` → `[frustrated]`
- `"reassuring"` → `[reassuring]`

**Tags Contextuales Adicionales:**
- `[pause]` - Pausas naturales entre turnos
- `[emphasis]` - Énfasis en términos técnicos, nombres de medicamentos, siglas médicas
- `[sighs]` - Suspiros para marcadores de duda ("um", "well", "uh")

### Configuración por Idioma
- **Español (`--language es`)**: Usa mapeos optimizados para español
- **Inglés (`--language en`)**: Usa mapeos optimizados para inglés
- **Otros idiomas**: Compatibilidad automática con voces multilingües

### Calidad de Audio
- **Archivo de salida**: MP3 de alta calidad (~1.2-1.5MB por conversación)
- **Tasa de muestreo**: 44.1kHz (calidad CD)
- **Bitrate**: 128kbps (equilibrio calidad/tamaño)
- **Conversión**: Una sola llamada API usando Text to Dialogue

## Procesamiento de Audio Post-Generación

El sistema incluye herramientas avanzadas para **post-procesamiento de audio** que añaden variabilidad realista a los datasets, mejorando la evaluación de sistemas STT en condiciones del mundo real.

### Efectos Disponibles

#### **Ruido de Fondo**
- **Tipos**: Blanco, rosa, marrón (dependiendo de la versión de pydub)
- **Nivel**: Configurable de 0.0 a 0.2
- **Propósito**: Simular ambientes ruidosos (oficinas, hospitales, transporte)

#### **Variación de Velocidad**
- **Rango**: ± porcentaje configurable (default: ±10%)
- **Aplicación**: 70% de probabilidad de modificación
- **Propósito**: Simular diferentes ritmos de habla

#### **Variación de Volumen**
- **Rango**: ±dB configurable (default: ±0.2dB)
- **Aplicación**: 80% de probabilidad de modificación
- **Propósito**: Simular diferentes distancias y condiciones de grabación

### Beneficios para Evaluación STT

✅ **Mayor Robustez**: Entrenar modelos con audio variable
✅ **Realismo**: Simular condiciones del mundo real
✅ **Diversidad**: Crear múltiples variantes del mismo contenido
✅ **Evaluación**: Medir rendimiento en condiciones adversas
✅ **Reproducibilidad**: Usar `--seed` para resultados consistentes

### Archivos Generados

```
original_audio.mp3
├── original_audio_processed_a1b2c3.mp3  # Con ruido + velocidad
├── original_audio_processed_d4e5f6.mp3  # Solo volumen modificado
└── original_audio_processed_g7h8i9.mp3  # Sin modificaciones
```

### Ejemplos de Uso

```bash
# Dataset básico para evaluación estándar
python cli.py generate --scenarios scenarios.json

# Dataset con variabilidad para evaluación robusta
python cli.py generate --scenarios scenarios.json
python cli.py process-audio -i ./generated_datasets/batch_xxx \
  --noise-level 0.05 --speed-variation 0.1 --volume-variation 0.2

# Crear múltiples variantes de un archivo específico
python cli.py process-audio -i archivo.mp3 --seed 42
python cli.py process-audio -i archivo.mp3 --seed 43
python cli.py process-audio -i archivo.mp3 --seed 44
```

## API Programática

También puedes usar la herramienta programáticamente:

```python
from dotenv import load_dotenv
from dataset_generator import STTDatasetGenerator
from models import ConversationScenario

load_dotenv()

# Inicializar generador
generator = STTDatasetGenerator()

# Crear escenario
scenario = ConversationScenario(
    scenario_id="test_01",
    title="Conversación de Prueba",
    description="Una conversación simple para probar",
    context="Llamada telefónica casual",
    participants=["Alice", "Bob"],
    target_duration=60,
    difficulty_level="easy"
)

# Generar entrada de dataset
entry = generator.generate_single_dataset_entry(scenario)
print(f"Generado: {entry.entry_id}")
```

## Uso para Evaluación STT

Los datasets generados están listos para evaluación STT:

1. **Archivo de Audio**: Conversación multiparlante en MP3
2. **Transcripción**: Texto de referencia con información de hablante
3. **Metadatos**: Información sobre voces, configuración, etc.

Ejemplo de uso con un sistema STT:

```python
import json
from pathlib import Path

# Cargar entrada del dataset
with open("entry_metadata.json") as f:
    entry_metadata = json.load(f)

audio_file = entry_metadata["audio_file_path"]
transcript_file = entry_metadata["transcript_file_path"]

# Cargar transcripción de referencia
with open(transcript_file) as f:
    reference = json.load(f)

# Tu sistema STT
predicted_text = your_stt_system.transcribe(audio_file)

# Comparar con referencia
reference_text = " ".join([turn["text"] for turn in reference["turns"]])
accuracy = calculate_accuracy(predicted_text, reference_text)
```

## Solución de Problemas

### Error de Clave API
```
ValueError: OpenAI API key is required
```
→ Asegúrate de que el archivo `.env` esté presente con claves API válidas

### Error de Proveedor TTS
```
ValueError: Gemini TTS generator not initialized
```
→ Configura `GOOGLE_API_KEY` en tu archivo `.env` para usar Gemini TTS

### Comparación de Proveedores TTS

| Característica | ElevenLabs | Google Gemini |
|----------------|------------|---------------|
| **Costo** | Premium (pago) | Gratuito (con cuotas) |
| **Calidad** | Excelente | Buena |
| **Voces** | Múltiples opciones | Voces estándar |
| **Latencia** | Baja | Media |
| **Límites** | Según plan | Cuotas diarias |

### Recomendaciones de Uso

- **Para desarrollo/pruebas**: Usa Gemini TTS (gratuito)
- **Para producción**: Usa ElevenLabs (mejor calidad)
- **Para grandes volúmenes**: Combina ambos según necesidades

### Error de Voz No Encontrada
```
Warning: No voice mapping found for speaker 'X'
```
→ Añade mapeos de voz para todos los hablantes en tus escenarios

### Error de Memoria/Rate Limiting
→ Reduce `--max-concurrent` para procesamiento por lotes

## Arquitectura

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   CLI Interface │    │   Scenario   │    │   Generated     │
│                 │───▶│ Definition   │───▶│  Conversation   │
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                     │
┌─────────────────┐    ┌──────────────┐             ▼
│  Audio Files +  │◀───│  ElevenLabs  │    ┌─────────────────┐
│  Transcripts    │    │   TTS API    │◀───│   OpenAI API    │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

## Próximos Pasos

1. **Añadir más idiomas**: Expandir soporte multilingüe
2. **Ruido de fondo**: Incluir audio ambiente para mayor realismo
3. **Métricas automáticas**: Calcular métricas de calidad del dataset
4. **Integración STT**: Conectores directos a sistemas STT populares
5. **Interfaz web**: GUI para configuración más fácil

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature
3. Añade pruebas
4. Envía un pull request

## Licencia

MIT License - ver archivo LICENSE para detalles.
