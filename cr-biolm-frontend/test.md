4. INTERACCIÓN DE COMPONENTES Y FLUJO DE DATOS
================================================

4.1 DIAGRAMA DE FLUJO DE DATOS

┌─────────┐     (1) especie      ┌────────────┐
│ Usuario │ ──────────────────► │ InputPanel │
└─────────┘                      └──────┬─────┘
                                        │
                                   (2) startJob()
                                        │
                                        ▼
                               ┌─────────────────┐
                               │                 │
                               │  API Backend    │
                               │                 │
                               └────────┬────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
               (3) jobId            (4) results         (7) llmTexts
                    │                   │                   │
                    ▼                   ▼                   ▼
            ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
            │ StatusConsole│     │  Dashboard  │     │ LLM Profiles│
            └─────────────┘     │   State     │     └──────┬──────┘
                                └──────┬──────┘            │
                                       │                   │
                                  (5) success         (6) getLLMProfile
                                       │                   │
                                       └─────────┬─────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │                 │
                                        │  API Backend    │
                                        │                 │
                                        └─────────────────┘


4.2 DIAGRAMA DE SECUENCIA

Usuario    InputPanel    API Backend    StatusConsole    Dashboard
   │           │              │              │              │
   │──(1)─────►│              │              │              │
   │           │──(2)────────►│              │              │
   │           │              │──(3)────────►│              │
   │           │              │              │──running────►│
   │           │              │              │              │
   │           │              │──(4)───────────────────────►│
   │           │              │              │              │──success
   │           │◄─(6)─────────│              │              │
   │           │──(7)─────────│              │              │
   │           │              │              │              │──display
   │           │              │              │              │


4.3 DIAGRAMA DE COMPONENTES

┌─────────────────────────────────────────────────────────────────┐
│                           DASHBOARD                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │             │    │             │    │                     │  │
│  │ InputPanel  │    │   Estado    │    │  Resultados         │  │
│  │             │    │   (running) │    │                     │  │
│  └──────┬──────┘    └─────────────┘    └──────────┬──────────┘  │
│         │                                          │             │
│         │ startJob()                              │ results     │
│         ▼                                          ▼             │
└─────────┼──────────────────────────────────────────┼─────────────┘
          │                                          │
          ▼                                          ▼
   ┌─────────────┐                            ┌─────────────┐
   │             │                            │             │
   │   Backend   │◄───────getLLMProfile───────│   LLM       │
   │     API     │                            │   Profiles  │
   │             │───────────llmTexts────────►│             │
   └─────────────┘                            └─────────────┘


4.4 LEYENDA

(1) especie      = Nombre científico ingresado por el usuario
(2) startJob()   = Función que inicia el pipeline
(3) jobId        = Identificador del trabajo en ejecución
(4) results      = Resultados completos del análisis
(5) success      = Indicador de éxito (true/false)
(6) getLLMProfile = Función que obtiene perfiles LLM
(7) llmTexts     = Textos generados por cada modelo LLM

───►  = Flujo de datos
◄───  = Retorno de datos