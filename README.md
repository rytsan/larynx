# LARYNX

Plugin de áudio (CLAP / VST3, mono) de reconstrução harmônica vocal — um "DLSS para voz".
Analisa a voz em harmônicos, reconstrói o espectro superior com uma rede neural pequena rodando em CPU
e devolve a voz mais rica, preservando pitch, vibrato, jitter e intenção do cantor. Não afina, não decide nota.

Projeto pessoal, não comercial. O plano completo (requisitos, arquitetura, passos e gates) está em [`planner.md`](planner.md).

## Status

| Passo | Conteúdo | Estado |
|---|---|---|
| 1 | Casca do VST: build, parâmetros, bypass bit-perfect, UI estática | código pronto, testes passando; falta validar em DAW |
| 2 | Analyzer (F0 / voicing) | código + testes sintéticos; falta validar flags e F0 em voz real |
| 3 | Síntese identidade | identidade/LSD/roundtrip/RTF verdes; faltam ABX e escuta com máscara |
| 4 | Controles harmônicos manuais | Brightness/Air/Even-Odd/Formant/Mix (`core/blend.h`), gráfico real + leitor de vogal, VU; falta uso real no DAW |
| 5 | Dados e treino | golden congelado; pycore, shards, QC, baseline, TCN, treino e export rodando ponta a ponta; corpora baixando |
| 6–7 | Neural Live / Studio | inferência C++ (`core/neuralcore`, = PyTorch em 4e-7), Live e Studio (motor atrasado D hops, PDC) no plugin, RTF 0.12/0.16 com a TCN completa; faltam pesos treinados |
| 8–9 | Camada criativa, fechamento | — |

O plugin reporta a latência final (1024 amostras @48 kHz ≈ 21.3 ms). O caminho processado já é analyzer → skeleton → synth
(máscara de ganho sobre a STFT original + aditivo para harmônicos ausentes) com todos os ganhos = 1: sai a entrada atrasada,
erro ≤ −130 dBFS. Live/Studio usam o mesmo caminho até o P6. O bypass continua cópia bit-perfect atrasada pela latência.

## Instalação rápida (Linux / WSL)

```bash
./install.sh            # deps (apt), venv, build + testes, golden set, gate pycore, plugin em ~/.vst3 e ~/.clap
./install.sh --data     # + baixa corpora (~30 GB) e gera shards
./install.sh --train    # + treina Live e Studio e exporta plugin/source/neural/model_{live,studio}.json
```

Opções: `--no-apt`, `--hours N` (teto por corpus nos shards), `--steps N`, `--jobs N`. Tudo é idempotente.

## Build

Requer CMake ≥ 3.24, compilador C++20 e git. JUCE 8, clap-juce-extensions e Catch2 são baixados pelo CMake
(versões fixadas por commit/tag em `CMakeLists.txt` e `tests/CMakeLists.txt`).

### Linux / WSL

```bash
sudo apt install -y build-essential cmake ninja-build pkg-config \
  libasound2-dev libfreetype-dev libfontconfig1-dev libx11-dev libxext-dev libxrender-dev \
  libxrandr-dev libxinerama-dev libxcursor-dev libxcomposite-dev libgl-dev libglu1-mesa-dev mesa-common-dev

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

### Windows

Visual Studio 2022 com "Desktop development with C++":

```bat
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

### macOS

```bash
cmake -B build -G Xcode
cmake --build build --config Release
```

Artefatos em `build/plugin/Larynx_artefacts/Release/`:

| Formato | Arquivo | Instalar em (Windows) |
|---|---|---|
| VST3 | `VST3/LARYNX.vst3` | `C:\Program Files\Common Files\VST3\` |
| CLAP | `CLAP/LARYNX.clap` | `C:\Program Files\Common Files\CLAP\` |
| Standalone | `Standalone/LARYNX` | — (útil para ver a UI sem DAW) |

No WSL, o Standalone abre via WSLg com `DISPLAY=:0`.

## Testes

`tests/test_processor.cpp` (Catch2) cobre:

- bypass bit-perfect a 44.1 e 48 kHz (`out[n] == in[n − latência]`), com blocos de tamanho irregular
- bypass pelo host (`processBlockBypassed`)
- caminho processado com trims em 0 dB também bit-perfect
- volta ao bit-perfect após o crossfade de 10 ms do bypass
- roundtrip de estado de todos os parâmetros
- `mode` não automatizável (muda latência)

`tests/test_analyzer.cpp` cobre o analyzer: senóides 65–1000 Hz (< 10 cents), vibrato 6 Hz ±50 cents
(< 15 cents), vogal sintética de 30 s com fundamental fraca (sem salto de oitava), silêncio/ruído/sibilante
e transientes sobre vogal.

`tests/test_engine.cpp` cobre o caminho DSP: identidade (< −100 dBFS a 44.1/48k), extração de senóides e de pilhas
harmônicas (± 0.2 dB), energia das bandas de ruído, máscara (−6 dB no H3) e aditivo (H15 ausente injetado a −30 dB).

```bash
./build/tools/bench_plugin/larynx_analyze golden/8_vozes/alto_ru/cantado.wav > frames.csv   # RTF no stderr
./build/tools/bench_plugin/larynx_render  golden/8_vozes/alto_ru/cantado.wav out.wav        # identidade: erro, LSD, RTF
./build/tools/bench_plugin/larynx_render  in.wav out.wav --bright 6 --evenodd -1            # máscara ativa (escuta do P3)
```

`out.wav` sai alinhado com a entrada (latência removida), pronto para ABX. Opções nas unidades dos knobs, mesmo
`core/blend.h` do plugin: `--bright` (dB/oct), `--air` (dB), `--evenodd` (−1…1 = ±6 dB), `--formant` (st), `--mix` (0…1).

Deriva de vogal (NFR5) usa Praat via parselmouth, num venv local:

```bash
python3 -m venv .venv && .venv/bin/pip install praat-parselmouth numpy
.venv/bin/python ml/eval/metrics.py in.wav out.wav [in2.wav out2.wav ...]   # mediana/p95 de |ΔF1|, |ΔF2| em %
```

## Trilha ML

Tudo num venv local (o `CMakeLists` compila `build/ml/pycore` com o Python do sistema; mesma versão do venv):

```bash
python3 -m venv .venv && .venv/bin/pip install praat-parselmouth numpy scipy soundfile pyloudnorm
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
PYTHONPATH=build/ml/pycore .venv/bin/python ml/pycore/check_equivalence.py golden/8_vozes/*/*.wav  # pycore == C++
ml/data/download_corpora.sh                                    # VocalSet + VCTK em data/ (~17 GB)
.venv/bin/python ml/data/build_shards.py vocalset vctk --max-hours 10
.venv/bin/python ml/train/train.py --name live0                # Studio: --delay 16
.venv/bin/python ml/train/model.py data/runs/live0/best.pt model_live.json
```

## Golden set

Conjunto de teste congelado — **nunca** usado em treino. 8 cantores do [GTSinger](https://huggingface.co/datasets/AaronZ345/GTSinger)
(CC BY-NC-SA 4.0) × 3 clipes (cantado, técnica, fala), mono 48 kHz. Os WAVs não ficam no git; o manifest e os hashes ficam.

```bash
python3 ml/data/fetch_golden.py   # baixa o que faltar e verifica SHA256SUMS
```

A primeira execução escolheu os clipes numa revisão fixa do dataset e gravou `golden/manifest.csv` e
`golden/8_vozes/SHA256SUMS`. As seguintes nunca reescolhem: falham se algum hash não bater.
O corpus GTSinger inteiro fica fora do treino. Créditos em [`ml/data/credits.md`](ml/data/credits.md).

## Estrutura

```text
plugin/
  CMakeLists.txt
  source/
    PluginProcessor.*   parâmetros, latência, bypass, estado
    PluginEditor.*      UI 900×540 renderizada via OpenGL
    core/               DSP puro, sem JUCE (testável headless): analyzer, skeleton, synth, engine, fft, harm_frame
    ui/                 LookAndFeel do rack, gráfico de harmônicos
tests/                  Catch2
tools/bench_plugin/     larynx_analyze (analyzer → CSV + RTF), larynx_render (caminho DSP → WAV + RTF/LSD)
ml/pycore/              pybind11 sobre core/ (analyze, net_input) + gate de equivalência
ml/data/                fetch_golden.py, download_corpora.sh, build_shards.py (QC + análise C++), masks.py
ml/eval/                metrics.py (deriva F1/F2), baseline.py (regressão de declínio + LSD)
ml/train/               model.py (TCN causal + export JSON), train.py (Live D=0 / Studio D=16)
golden/                 manifest + hashes do golden set
planner.md              plano do projeto
```

## Parâmetros

IDs são contrato de automação — nunca renomear, só acrescentar.

| Parâmetro | ID | Range | Default |
|---|---|---|---|
| Mode | `mode` | DSP / Live / Studio (não automatizável) | DSP |
| Prior | `prior` | 0–100 % | 70 |
| Mix | `mix` | 0–100 % | 100 |
| Brightness | `bright` | −12…+12 dB/oct | 0 |
| Air | `air` | −12…+12 dB | 0 |
| Even/Odd | `evenodd` | −1…+1 | 0 |
| Formant | `formant` | −6…+6 st | 0 |
| Sub | `sub` | 0–100 % | 0 |
| Breath | `breath` | 0–100 % | 0 |
| In / Out trim | `in`, `out` | −24…+24 dB | 0 |
| Bypass | `bypass` | bool | off |

No Passo 1 só `in`, `out` e `bypass` afetam o áudio.
