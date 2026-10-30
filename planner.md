# Projeto LARYNX
VST de reconstrução harmônica vocal ("DLSS vocal") · pessoal, não comercial · voz, tempo real, ML
Documento único e autocontido. Nada aqui depende de lembrar conversas anteriores.

> **Revisão 2026-10-15 (riscos de implementação).** Correções de contradições internas e de riscos que impediriam os gates. Resumo das mudanças (detalhe em §16, R9–R16):
> 1. Restauração deixa de ser ressíntese total com fase propagada: vira **máscara de ganho sobre o STFT original** (fase original preservada) + aditivo só para harmônicos ausentes. Remove a maior parte do R1 (metálico) e torna o gate P3 alcançável.
> 2. Ordem do M4 corrigida: Prior escala só a rede; controles manuais vêm depois (antes, Prior=0 desligava Brightness/Air/etc. e o "delta-only" impedia cortes).
> 3. Studio redefinido como **mesma TCN causal com alvo atrasado (lookahead fixo de 16 frames)**: o RF bidirecional ±0.5 s era incompatível com PDC de 80–120 ms.
> 4. TCN: "5 canais residuais" era incompatível com ~300–500k params → 96 canais.
> 5. I/O da rede: entrada passa a incluir as 8 bandas de ruído (sem elas `delta_noise` é impossível) e máscaras de Nyquist/banda no loss.
> 6. Dados: fontes de 24 kHz (LibriTTS-R, JVS) ensinariam a rede a apagar agudos → máscara de banda efetiva no loss. Sua voz fora do treino (estava no treino e no golden ao mesmo tempo).
> 7. Trilha ML depende do C++ (pybind11) do P2/P3 → P5 dividido em 5a (imediato) e 5b (após P3).
> 8. Métrica ABX corrigida ("confusão ≥ 70%" é impossível: chute puro dá 50%).
> 9. Bypass bit-perfect redefinido com latência reportada; Mode não automatizável; troca de modelo sem alocação no audio thread.

## 0. Resumo do produto (a folha de rosto)
LARYNX é um plugin de áudio (CLAP/VST3, mono) que analisa a voz em harmônicos, reconstrói o espectro superior com uma rede neural pequena rodando em CPU, e devolve a voz mais rica — preservando pitch, vibrato, jitter e intenção do cantor. Não afina, não decide nota, não robotiza. Tem um modo Live (causal, para cantar monitorado) e um modo Studio (lookahead, qualidade máxima), mais uma camada Criativa (sub-harmônicos, sopro, morphing de timbre). UI de rack analógico com visualizador de harmônicos em tempo real.

| Parâmetro | Detalhe |
|---|---|
| Entrada | voz solo mono, 44.1/48 kHz |
| Motor | DSP determinístico + TCN (~300–500k params, RTNeural) |
| GPU no desenvolvimento | RTX 2080Ti (treino) → export p/ CPU (uso) |
| Formatos | CLAP primeiro, VST3 junto; Win/macOS |
| Latência | Live ≤ 25 ms · Studio ~80–120 ms (PDC) |
| Filosofia | delta-prediction: a rede só acrescenta o que falta, nunca reescreve o que já existe |

## 1. Requisitos travados (não reabrir em discussão)

### 1.1 Funcionais
| ID | Requisito |
|---|---|
| RF1 | Voz solo mono 44.1/48 kHz |
| RF2 | F0 em tempo real como andaime interno (nunca exposto como correção) |
| RF3 | Decomposição harmônica: H1..H80 + 8 bandas de ruído |
| RF4 | Neural upsampler: espectro pobre → espectro rico, blend preservativo |
| RF5 | Síntese híbrida: harmônicos **existentes** (qualquer n, input acima do piso) = máscara de ganho sobre os bins do STFT original, fase original intacta (sem phase vocoder — pitch nunca muda); harmônicos **ausentes** (reconstruídos) e criação (M5) = banco aditivo de fase sintética coerente (2πn∫f0dt) |
| RF6 | Knob Prior (0% transparente → 100% neural) |
| RF7 | Mix dry/wet, in/out trim (Mix opera no domínio espectral sobre os envelopes harmônicos antes da ressíntese, não no domínio temporal) |
| RF8 | Controles harmônicos: Brightness, Air, Even/Odd, Formant |
| RF9 | Camada criativa: Sub, Breath, Morph (v2), editor de envelope no gráfico |
| RF10 | Todos os parâmetros automatizáveis, sem glitches — exceto `mode` (muda latência/PDC; não automatizável) |
| RF11 | Jitter/shimmer/vibrato atravessam o sistema (critério de aceitação medido) |
| RF12 | Funciona com qualquer voz sem fine-tune (split por cantor no treino) |

### 1.2 Não-funcionais
| ID | Alvo |
|---|---|
| NFR1 | Latência algorítmica Live ≤ 25 ms |
| NFR2 | RTF < 0.2 em CPU de laptop 2018+ sem GPU |
| NFR3 | Zero flutter espectral (phase-locking + smoothing) |
| NFR4 | Bypass bit-perfect: saída = entrada atrasada exatamente pela latência reportada (`getLatencySamples`), após o crossfade de 10 ms da transição |
| NFR5 | Preservação de vogal: deriva F1/F2 abaixo de limiar calibrado no P3 |
| NFR6 | Vida: jitter/shimmer mudam < 10% pós-processo |
| NFR7 | ABX (original rico vs. restaurado), ≥ 3 ouvintes × 20 trials: taxa de **acerto** ≤ 65% Studio, ≤ 70% Live (chance = 50%; "confusão ≥ 60%" era inatingível) |
| NFR8 | Sessão de 1 h sem dropout/glitch |

### 1.3 Não-requisitos (fora de escopo, explícitos)
Correção de pitch · snap a escala · polifonia · clonagem de voz · difusão em RT · ARA/edição offline.

## 2. Arquitetura técnica

### 2.1 Fluxo de sinal (contrato entre módulos)
```text
IN ──► [M1 ANALYZER] ──► [M2 SKELETON] ──► [M3 NEURAL CORE] ──► [M4 BLEND]
        f0, voiced,        H1..H80 pobre     switch:               prior, mix,
        transient flag     + noise 8b        DSP / Live / Studio   envelope final
                                                                            │
                              [M5 CRIATIVE] ◄───────────────────────────────┤
                              sub, breath, morph, editor                    │
                                                                            ▼
                              OUT ◄── [M7 IO] ◄──── [M6 SYNTH] ◄────────────┘
                              gain, bypass        espectral (restauração)
                              pedal, meter        + aditiva (criação)
```

Ordem interna do M4 (contrato, não reordenar):
1. `neural = skeleton + prior · max(0, delta_rede)` nos harmônicos existentes; reconstruídos (ausentes) = `prior · h_rede`. Delta-only/preservação vale **só aqui**.
2. Operadores manuais de timbre (Brightness, Air, Even/Odd, Formant) sobre o resultado — podem cortar abaixo do input.
3. Mix (dry/wet espectral) entre skeleton e resultado.
Com prior=0 e timbre em 0, M4 é identidade exata (skeleton intacto).

Regra de fluxo: M5 (Criativo) nunca retroalimenta M3 (Neural Core). Operações criativas (sub, breath, morph, editor) são aplicadas pós-blend sobre o envelope final de M4, e passam diretamente ao sintetizador M6. No modo DSP (sem rede), M5 recebe o skeleton direto de M2 via M4 (prior=0). O knob Mix (RF7) opera no domínio espectral (sobre envelopes harmônicos), não no domínio temporal — evitando comb filtering.

### 2.2 Contrato de dados entre módulos (a linguagem interna do plugin)
```cpp
struct HarmFrame {
    float f0;            // Hz (0 = unvoiced)
    float confidence;    // 0..1 (clarity NSDF)
    uint8_t flags;       // bit0 voiced, bit1 sibilante, bit2 transiente
    uint8_t nValid;      // nº de harmônicos com n·f0 < 0.95·fs/2 (acima disso h = -100 e é ignorado)
    float h[80];         // dBFS de amplitude senoidal (seno full-scale = 0 dB), harmônicos 1..80  [-100, 0]
    float hz[80];        // freq. medida de cada pico (refino local), usada pela máscara do synth
    float noise[8];      // dBFS, bandas ERB do resíduo (piso -100, nunca negativo em linear)
};
// Frame rate: hop 256 @48 kHz = 5.33 ms/frame
// F0, h e noise referem-se ao MESMO instante (centro da janela STFT). Ver §2.3.
```
Diferença "ausente" vs. "acima de Nyquist": só `nValid` distingue. Rede, loss e synth usam `nValid` como máscara.
Todo módulo lê/escreve HarmFrame ou áudio — módulos são testáveis isoladamente contra esse contrato.

### 2.3 Decisões de motor (especificação dura)
| Subsistema | Decisão | Especificação |
|---|---|---|
| F0 | NSDF/MPM próprio | janela 2048 causal (amostras passadas), interpolação parabólica; voicing com histerese (entra clarity > 0.75, sai < 0.60). Mediana causal de 5 frames usada **só para rejeitar octave-jump** (decide a oitava), nunca como valor de F0 — a mediana atrasa ~2 frames e, em vibrato, desloca n·f0 por vários bins nos harmônicos altos |
| Alinhamento temporal | F0 e STFT no mesmo instante | janela F0 (2048) e janela STFT (1024) terminam na mesma amostra; o atraso residual (centro da 2048 = 512 amostras antes do centro da 1024) é absorvido pelo refino local de pico abaixo |
| STFT | janela 1024, hop 256, √Hann análise e síntese (soma de janelas constante → reconstrução perfeita com ganho 1) | latência algorítmica = 1024 amostras / 21.3 ms @48k (ver §2.5) |
| Extração harmônica | por harmônico n ≤ nValid: busca de pico em n·f0 ± f0/2, interpolação parabólica (freq. e magnitude); para F0 < 100 Hz (harmônicos não resolvidos pela Hann 1024), fallback de pico por bin | correção de magnitude pela janela; resíduo de ruído = |X|² − energia dos lóbulos harmônicos, clamp ≥ 0, em 8 bandas ERB |
| Resíntese restauração | Harmônico existente (input > −90 dB): ganho G_n = 10^((h_final − h_input)/20) aplicado aos bins do lóbulo do pico (±2 bins), fase original; resíduo multiplicado pelo ganho da banda ERB; harmônico ausente: oscilador aditivo, fase 2πn∫f0dt | com todos os ganhos = 1 o caminho é STFT→ISTFT identidade (P3 por construção). Histerese existente/ausente (−90/−84 dB) com crossfade de 20 ms para não piscar entre máscara e aditivo |
| Resíntese criação | banco aditivo (fase sintética 2πn∫f0dt) com portamento de amplitude ~15 ms | usado pelos controles do M5. Saída aditiva atrasada para o mesmo instante da saída OLA (soma alinhada à latência) |
| Neural Live | TCN causal: 1 bloco residual com 7 camadas (dilations {1,2,4,8,16,32,64}), k=3, GLU, residual + skip; **96 canais residuais** (≈ 6·C²·7 ≈ 390k params) | RF histórico: 255 frames ≈ 1.36 s. Se empilhar blocos, recalcular RF e buffers |
| Neural Studio | **mesma TCN causal**, treinada com alvo atrasado D = 16 frames (prevê o frame t−D vendo até t) | lookahead efetivo 85 ms; PDC = 1024 + 16·256 = 5120 amostras ≈ 107 ms. Mesmo código de inferência do Live, só outro arquivo de pesos e um delay de D frames no skeleton |
| Inferência | camadas `RTNeural::Conv1D` (stateful, dilatação nativa) ligadas à mão: GLU, residual e skip em ~50 linhas C++ | sem buffers circulares próprios (reduz R2). Pesos em JSON gerado pelo export |
| Condicionamento | log(f0) contínuo normalizado monotônico (sem wrap para preservar oitavas/registros vocais) + voiced flag + RMS dB | canais extras na entrada da rede |

### 2.4 I/O da rede (a especificação do "DLSS")
```text
ENTRADA (51 canais/frame):
  h[0..39]     dB dos 40 primeiros harmônicos RELATIVOS ao rms_db do frame
               (h − rms_db, invariante ao ganho de entrada/trim), com taper
               cossenoidal nos últimos 4 (mesma função no C++ e no Python).
               Normalizado: (clamp(h − rms_db, -100, 0)+100)/100
  noise[0..7]  idem, relativo ao rms_db (sem isso delta_noise não tem referência)
  log_f0_norm, voiced, rms_db (condicionamento contínuo)

SAÍDA (88 canais/frame), tudo relativo ao rms_db do frame:
  delta_h[0..39]          (dB) — correção residual somada ao skeleton existente
  h_reconstructed[40..79] (dB) — envelope ABSOLUTO dos harmônicos superiores (não delta:
                                 o input lá é -100, delta seria alvo gigante e instável)
  delta_noise[0..7]       (dB) — ajuste nas bandas de ruído ERB
Saídas com n > nValid são descartadas (synth não gera acima de Nyquist).
```
Regra de ouro (anti-robotizar): nos harmônicos existentes (<40), a rede aprende delta ≈ 0 caso a voz já seja rica; nos harmônicos ausentes (≥41), a injeção é proporcional ao envelope previsto e escalada pelo knob Prior. Degradação graciosa por construção.

### 2.5 Latency budget
| Componente | Amostras @48k | Tempo |
|---|---|---|
| STFT analysis (window−hop) | 768 | 16.0 ms |
| Output buffer (hop) | 256 | 5.3 ms |
| **Total algorítmico (reportado ao host)** | **1024** | **21.3 ms** |
| Margem até NFR1 | — | 3.7 ms |
| **Budget NFR1** | — | **25.0 ms** |
| Studio: + lookahead neural D=16 frames | +4096 | **5120 amostras ≈ 107 ms** |

Notas:
- Inferência (~1 ms) não é latência algorítmica: é custo de CPU dentro do bloco (entra no RTF, não no PDC). Somá-la no total confundia as duas coisas.
- O buffer de áudio do host (64–512) soma por fora e não é controlado pelo plugin; NFR1 mede só a latência reportada.
- A 44.1 kHz a latência em ms é a mesma (hop/janela escalados, §2.7).
- Plano B se precisar de folga: hop 128 (−2.7 ms, dobra carga).

### 2.6 Threading e Sincronização
- Audio thread: `processBlock` — zero allocation, lock-free, no blocking calls. `ScopedNoDenormals`.
- UI thread: JUCE timer at 60 fps reads SPSC ring buffer (`juce::AbstractFifo`) of `HarmFrame` for visualization. If full, audio thread drops the frame (never waits).
- Model swap (Live↔Studio): os dois modelos (~1.5 MB cada) são carregados e alocados em `prepareToPlay`, nunca no audio thread. A troca é só um índice atômico lido no início do bloco — nada é alocado, liberado ou parseado em tempo real (evita o problema de quem libera o modelo antigo).
- Mode switch sequence (audio thread): rampa prior→0 em 30 ms, troca índice, zera estado da TCN nova, rampa prior→valor em 300 ms (warm-up do RF). Latência nova é reportada com `setLatencySamples` + `updateHostDisplay`; hosts só reaplicam PDC no próximo stop/start — documentar.
- Parameter smoothing: `SmoothedValue` on audio thread, avançado por frame (hop) para parâmetros de envelope; UI reads atomic snapshots. Ao chegar no alvo o valor é exatamente o alvo (necessário para os testes bit-identical).

### 2.7 Estratégia de Sample Rate
- Processamento na taxa nativa do host, sem SRC. A rede vê features indexadas por harmônico (dB) e frames de duração fixa, que independem de fs se o tempo for preservado:
  - 48 kHz: janela 1024, hop 256 (referência).
  - 44.1 kHz: janela 941 (FFT 1024 com zero-pad), hop 235 → mesmos 21.3 ms / 5.33 ms.
- Motivo: SRC de entrada+saída com razão 147/160 exige FIFO extra (blocos de tamanho variável) e estourava a margem de 3.7 ms do NFR1.
- Outras taxas (88.2/96 kHz+): bypass atrasado (NFR4) + aviso na UI. Não crashar, não processar.
- Dados de treino: a 48 kHz. Gate de equivalência (§9 gate) roda também a 44.1 kHz: h[80] do mesmo áudio a 44.1 vs. 48 deve diferir < 0.5 dB nos harmônicos válidos.
- Nyquist: harmônicos com n·f0 ≥ 0.95·fs/2 ficam fora de `nValid` (a 48 kHz, F0=300 Hz → só H1..H76).

## 3. Estrutura do repositório
```text
larynx/
├── plugin/                      # JUCE + clap-juce-extensions
│   ├── CMakeLists.txt
│   └── source/
│       ├── PluginProcessor.{h,cpp}
│       ├── PluginEditor.{h,cpp}
│       ├── core/                # DSP puro, sem JUCE — testável headless
│       │   ├── harm_frame.h         # contrato (§2.2)
│       │   ├── analyzer.{h,cpp}     # M1: NSDF-F0, voicing, flags
│       │   ├── skeleton.{h,cpp}     # M2: STFT, extração harmônica
│       │   ├── neuralcore.{h,cpp}   # M3: wrapper RTNeural, switch modo
│       │   ├── blend.{h,cpp}        # M4
│       │   ├── creative.{h,cpp}     # M5
│       │   ├── synth.{h,cpp}        # M6: resíntese espectral + aditiva
│       │   └── smoothing.h          # one-pole/atomic p/ automação
│       ├── neural/
│       │   ├── model_live.json      # pesos TCN causal (export)
│       │   ├── model_studio.json
│       │   └── rtneural/*.h
│       └── ui/                  # LookAndFeel, knobs, harmongraph (OpenGL)
├── ml/                          # Python — fábrica de dataset e treino
│   ├── pycore/                      # pybind11 sobre plugin/source/core (analyzer + skeleton) — única fonte de features
│   ├── data/
│   │   ├── download_corpora.sh
│   │   ├── decompose.py             # pycore → shards base (WORLD só como comparação)
│   │   ├── masks.py                 # corrupções (§9.3)
│   │   ├── qc_spectral.py           # HNR, inarmônicidade, RT60, jitter
│   │   └── build_shards.py          # .npz + manifest.csv
│   ├── train/
│   │   ├── model.py                 # TCN causal (Live D=0, Studio D=16)
│   │   ├── train.py                 # PyTorch + AMP
│   │   └── export.py                # → model_live.json / model_studio.json
│   └── eval/
│       ├── metrics.py               # LSD, formants, jitter/shimmer
│       ├── abx.py                   # gera pares ABX
│       └── baseline.py              # interpolação não-neural
├── tools/
│   ├── bench_plugin/                # RTF/latência headless
│   └── abx_player/                  # CLI A/X/B cego → CSV
├── golden/                          # congelado no dia 1 — NUNCA treinar
│   ├── manifest.csv                 # origem de cada clipe (GTSinger, revisão fixada)
│   ├── 8_vozes/<voz>/{cantado,tecnica,fala}.wav + SHA256SUMS   # wavs fora do git; `python3 ml/data/fetch_golden.py`
│   └── protocolo_abx.md
├── docs/
│   ├── planner.md (este documento)
│   └── adrs/                        # decisões arquiteturais curtas
└── tests/                           # Catch2: core/ 100% coberto nos gates
```
Regra estrutural: `plugin/source/core/` não inclui nada de JUCE. Compila standalone (`tools/bench_plugin`) — isso torna o motor testável, benchmarkável e portável para sempre.

## 4. Ambiente
| Ferramenta | Versão/alvo | Uso |
|---|---|---|
| C++ | C++20, CMake ≥ 3.24 | plugin |
| JUCE | 8.x | framework |
| clap-juce-extensions | latest | alvo CLAP |
| Catch2 | v3 | testes |
| Python | 3.10 + conda | trilha ML |
| PyTorch | cu121, AMP fp16 | treino na 2080Ti |
| pyworld, librosa, torchcrepe, praat-parselmouth, pyloudnorm, soundfile | — | dados/eval |
| RTNeural | 1.x | inferência no plugin |

```bash
conda create -n larynx python=3.10 && conda activate larynx
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install pyworld librosa torchcrepe praat-parselmouth pyloudnorm soundfile mir_eval matplotlib jupyter
```

## 5. PASSO 1 — Casca do VST
Objetivo: o plugin existe, abre no DAW, dá bypass perfeito, tem a cara final da UI.

**Tarefas**
- Repo + CMake + JUCE + clap-juce-extensions compilando CLAP e VST3
- PluginProcessor: bus layout mono→mono, processBlock com bypass bit-perfect (cópia in→out atrasada pela latência reportada, sem denormais, sem alocação)
- Reportar a latência final (1024 amostras @48k) **já no P1**, com delay line no caminho de bypass: o host compensa desde o primeiro dia e nada muda de alinhamento quando o DSP real entrar no P3. Teste de bypass compara `out[n]` com `in[n − latência]`
- Transição de bypass com crossfade de 10 ms (sem clique); bit-perfect após o crossfade
- Tabela de parâmetros completa criada desde já (IDs estáveis, são contrato para automação):

| Param | ID | Range | Default |
|---|---|---|---|
| Mode | mode | 0=DSP,1=Live,2=Studio (não automatizável) | 0 |
| Prior | prior | 0–100% | 70 |
| Mix | mix | 0–100% | 100 |
| Brightness | bright | −12…+12 dB/oct | 0 |
| Air | air | −12…+12 dB | 0 |
| Even/Odd | evenodd | −1…+1 | 0 |
| Formant | formant | −6…+6 st | 0 |
| Sub | sub | 0–100% | 0 |
| Breath | breath | 0–100% | 0 |
| In/Out trim | in,out | −24…+24 dB | 0 |
| Bypass | bypass | bool | false |

- UI rack estática: 3 seções + seletor Live/Studio/DSP + bypass pedal + medidores; knobs giratórios custom (LookAndFeel), fundo fosco escuro, sem gradientes
- Gráfico OpenGL: janela desenhando 40 barras dummy a 60 fps (troca por dados reais no P4)
- Estado: getStateInformation salvando XML de todos params; testar roundtrip
- Smoothing: todos os params passam por SmoothedValue (tempo 30 ms)

**Gate**
- [ ] Carrega no Reaper e Cubase (CLAP + VST3)
- [ ] Bypass bit-perfect: diff entre out e in atrasado pela latência = 0 (teste automático)
- [x] Golden set congelado (SHA256SUMS + manifest commitados) antes de qualquer outra coisa — P2/P3 já dependem dele. Reconstruir/verificar: `python3 ml/data/fetch_golden.py`
- [ ] Automação de mix gravada e reproduzida sem zipper
- [ ] Sessão 1 h sem glitch; gráfico 60 fps estável
- [ ] UI aprovada por você (print salvo em docs/)

## 6. PASSO 2 — Analyzer (M1)
Objetivo: extrair F0/voicing/flags de qualidade em tempo real.

**Tarefas**
- analyzer: NSDF (algoritmo MPM) causal (amostras passadas), janela 2048, hop 256 alinhado ao STFT (§2.3); parabolic peak; mediana causal de 5 frames só para decisão de oitava; histerese de voicing (entra clarity>0.75, sai <0.60)
- Teste de vibrato: senóide com vibrato 6 Hz ±50 cents → erro de F0 instantâneo < 15 cents (pega o atraso da mediana se ela vazar para o valor)
- Classificador de não-processáveis: spectral flatness > limiar = sibilante; transiente (fluxo de energia + centroid jump) = percussivo → flag passthrough
- Teste unitário: senóides sintéticas 65–1000 Hz → erro < 10 cents; vogal sintética WORLD conhecida → F0 igual ao usado na síntese
- Bench headless: RTF do analyzer isolado

**Gate**
- [ ] F0 estável 30 s em vogal sustentada, sem octave-jump
- [x] Erro < 10 cents RMS vs. ground truth sintético (`tests/test_analyzer.cpp`: pior caso < 0.02 cents, 65–1000 Hz, 44.1/48k; vibrato 6 Hz ±50 cents < 5 cents)
- [ ] Sibilantes/transientes flagados ≥ 90% no golden set (anotar à mão 3 min)
- [x] RTF(analyzer) < 0.05 (≈ 0.02 no golden set, Release, `larynx_analyze`)

## 7. PASSO 3 — Síntese identidade (M6, parte 1) — o passo crítico
Objetivo: voz entra → decomposta → ressintetizada 1:1, transparente. Todo o projeto depende deste gate.

**Tarefas**
- skeleton: STFT 1024/256 √Hann; extração harmônica para HarmFrame (80 com `nValid`, refino de pico ±f0/2) + 8 bandas ERB de ruído (resíduo em energia, clamp ≥ 0)
- synth modo máscara (§2.3): ganhos por lóbulo harmônico e por banda ERB aplicados ao STFT original, fase original, ISTFT/OLA. Ganhos todos = 1 → identidade
- synth aditivo: osciladores de fase 2πn∫f0dt para harmônicos ausentes, alinhados à latência da OLA
- Teste de calibração (Catch2): aditivo gera H_n a X dB → skeleton re-extrai X ± 0.2 dB (a mesma escala de dB nos dois caminhos; sem isso a rede injeta no nível errado)
- Teste de identidade: ganhos = 1 → `out[n]` vs. `in[n − 1024]`, erro < −100 dBFS
- Caminho de teste no plugin: DSP mode = analyzer→skeleton→synth→out (sem rede)
- Plano B se a máscara gerar artefato em ganhos grandes: suavizar ganho entre frames (one-pole ~20 ms), limitar |ΔG| a ±12 dB por harmônico, alargar lóbulo para ±3 bins

**Gate**
- [x] Identidade numérica: ganhos = 1 → erro < −100 dBFS (deve passar por construção; se não passar é bug, não pesquisa) — ≤ −130 dBFS no golden, testes a 44.1/48k
- [ ] ABX cego próprio: 20 clipes (vogais sustentadas + frases), original vs. caminho DSP completo com ganhos = 1 → acerto ≤ 14/20 (não significativo, p ≥ 0.05)
- [ ] Escuta com máscara ativa (único teste real do R1 agora): Brightness ±6 e Even/Odd ±1 nos mesmos 20 clipes, checklist anotado sem metálico/flutter/pré-eco
- [x] LSD < 0.5 dB (harmônicos H1–H40) no caminho identidade — 0.00–0.01 dB no golden (`larynx_render`)
- [x] Roundtrip aditivo→skeleton dentro de ±0.2 dB (`tests/test_engine.cpp`)
- [x] Calibrar aqui o limiar de deriva de vogal (NFR5): medir F1/F2 antes/depois na ressíntese perfeita → esse desvio é o piso do sistema — `ml/eval/metrics.py` (Praat Burg, 10 ms, frames com pitch), 24 clipes do golden, 13 368 frames:
  - identidade: 0.00% (mediana e p95) — o caminho é numericamente transparente, então o piso real é o do medidor
  - piso do medidor (ruído branco somado): −90 dBFS → mediana 0.01%, p95 0.18%; −60 dBFS → mediana 0.2–0.3%, **p95 10–18%** (Praat troca de pico) → p95 não serve de métrica, só mediana
  - referência com máscara: Even/Odd ±1 dB → F1 2.3% / F2 0.7%; Brightness ±3 dB/oct → F1 8–11% / F2 2.7–3.0%; ±6 dB/oct → F1 16–26% / F2 6–9% (medianas)
  - **Limiar proposto para o P4: mediana F1 e F2 < 3%** (≈ 10× o piso do medidor, abaixo do JND de formante de ~3–5%). Pendente de decisão: Brightness muda o tilt e o LPC lê isso como F1 deslocado (8–11% em ±3) — ou Brightness sai do gate de deriva, ou o gate passa a ser escuta
- [x] RTF(synth) < 0.1 — engine inteiro 0.03–0.04 em identidade, ≤ 0.08 com aditivo pesado

Se não passar: não avançar. Iterar aqui (é o risco R1 do §16).

## 8. PASSO 4 — Controles harmônicos manuais (M2 + M4 DSP)
Objetivo: plugin útil sem ML. Os knobs do rack ganham função real.

**Tarefas**
- Operadores de envelope harmônico (em dB, sobre HarmFrame.h):
  - Brightness: tilt slope·20·log10(n) por índice n
  - Air: shelf suave a partir do harmônico mais próximo de 8 kHz (dependente de F0; para notas graves onde 80·f0 < 8 kHz, atua nas bandas de ruído ERB superiores)
  - Even/Odd: ganho alternado pares/ímpares via (−1)^n · k (ou n % 2 == 0 ? +k : -k)
  - Formant: reindexação do envelope com interpolação espectral (escala pelo fator α = 2^(formant/12), novo índice n' = n / α) — sem mexer em F0
- Blend (M4) na ordem do §2.1: (1) neural escalado por prior com preservação delta-only (só a contribuição da rede; neste passo, rede = 0); (2) operadores manuais — podem cortar abaixo do input; (3) mix espectral. Antes, `lerp(skeleton, rede_ou_manual, prior)` fazia Prior=0 desligar os knobs manuais e o delta-only impedia Brightness/Air negativos
- Gráfico real: barras input (apagado) vs. processado (aceso), F0/voicing ao vivo, leitor de vogal (F1/F2 → nearest vowel no mapa a/e/i/o/u)
- Metering in/out com ballistics VU (peak + rms)

**Estado (2026-10-30):** código pronto — `core/blend.h` (ordem do M4; Even/Odd ±1 = ±6 dB, H1 fixo; Air = shelf smoothstep de 1 oitava centrado em 8 kHz, também nas bandas ERB; Formant interpola o envelope em dB entre harmônicos; operadores só tocam harmônicos presentes no input), controles suavizados 30 ms por amostra e lidos por hop, gráfico com H1–H40 reais (input apagado, processado aceso, corte em contorno), F0 e leitor de vogal (2 maiores picos em 250–2500 Hz), VU rms + linha de pico. `larynx_render` usa o mesmo blend. Testes: identidade bit a bit com tudo em 0, Prior=0+Brightness=+6 altera, Even/Odd, Air, Mix, Formant +12 st, leitor de vogal.

**Gate**
- [ ] Soa como "EQ harmônico musical" — você usa numa música real por 1 semana
- [ ] Deriva de vogal < limiar do P3 em todos os knobs nos defaults+±3 — medido (golden, mediana F1/F2, limiar proposto 3%): Air ±3 dB 0.0/0.0% ✔; Brightness ±3 dB/oct 8–11% / 2.7–3.0% ✘; Even/Odd ±1 (±6 dB) 10–13% / 3.7–4.1% ✘; Formant ±3 st 14–16% / 16–17% (é a função do knob). **Decisão pendente:** Brightness e Even/Odd mudam o tilt/paridade e o LPC lê isso como formante deslocado; o gate como escrito só faz sentido para Air (e Prior/rede no P6). Opções: gate de deriva só para Air + rede, Brightness/Even-Odd por escuta; ou reduzir escala (Even/Odd ±1 = ±3 dB)
- [ ] Gráfico a 60 fps com dados reais; automação sem zipper
- [x] Prior=0 e timbre todo em 0 = bit-identical ao P3; Prior=0 com Brightness=+6 ainda altera o som (teste de regressão da ordem do M4) — `tests/test_blend.cpp`, `tests/test_processor.cpp`

## 9. PASSO 5 — Trilha de dados + treino (paralela aos passos 2–4)
Objetivo: o modelo da rede, com dados suficientes e avaliação honesta.

### 9.1 Fontes e papéis
| Fonte | Papel | Volume alvo |
|---|---|---|
| VocalSet (44.1 kHz) | vogais/técnicas rotuladas | ~10 h |
| VCTK (48 kHz) + LibriTTS-R (24 kHz) | diversidade de falantes/voicing | 20 h (subamostra) |
| JVS-MuSiC (24 kHz) | canto + F0 | 5 h |
| PTDB-TUG (48 kHz) | referência fonte glotal (EGG) | 1 h |
| ACE-Step a cappella | "cantores virtuais" diversos | 30–60 min gerados |
| GTSinger (48 kHz/24-bit, CC BY-NC-SA 4.0) | **corpus inteiro fora do treino**: 8 cantores = golden; 12 restantes = vozes novas (P6 A/B) e perfis de morph (P8) | 20 cantores |
| Golden set | teste — nunca treino | 8 vozes × 3 clipes (cantado, técnica, fala) |
| Suas gravações (opcional, futuro) | perfis de morph extras; nunca treino/val | — |

Regras de fonte:
- **Banda efetiva por clipe.** Fontes de 24 kHz não têm nada acima de 12 kHz; resamplear para 48k não cria agudo. Se entrarem como alvo sem máscara, a rede aprende a apagar H41–80. Cada clipe guarda `bw_hz` (sample rate original ou rolloff medido, o menor) no manifest; o loss ignora harmônicos com n·f0 > bw_hz.
- **MIR-1K removido.** 16 kHz sem alvo rico não forma par; a mask de lowpass (§9.3) já simula canal limitado.
- **Golden vem de corpus fora do treino.** Sem gravações próprias, o golden sai do GTSinger, e o GTSinger inteiro fica fora do treino (não só os 8 cantores do golden) — o teste de split vira trivial. Atribuição em `ml/data/credits.md`.
- Conferir sample rate real de cada corpus no download (os valores acima são os esperados) e anotar em `ml/data/credits.md`.

*(Roteiro de gravações: vogais sustentadas × extensão × dinâmica, escalas a cappella, sibilantes, fala)*

### 9.2 QC espectral (filtro do "certo") — qc_spectral.py
Rejeitar clipe se: HNR < 12 dB · inarmônicidade > limiar (energia fora de n·f0±δ) · RT60 estimado > 0.5 s (seco obrigatório; ACE-Step: gerar a cappella, nunca separar stems) · jitter/shimmer anormalmente baixos (voz "de IA morta") ou altos demais · clipping/loudness fora de −20±3 LUFS.

### 9.3 Fábrica de pares (a corrupção)
`decompose.py`: F0 para condicionamento da rede **deve** usar o mesmo NSDF/MPM do C++ de `plugin/source/core/analyzer`, e a extração harmônica o de `core/skeleton`, ambos via `ml/pycore` (pybind11) — obrigatório, implementações divergentes causam feature shift. Consequência de dependência: shards só podem ser gerados depois que analyzer (P2) e skeleton (P3) passarem seus gates (ver §18, P5b). Cada shard grava no manifest o hash git de `core/analyzer.*` + `core/skeleton.*`; mudou o hash → shards inválidos, regenerar antes de treinar. O consenso pYIN ∩ CREPE (discordância > 20 cents → descarta frame) é usado apenas como ground truth para QC/filtragem. Extração de picos STFT 1024 Hann idêntica ao C++ (via pybind11). Bandas de ruído ERB extraídas como resíduo direto: energia(sinal original − ressíntese harmônica pura) em 8 bandas ERB. Usa exatamente o processo do C++ (via pybind11), eliminando feature shift do D4C. WORLD/D4C disponível apenas como fallback de comparação. Shards base com h (80), noise (8), f0, voiced.

`masks.py` (por clipe, sorteada):
| Mask | Simula | Peso no treino |
|---|---|---|
| truncar em H40 com taper | padrão do plugin (sempre) | 100% dos clipes |
| truncar adicionalmente em H8–H32 aleatório | captação pobre | 40% |
| dropout de 3–5 harmônicos aleatórios | ressonâncias | 30% |
| soft clip antes da decomposição | saturação de entrada | 20% |
| lowpass 6–8 kHz antes da decomposição | canal limitado | 20% |

Alvo: sempre o envelope completo original (H1–80 + noise), mascarado por `nValid` e `bw_hz`.

Nota sobre a mask "truncar em H40": no plugin real a rede também só recebe H1..H40 (os H41+ reais existem mas não entram na rede; no M4 prevalece o maior entre input e reconstruído). A truncagem é a mesma função de taper no C++ e no Python.

### 9.4 Modelo e treino — model.py, train.py
- TCN conforme §2.3; duas variantes com a mesma arquitetura causal: Live (D=0) e Studio (alvo atrasado D=16 frames). Mesmo `export.py`, mesmo código C++
- Loss (tudo em dB relativo ao rms_db):
  - H1..40 e noise: MSE(delta_pred, h_full − h_input)
  - H41..80: MSE(h_pred, h_full) (absoluto, §2.4)
  - Pesos por banda: H1–8 ×2.0, H9–40 ×1.0, H41–60 ×0.8, H61–80 ×0.5, noise ×0.5
  - Máscara: zera termos com n > nValid, n·f0 > bw_hz, frames unvoiced (só noise conta), e alvo no piso −100 dB (senão o piso domina o MSE)
- Loss auxiliar (peso 0.1): MSE nos picos F1/F2 extraídos do envelope predito vs. target (Praat offline no dataset)
- Data augmentation de condicionamento: jitter uniforme ±15 cents no log_f0, dropout aleatório do voiced flag (5% dos frames voiced → unvoiced) — simula erros do detector NSDF em produção. Ganho de entrada aleatório ±12 dB por clipe (robustez a trim/nível, R15).
- Split por cantor; 80/10/10; golden set, GTSinger inteiro e suas gravações fora de tudo (só inferência). Checar no manifest: nenhum speaker_id aparece em mais de um split nem no golden (teste automático)
- AdamW, lr 3e-4, cosine, batch 64, AMP fp16
- Early stop no pior LSD por cantor no validation set (minimax), com restrição de que nenhum cantor tenha LSD > 2.0 dB.
- 2080Ti: 1–2 h por corrida → varrer hyperparâmetros (dilations, n_blocos, dropout)

### 9.5 Avaliação — eval/
- Baseline obrigatório (`baseline.py`): extrapolação do envelope por regressão de declínio espectral, sem rede. A rede só é aceita se vencer com folga — é aqui que se prova o DLSS
- LSD (H20–80), formant drift F1/F2 (Praat), jitter/shimmer delta ( Praat) — nos 8 golden voices
- ABX: `tools/abx_player` CLI A/X/B cego → CSV; original rico vs. reconstruído
- Ouvir 20 clipes do golden set manualmente (checkpoint de realismo)

### 9.6 Export — export.py
- Pesos → `model_live.json` (D=0) e `model_studio.json` (D=16) no formato RTNeural; JSON inclui `D`, `n_channels`, dilations e versão do contrato de I/O (51→88). O plugin recusa carregar arquivo com contrato diferente (fallback DSP)
- Teste de equivalência: mesmo input → saída PyTorch vs. saída RTNeural C++ headless, diff < 1e-4

**Gate**
- [x] Vence baseline com folga (LSD −3 dB ou mais) — **60k passos, 4 corpora: Live 7.26 vs 10.60 = +3.34 dB ✔, Studio 7.05 vs 10.60 = +3.55 dB ✔** (pior voz grave_baixo_es +2.2/+2.4; val 6.53 vs 12.18). Histórico: primeiro treino real (Live, só VocalSet 8.1 h, 20k passos): val (cantores VocalSet fora do treino) 7.84 vs baseline 14.00 dB; **golden (`ml/eval/eval_golden.py`) 8.77 vs 10.60 dB = +1.83 dB ✘**, baixos piores (grave_baixo_es −2.3 dB). Studio (D=16, mesmos dados): golden 8.47 vs 10.60 = +2.13 dB ✘. Softclip ensinava a rede a escurecer voz alta limpa (grave_baixo_es −11 dB em H20–40 intactos) → softclip fora por padrão + ganho ±24 dB. 4 corpora (VocalSet 8.1 + LibriTTS-R 8.5 + PTDB 7.5 + VCTK 5.8 h), 20k passos: Live +2.24, Studio +2.50 dB ✘ (grave_baixo_es −0.3/−0.6: rede soma 1–2 dB de erro em harmônicos presentes). Val ainda caindo → 60k passos (acima).
- [ ] Preserva vogal e vida nos 8 golden voices (NFR5/6)
- [ ] ABX próprio no modo Studio: acerto ≤ 65% (original rico vs. restaurado)
- [x] Export equivalente em C++ validado — `ml/train/check_export.py`: modelo treinado 5.7e-5 vs PyTorch float64 (o próprio float32 do PyTorch erra 1.7e-4)
- [x] Para 100 frames aleatórios do dataset, a extração harmônica via pybind11 e via C++ headless (tools/bench_plugin) produzem h[80] com diff < 0.01 dB. Se não, consertar antes de treinar. — `ml/pycore/check_equivalence.py`: pior 0.0001 dB (arredondamento do CSV) no golden; rodar de novo sobre os shards reais
- [ ] DatasetQC: taxa de descarte documentada; manifest com sha256

## 10. PASSO 6 — Neural Core Live (M3)
Objetivo: o DLSS vocal funcional, em tempo real, dentro do plugin.

**Tarefas**
- neuralcore: inferência TCN causal com delay buffers circulares por camada; buffer circular de HarmFrame; condicionamento (log_f0/voiced/rms) concatenado
- Modo Live no switch: causal, frame-a-frame, sem lookahead
- Latência medida e exibida no plugin (rótulo honesto)
- Denormals: juce::ScopedNoDenormals (multiplataforma: compatível com x86 SSE e ARM Apple Silicon); afinar se preciso com buffer de 512
- Benchmark: `tools/bench_plugin` — RTF total do chain (analyzer→neural→synth) em CPU de laptop (alvo: i5-8250U classe)

**Estado (2026-10-30):** inferência própria em `core/neuralcore` (sem RTNeural: conv causal com histórico circular por camada, pesos reorganizados por tap), carregada de `plugin/source/neural/model_{live,studio}.json` embutidos via BinaryData; contrato de I/O conferido, arquivo inválido → DSP. `ml/train/check_export.py`: C++ vs PyTorch 4e-7 (gate §9.6). Live e Studio no processor: Studio roda um segundo motor com entrada atrasada D hops (latência 1024 + D·hop, bypass com o mesmo atraso; testado a 44.1/48k), troca de modo reconfigura com o callback suspenso e rampa Prior 0→1 em 300 ms. RTF (teste `[rtf]`, TCN 96 ch × 7, nesta máquina): Live 0.12, Studio 0.16. Falta: pesos treinados, medir em laptop, sessão de 1 h, A/B nos 12 cantores GTSinger.

**Gate**
- [ ] RTF total < 0.2 sem GPU; latência total ≤ 25 ms
- [ ] Sessão 1 h sem dropout; automação de prior durante neural ativa sem glitch
- [ ] Comparação A/B no DAW: DSP mode vs. Live mode audível e positiva em vozes novas (os 12 cantores do GTSinger fora do golden)

## 11. PASSO 7 — Modo Studio (M3, variante 2)
**Tarefas**
- Mesmo motor do Live com `model_studio.json` (D=16) e skeleton atrasado D frames para alinhar com a saída da rede; setLatencySamples(5120) → host compensa (PDC). Antes: "bidirecional ±0.5 s em blocos de 2048" exigiria ≥ 0.5 s de PDC, incompatível com os 80–120 ms prometidos
- Troca de modelo em runtime pela sequência do §2.6 (rampa de prior, índice atômico, warm-up)
- Indicador visual: Live (LED âmbar) / Studio (LED ciano) + latência real

**Gate**
- [ ] Studio > Live no ABX (a latência compra melhoria audível no topo)
- [ ] PDC correto: bounce no Reaper/Cubase alinhado (teste com click track)
- [ ] Trocar Live↔Studio durante playback sem pop (alinhamento correto só após stop/start do transporte — limitação de host, documentada no manual)

## 12. PASSO 8 — Camada criativa (M5)
Objetivo: o instrumento experimental.

**Tarefas**
- Sub: banco aditivo injeta sub-oitavas musicais em f0/2 e f0/4 com envolvente derivada de H1–H4 (ganho sub); fase sintética coerente
- Breath: ganho no canal de ruído (bandas ERB) com detector de vogal aberta (mais sopro nas abertas — realismo)
- Morph (v2): Perfis extraídos: média geométrica (em dB) dos envelopes harmônicos por combinação vowel×pitch_bin (bins de 2 semitons) dos cantores do GTSinger fora do golden (ou gravações próprias, se surgirem). Espaço de morphing: PCA dos envelopes (primeiros 8 componentes), interpolação esférica (slerp) entre perfis no espaço PCA. Formato: .data binário com header descrevendo dimensões. Escalabilidade: novos perfis podem ser adicionados por qualquer usuário gravando o roteiro de 15 min (§9.1).
- Editor de envelope no gráfico: arrastar barras H1–80; overlay do neural em trilha fantasma; duplo-clique = reset
- Presets de fábrica: Studio Restore · Telefone→Estúdio · Growl suave · Sopro etéreo · Duplo virtual (sub+detrune leve de fase, sem pitch) · Pass Through
- Modo Studio + editor = render "offline" com PDC

**Gate**
- [ ] 1 música sua real produzida usando só os controles criativos
- [ ] Editor: arrastar sem zipper; undo funcional (opcional mas desejado)
- [ ] Presets soam como prometem (auditoria cega por 1 amigo)

## 13. PASSO 9 — Fechamento de qualidade
**Tarefas**
- Golden set (congelado no P1 — aqui só auditar SHA256 com `fetch_golden.py`, não recriar): 8 cantores do GTSinger — 2 baixos (IT, ES), soprano (FR), soprano soprosa (KO), alto (RU), alto soprosa (EN), tenor com falsete (EN), tenor faríngeo (DE). Lacuna conhecida: sem voz infantil e sem rouquidão patológica (opcional: vogais sustentadas da Saarbrücken Voice Database). Protocolo em `golden/protocolo_abx.md`
- ABX com 3+ ouvintes externos: NFR7 (acerto ≤ 65% Studio; ≤ 70% Live)
- Taxas não suportadas (88.2/96k): bypass atrasado + aviso, sem crash
- Edge cases: voz fora da distribuição → Prior baixo degrada graciosamente (validar); DC offset; clipping de entrada; sample rate 44.1k; silêncio; unvoiced longo
- Docs: manual de 2 páginas, README com gifs, vídeo de 60 s
- Build final assinado (ad-hoc no macOS), instalador zip

**Gate**
- [ ] Todos os NFRs verdes em tabela final
- [ ] 1 pessoa externa usa o plugin sem ajuda
- [ ] Tag v0.1 no repo

## 14. Especificação de UI (transversal — construída em P1, vivida em P4/P8)
```text
900 × 540 px · fundo #14161A · painéis #1D2025 · acento âmbar #E8A33D (Live) / ciano #4DC9D6 (Studio)
├── Cabeçalho: nome + toggle [DSP | LIVE | STUDIO] + LED de latência + CPU%
├── Seção RESTAURAR: Prior, Mix (+ botão "solo neural")
├── Seção TIMBRE: Brightness, Air, Even/Odd, Formant
├── Seção CRIAR: Sub, Breath, Morph (P8)
├── Gráfico OpenGL central: barras H1–40 (input apagado / processado aceso),
│   cursor de F0, etiqueta de vogal detectada, overlay de perfil (P8, arrastável)
└── Rodapé: IN/OUT trim + meters, bypass pedal-style
```
Knobs: LookAndFeel custom — arco de LED, marcador de valor, tooltip com valor exato; scroll fino com Shift. Automação visível (arco pulsa quando host automatiza).

## 15. Testes e qualidade (transversal)
| Camada | Teste | Frequência |
|---|---|---|
| Unit (Catch2) | F0 sintético, roundtrip decompose→synthesize (LSD), masks, blend bit-exact, smoothing | CI local a cada push |
| Equivalência ML | PyTorch vs. RTNeural < 1e-4 | a cada export |
| Bench headless | RTF por módulo e total | a cada P6+ |
| Bypass | diff in/out = 0 | a cada build |
| Regressão de audição | 10 clipes fixos ouvidos após mudanças grandes de synth/neural | semanal na prática |
| Regressão áudio | 3 clipes golden renderizados, LSD comparado com referência salva, falha se degradar > 0.3 dB | CI a cada push (P3+) |
| Equivalência C++↔Python | 100 frames aleatórios, diff extração < 0.01 dB | CI a cada push |
| Identidade synth | ganhos = 1 → out vs. in atrasado < −100 dBFS | CI a cada push (P3+) |
| Ordem do M4 | prior=0 + timbre 0 = identidade; prior=0 + Brightness ≠ 0 altera saída | CI a cada push (P4+) |
| Split de dados | nenhum speaker_id em dois splits nem no golden | a cada build de shards |
| Benchmark gate | RTF total < threshold (0.2 Live, 0.4 Studio), falha se exceder | CI a cada push (P6+) |
| ABX formal | protocolo do golden set | P9 |

## 16. Riscos e planos B
| # | Risco | Sinal de alarme | Plano B |
|---|---|---|---|
| R1 | Artefato da máscara espectral com ganhos grandes (era: ressíntese metálica — reduzido pela fase original) | escuta do P3 com Brightness ±6 falha | suavizar ganho entre frames; limitar ±12 dB/harmônico; janela 2048 (+10 ms, só Studio); último recurso: aditivo com fase extraída |
| R2 | Divergência numérica RTNeural (Conv1D stateful + GLU/residual manuais) vs. PyTorch | diff > 1e-4 no teste de equivalência | checar ordem de kernel/dilatação no export primeiro (erro mais comum); depois simplificar topologia |
| R3 | Overfit a corpora (poucos cantores) | val-LSD bom, vozes novas ruins | augmentation (EQ/reverb curto/ruído), ACE-Step diversidade, Prior como escape já desenhado |
| R4 | Vogais artificializadas | deriva F1/F2 | delta-only + peso maior nos H baixos + treino com loss de formante (opcional) |
| R5 | CPU estourar em laptops velhos | RTF > 0.2 | reduzir dilations, quantizar pesos (int8), hop 512 (latência +10 ms) |
| R6 | Motivação (projeto longo, solo) | semanas sem commit | regra dos gates: cada passo termina com algo audível no DAW; P4 já é plugin útil — pode "morar" lá semanas sem culpa |
| R7 | ACE-Step degrada dataset | QC rejeita > 60% | reduzir peso da fonte; QC é soberano |
| R8 | Feature shift Python↔C++ | Diff extração > 0.01 dB no gate de equivalência | Forçar pybind11 único; se impossível, treinar com noise augmentation nos harmônicos (±0.5 dB uniforme) |
| R9 | F0 atrasado em relação ao STFT (mediana causal + centro da janela 2048) | em vibrato, picos de H20+ perdidos; RF11/NFR6 falham | F0 só para região de busca, freq. real de cada pico por refino local (§2.3); teste de vibrato no P2 |
| R10 | Fontes de 24 kHz ensinam a rede a apagar agudos | H41–80 previstos sempre baixos; LSD H41+ ruim | `bw_hz` por clipe no manifest + máscara no loss (§9.1) |
| R11 | Vazamento de cantor entre treino e golden | golden muito melhor que vozes novas | golden de corpus inteiro fora do treino (GTSinger); teste automático de split |
| R12 | Shards gerados com core C++ desatualizado | extração no plugin ≠ treino após mudança no P3/P4 | hash do core no manifest; regenerar shards ao mudar |
| R13 | Troca de modelo/latência em playback | pop, desalinhamento, alocação no audio thread | modelos pré-carregados, índice atômico, rampa de prior; PDC novo só após stop/start |
| R14 | Gate P5 parado esperando o C++ | trilha ML "paralela" sem features para treinar | P5a (dados, QC, baseline, treino com features WORLD só para depurar o pipeline) já; P5b (shards reais) após P3 |
| R15 | Ganho de entrada muda a resposta da rede | Trim/volume do cantor muda o timbre restaurado | features relativas ao rms_db (§2.4) + augmentation de ganho ±12 dB |
| R16 | Host em 88.2/96 kHz | crash ou som errado | bypass atrasado + aviso (§2.7) |

## 17. Licenças (checklist)
| Item | Licença | Ação |
|---|---|---|
| JUCE | GPL/comercial | projeto pessoal não distribuído comercialmente = ok; se um dia vender, licença paga |
| CLAP / clap-juce-extensions | MIT | ok |
| RTNeural | MIT | ok |
| WORLD (pyworld) | MIT (código) | ok — só para pipeline de dados |
| VST3 SDK | GPLv3/proprietário Steinberg | usar conforme termos atuais; CLAP cobre o risco |
| Corpora | variadas | conferir cada uma no download; anotar em `ml/data/credits.md` |
| ACE-Step | confirmar no model card da v1.5 | uso pessoal/pesquisa; letras próprias apenas |
| Suas gravações / amigos | — | consentimento escrito simples |

## 18. Ordem de Execução e Dependências (Passo a Passo)

A progressão do projeto é estritamente guiada pelo grafo de dependências técnicas e validação de gates funcionais (sem estimativas de calendário):

**Trilha Principal: Motor DSP e Plugin (Sequencial / Caminho Crítico)**
1. **Passo 1 — Casca do VST**
   - Gate: Carrega no DAW (CLAP/VST3), bypass bit-perfect verificado (diff=0), UI estática aprovada.
2. **Passo 2 — Analyzer (M1)**
   - Pré-requisito: Passo 1
   - Gate: F0 estável (erro < 10 cents, sem octave jump), sibilantes/transientes flagados, RTF < 0.05.
3. **Passo 3 — Síntese Identidade (M6, parte 1) [Ponto Crítico de Inflexão]**
   - Pré-requisito: Passo 2
   - Gate: identidade numérica (< −100 dBFS), ABX acerto ≤ 14/20, LSD < 0.5 dB, máscara ativa sem artefato na escuta. Se não passar, o projeto não avança.
4. **Passo 4 — Controles Harmônicos Manuais (M2 + M4 DSP) [Marco: VST Útil sem ML]**
   - Pré-requisito: Passo 3
   - Gate: Plugin utilizável musicalmente no DAW como EQ harmônico; automação sem zipper; deriva de vogal controlada.
5. **Passo 6 — Neural Core Live (M3)**
   - Pré-requisito: Passo 4 + pesos exportados do Passo 5b
   - Gate: RTF < 0.2 em CPU, latência Live <= 25 ms, sessão estável sem glitches, teste A/B positivo.
6. **Passo 7 — Modo Studio (M3, variante 2)**
   - Pré-requisito: Passo 6 + pesos Studio (D=16) do Passo 5b
   - Gate: Latência compensada por PDC sem desalinhamento, troca Live <-> Studio sem estalos, melhoria audível no ABX.
7. **Passo 8 — Camada Criativa (M5)**
   - Pré-requisito: Passo 4 e Passos 6/7
   - Gate: Sub, Breath, Morphing e editor interativo no gráfico funcionando musicalmente sem artefatos.
8. **Passo 9 — Fechamento de Qualidade**
   - Pré-requisito: Passos 7 e 8
   - Gate: Golden set formal com ouvintes externos aprovado, edge cases validados, build final v0.1.

**Trilha Paralela: Dados e Machine Learning**
- **Passo 5a — Dados e pipeline (imediato, paralelo aos Passos 1–3)**
   - Download, licenças, QC espectral, `bw_hz` por clipe, split por cantor, `baseline.py`, `model.py`/`train.py`/`export.py` depurados com features provisórias (WORLD). Modelos daqui são descartáveis — só provam o pipeline.
   - **Estado (2026-10-30):** WORLD pulado — `ml/pycore` já expõe analyzer + skeleton + `core/features.h` (entrada da rede definida uma vez, em C++), então o pipeline foi feito direto sobre as features reais. Rodando ponta a ponta (smoke): `build_shards.py` (decompose + QC num arquivo só; sem RT60 e sem inarmonicidade, ver docstring), `masks.py`, `baseline.py`, `model.py` (TCN 402k params + export), `train.py`. Falta: corpora (VocalSet/VCTK baixando; LibriTTS-R, JVS-MuSiC, PTDB-TUG manuais), loss auxiliar F1/F2. Se o core mudar após o gate de escuta do P3, o hash do core invalida os shards.
- **Passo 5b — Shards reais e treino (após gate do Passo 3)**
   - Pré-requisito: `core/analyzer` (P2) e `core/skeleton` (P3) congelados e expostos via `ml/pycore`.
   - Foco: shards com hash do core, treino Live (D=0) e Studio (D=16), superação obrigatória do baseline.
   - Ponto de Convergência: export (`model_live.json` / `model_studio.json`) desbloqueia os Passos 6 e 7. Mudança posterior no core = regenerar shards e retreinar.

Regra de sobrevivência do projeto: nenhum passo termina sem você ter ouvido o resultado no DAW. O P4 já te dá um plugin utilizável — todo o resto é upgrade incremental.
