# LARYNX — manual

Reconstrução harmônica vocal em tempo real. Entrada e saída mono, 44.1 ou 48 kHz (outras taxas: passa direto, atrasado).

## Modos (cabeçalho)

| Modo | O que faz | Latência |
|---|---|---|
| DSP | Só os controles manuais sobre o envelope harmônico. Sem rede. | 21.3 ms |
| LIVE | Rede causal reconstrói/realça harmônicos do frame atual. | 21.3 ms |
| STUDIO | Mesma rede vendo 16 frames à frente (85 ms); mais precisa. O host compensa (PDC). | 106.7 ms |

Sem pesos treinados embutidos, LIVE/STUDIO rodam como DSP. A troca de modo muda a latência: alguns hosts só
reaplicam o PDC no próximo stop/play.

## Controles

**RESTAURAR**
- **Prior** (0–100%): quanto da sugestão da rede entra. 0 = rede desligada.
- **Mix** (0–100%): mistura em dB entre o envelope original e o processado (operadores manuais incluídos).

**TIMBRE** (só agem em harmônicos que existem)
- **Brightness** (±12 dB/oct): inclinação a partir de H1.
- **Air** (±12 dB): prateleira suave em 8 kHz, harmônicos e ruído.
- **Even/Odd** (±1 = ±6 dB): pares sobem e ímpares descem (ou o inverso). H1 fica fixo.
- **Formant** (±6 st): desloca o envelope sem mudar o F0.

**CRIAR**
- **Sub** (0–100%): sub-oitavas f0/2 e f0/4 com a envolvente de H1–H4.
- **Breath** (0–100%): até +12 dB nas bandas de ruído em trechos vozeados, mais em vogais abertas.
- **Morph** (0–100%) + voz: puxa o envelope para o perfil de um dos 12 cantores de referência, por vogal e altura.

**Rodapé**: IN/OUT trim (±24 dB), medidores (barra = VU rms, linha = pico), pedal de bypass (crossfade de 10 ms,
depois cópia bit a bit da entrada atrasada).

## Gráfico

Barras H1–H40: apagado = entrada, aceso = saída; contorno = o que foi cortado; traço fino = o que a rede pede.
Canto: F0, vogal detectada, F1/F2.

**Editor**: arraste uma barra para cima/baixo (±24 dB) ou arraste através de várias para desenhar. Duplo clique
zera a curva. Ctrl+Z / Ctrl+Shift+Z desfazem e refazem. A curva é salva com o projeto.

## Presets

Pass Through · Studio Restore · Telefone → Estúdio · Growl suave · Sopro etéreo · Duplo virtual.
Escolher um preset reajusta todos os parâmetros. A curva do editor continua como estava.

## Limites conhecidos

- Voz solo, sem acompanhamento. Polifonia e reverb longo confundem o detector de F0.
- Valores extremos (Brightness +12 em material saturado) podem passar de 0 dBFS. Use o OUT trim.
