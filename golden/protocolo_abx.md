# Protocolo ABX

Teste cego próprio, A = original, B = processado, X = um dos dois sorteado a cada par.

1. Fones fechados, volume fixo do início ao fim, sala silenciosa. Sem olhar forma de onda nem espectro.
2. Preparar os pares (20 clipes do golden, 8 s cada, cobrindo cantado, técnica e fala):
   ```bash
   python3 ml/eval/abx.py prepare abx/identidade                  # P3: caminho DSP com ganhos = 1
   python3 ml/eval/abx.py prepare abx/bright6 -- --bright 6        # P3: escuta com máscara ativa
   ```
3. Rodar: `python3 ml/eval/abx.py run abx/identidade --name <você>`. Ouvir A, B e X quantas vezes quiser; responder A ou B.
4. Critério P3 (identidade): acerto ≤ 14/20 (p ≥ 0.05, não distinguível). P5 (Studio): acerto ≤ 65%.
5. Escuta com máscara ativa (Brightness ±6, Even/Odd ±1): não é ABX; anotar por clipe metálico / flutter / pré-eco em
   `abx/<pasta>/notas.md`.
6. Uma sessão por condição e por dia (fadiga muda o resultado). O CSV fica em `abx/<pasta>/results_<nome>_<data>.csv`.
