import json

for n in [1, 2, 3]:
    print(f"\n{'='*55}\nPIPELINE {n} - judge verdicts + any error text\n{'='*55}")
    with open(f"results/pipeline_{n}_scored.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    pass_c = fail_c = err_c = 0
    first_error_shown = False
    for d in data:
        v = d.get("llm_judge_verdict", "?")
        if v == "PASS": pass_c += 1
        elif v == "FAIL": fail_c += 1
        else:
            err_c += 1
            if not first_error_shown:
                # The raw error was stored in llm_judge_verdict or a reasoning field
                print(f"  First ERROR at {d.get('id')}: verdict field = {repr(v)}")
                first_error_shown = True
    print(f"  PASS={pass_c}  FAIL={fail_c}  ERROR/other={err_c}")
    print(f"  Valid (non-error) verdicts: {pass_c + fail_c}/30")