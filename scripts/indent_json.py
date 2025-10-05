
#!/usr/bin/env python3
import sys, json, pathlib

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    root = pathlib.Path(args.run_dir)
    for p in root.rglob("*"):
        if p.suffix.lower() in [".json", ".jsonl"]:
            try:
                if p.suffix.lower() == ".jsonl":
                    # re-write as JSONL but with each line pretty-printed JSON
                    lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
                    with p.open("w", encoding="utf-8") as f:
                        for obj in lines:
                            f.write(json.dumps(obj, indent=2, ensure_ascii=False))
                            f.write("\n")
                else:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
                print("indented:", p)
            except Exception as e:
                print("skip:", p, "->", e)

if __name__ == "__main__":
    main()
