"""Before anything is made public: does it use a word that comes from the private documents?

    git diff --cached | python tools/privcheck.py private/drawings -
    python tools/privcheck.py private/drawings README.md viewer/index.html
    python tools/privcheck.py private/drawings --ok private/privcheck-ok.txt --learn $(git ls-files)

This repository is public and is worked on beside documents that are not: drawings with
the name of the house, the address, the people who drew them and map references in
their title blocks. The risk is that one of those words is typed into a note, a commit
or the page. This reads every PDF and text file in the folder given, and the names of
the files in it, and lists what the text being checked has in common with them: words of
four letters or more and numbers of five digits or more. Of a diff, only the lines it
adds are read. A drawing's lettering is sometimes drawn as shapes, which no reader can
see as words: put anything that must never appear (a name, a street, a village) in a
text file in the same folder, one word to a line, and it is checked for as well.

Most of what the two share is innocent ("garden", "window", "kitchen"). --ok names a
list of words already looked at and passed, one to a line; they are not reported again,
so what is printed is only what is new, and a name stands out in a short list. --learn
adds everything found to that list instead of reporting it: do that once, after reading
the list it prints, and never as a way to make a warning go away.

Exit status 1 if anything is reported. It is an aid to reading, not a filter: it cannot
see a description that gives the place away without using a single word of the drawings.
"""
import os
import re
import sys

WORD = re.compile(r"[a-z][a-z']{3,}|\d{5,}")
GRID_REF = re.compile(r"\b[a-z]{2}\s?\d{4,}\s?\d{4,}\b")          # a map reference: two letters and two groups of digits


def tokens(text):
    low = text.lower()
    return set(WORD.findall(low)) | {re.sub(r"\s+", "", m) for m in GRID_REF.findall(low)}


def pdf_text(path):
    """A PDF's text read two ways: one keeps words apart that the other runs together, and the other way about."""
    import logging
    from pypdf import PdfReader
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    out = []
    for page in PdfReader(path).pages:
        out.append(page.extract_text() or "")
        try:
            out.append(page.extract_text(extraction_mode="layout") or "")
        except Exception:
            pass
    return " ".join(out)


def words_of_folder(folder):
    found = set()
    for root, _, files in os.walk(folder):
        for f in files:
            found |= set(WORD.findall(re.sub(r"[^A-Za-z0-9']+", " ", os.path.splitext(f)[0]).lower()))
            path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            try:
                if ext == ".pdf":
                    text = pdf_text(path)
                elif ext in (".txt", ".md", ".csv", ".tsv", ".json"):
                    with open(path, errors="ignore") as fh:
                        text = fh.read()
                else:
                    continue
            except Exception as e:
                print(f"   could not read {f}: {type(e).__name__}", file=sys.stderr)
                continue
            found |= tokens(text)
    return found


def main():
    a = sys.argv[1:]
    ok_path, learn = None, False
    if "--ok" in a:
        i = a.index("--ok")
        ok_path = a[i + 1]
        del a[i:i + 2]
    if "--learn" in a:
        a.remove("--learn")
        learn = True
    if len(a) < 2:
        raise SystemExit(__doc__)
    private = words_of_folder(a[0])
    ok = set()
    if ok_path and os.path.exists(ok_path):
        with open(ok_path) as fh:
            ok = {line.strip().lower() for line in fh if line.strip() and not line.startswith("#")}
    hits = {}
    for src in a[1:]:
        if src == "-":
            text = sys.stdin.read()
        elif os.path.isfile(src):
            with open(src, errors="ignore") as fh:
                text = fh.read()
        else:
            continue
        if src == "-" or src.endswith((".diff", ".patch")):
            text = "\n".join(line[1:] for line in text.split("\n") if line.startswith("+") and not line.startswith("+++"))
        for w in tokens(text) & private:
            hits.setdefault(w, set()).add("the text piped in" if src == "-" else src)
    new = {w: s for w, s in hits.items() if w not in ok}
    print(f"{len(private)} words in {a[0]}; {len(hits)} of them in the text checked; {len(new)} not passed before", file=sys.stderr)
    if learn:
        if not ok_path:
            raise SystemExit("--learn needs --ok <file> to write to")
        with open(ok_path, "w") as fh:
            fh.write("# words that the public text shares with the private documents, read and passed as harmless\n")
            fh.write("\n".join(sorted(ok | set(new))) + "\n")
        print(" ".join(sorted(new)))
        print(f"{len(new)} words added to {ok_path}", file=sys.stderr)
        return
    for w in sorted(new):
        where = sorted(new[w])
        print(f"{w:24s} {', '.join(where[:3])}" + (f" and {len(where) - 3} more" if len(where) > 3 else ""))
    sys.exit(1 if new else 0)


if __name__ == "__main__":
    main()
